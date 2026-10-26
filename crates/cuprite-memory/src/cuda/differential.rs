//! Randomized GPU-versus-CPU comparison. Every case runs the GPU path and
//! a CPU reference on the same input and requires bit-identical results.
//! On failure, proptest shrinks the input to a minimal reproduction.

use std::cell::RefCell;

use cudarc::driver::CudaFunction;
use proptest::{
    prelude::*,
    test_runner::{Config, TestRunner},
};

use super::{
    fixture::{add_seven, launch_in_session, lcg64, lcg_step, LCG_ROUNDS},
    CudaBackend,
};
use crate::{manager::Manager, MapStep, MemoryConfig};

#[derive(Clone, Copy, Debug)]
enum Kernel {
    AddSeven,
    Lcg,
}

impl Kernel {
    fn apply_cpu(self, values: &mut [u32], reps: usize) {
        for value in values {
            for _ in 0..reps {
                *value = match self {
                    Kernel::AddSeven => value.wrapping_add(7),
                    Kernel::Lcg => (0..LCG_ROUNDS).fold(*value, |x, _| lcg_step(x)),
                };
            }
        }
    }
}

struct Gpu {
    manager: Manager<CudaBackend>,
    add_seven: CudaFunction,
    lcg: CudaFunction,
}

impl Gpu {
    fn open() -> Self {
        let config = MemoryConfig::default();
        let manager = Manager::new(CudaBackend::open(config).unwrap(), config);
        let stream = &manager.backend.stream;
        let (add_seven, lcg) = (add_seven(stream), lcg64(stream));
        Self {
            manager,
            add_seven,
            lcg,
        }
    }

    fn function(&self, kernel: Kernel) -> CudaFunction {
        match kernel {
            Kernel::AddSeven => self.add_seven.clone(),
            Kernel::Lcg => self.lcg.clone(),
        }
    }
}

fn runner(cases: u32) -> TestRunner {
    TestRunner::new(Config {
        cases,
        // Tests built from a manual runner have no source file to save to.
        failure_persistence: None,
        // Vectors shrink one deleted element per step, so the default budget
        // (4 x cases) cannot reduce a failing 5000-element slice. Only
        // failing runs pay for this.
        max_shrink_iters: 100_000,
        ..Config::default()
    })
}

/// Values weighted toward zero and the top of the range, where wrapping
/// arithmetic overflows.
fn element() -> impl Strategy<Value = u32> {
    prop_oneof![
        3 => any::<u32>(),
        1 => Just(0u32),
        1 => (u32::MAX - 8)..=u32::MAX,
    ]
}

/// Lengths weighted toward edge cases: tiny slices and both sides of
/// 256-thread block boundaries, plus anything up to a few blocks. Each
/// range is its own vector strategy so a failure shrinks to a short slice
/// (a length chosen by `prop_flat_map` would not shrink).
fn values() -> impl Strategy<Value = Vec<u32>> {
    let vec = |len| prop::collection::vec(element(), len);
    prop_oneof![vec(0..=4), vec(250..=262), vec(1020..=1030), vec(0..=5000)]
}

fn kernel() -> impl Strategy<Value = Kernel> {
    prop_oneof![Just(Kernel::AddSeven), Just(Kernel::Lcg)]
}

#[test]
fn kernels_match_the_cpu_on_random_inputs() {
    // The runner reruns cases while shrinking, so it needs an `Fn` closure.
    let gpu = RefCell::new(Gpu::open());
    let strategy = (kernel(), values(), 1..=8usize);

    runner(256)
        .run(&strategy, |(kernel, input, reps)| {
            let gpu = &mut *gpu.borrow_mut();
            let function = gpu.function(kernel);
            let mut expected = input.clone();
            kernel.apply_cpu(&mut expected, reps);

            let mut values = input;
            let mut session = gpu.manager.begin_mut(&mut values).unwrap();
            for _ in 0..reps {
                launch_in_session(&mut session, &function);
            }
            session.finish().unwrap();

            prop_assert_eq!(values, expected);
            Ok(())
        })
        .unwrap();
}

#[test]
fn map_steps_match_the_cpu_on_random_inputs() {
    let gpu = RefCell::new(Gpu::open());
    let step = (element(), element()).prop_map(|(mul, add)| MapStep { mul, add });
    let strategy = (values(), prop::collection::vec(step, 1..=6));

    runner(256)
        .run(&strategy, |(input, steps)| {
            let gpu = &mut *gpu.borrow_mut();
            let expected: Vec<u32> = input
                .iter()
                .map(|&v| steps.iter().fold(v, |x, step| step.apply(x)))
                .collect();

            let mut values = input;
            let mut session = gpu.manager.begin_mut(&mut values).unwrap();
            for &step in &steps {
                session.map(step).unwrap();
            }
            session.finish().unwrap();

            prop_assert_eq!(values, expected);
            Ok(())
        })
        .unwrap();
}

/// One step of a random session sequence.
#[derive(Clone, Debug)]
struct Step {
    input: Vec<u32>,
    kernel: Kernel,
    /// Zero means the session never requests write access.
    reps: usize,
    /// Finish the session, or drop it to discard device changes.
    commit: bool,
}

fn step() -> impl Strategy<Value = Step> {
    (values(), kernel(), 0..=4usize, any::<bool>()).prop_map(|(input, kernel, reps, commit)| Step {
        input,
        kernel,
        reps,
        commit,
    })
}

/// Long-lived manager, mixed lengths, writes or not, finish or drop: the
/// slice must match a CPU model after every step, and transfer counts must
/// show one upload per nonempty session and a download only when needed.
#[test]
fn random_session_sequences_match_a_cpu_model() {
    let gpu = RefCell::new(Gpu::open());
    let strategy = prop::collection::vec(step(), 1..12);

    runner(64)
        .run(&strategy, |steps| {
            let gpu = &mut *gpu.borrow_mut();
            for step in steps {
                let function = gpu.function(step.kernel);
                let mut expected = step.input.clone();
                if step.commit {
                    step.kernel.apply_cpu(&mut expected, step.reps);
                }
                let before = gpu.manager.stats();

                let mut values = step.input;
                let mut session = gpu.manager.begin_mut(&mut values).unwrap();
                for _ in 0..step.reps {
                    launch_in_session(&mut session, &function);
                }
                if step.commit {
                    session.finish().unwrap();
                } else {
                    drop(session);
                }

                prop_assert_eq!(&values, &expected);
                let after = gpu.manager.stats();
                let nonempty = !values.is_empty();
                prop_assert_eq!(after.uploads - before.uploads, u64::from(nonempty));
                let downloaded = nonempty && step.reps > 0 && step.commit;
                prop_assert_eq!(after.downloads - before.downloads, u64::from(downloaded));
                prop_assert!(!gpu.manager.poisoned);
            }
            Ok(())
        })
        .unwrap();
}
