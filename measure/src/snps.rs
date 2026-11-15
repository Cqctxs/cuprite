//! Pairwise SNP distances with `psdm`: every pair of aligned genomes, the
//! number of positions where they differ, ignoring `N` and `-`.

use clap::Parser;
use cuprite_memory::{Arg, Grid};
use psdm::{hamming_distance, Transformer};
use rayon::prelude::*;

use crate::{Case, Gpu, Output};

/// `n` aligned genomes of `len` bases: a random reference, each sample
/// with SNPs at about 1 in 500 positions, and runs of `N` and gaps.
#[derive(Clone, Debug)]
pub struct SnpCase {
    pub n: usize,
    pub len: usize,
}

impl SnpCase {
    /// The alignment as psdm loads it: FASTA, through its `Transformer`
    /// with the command-line defaults, which marks `N` and `-` as ignored.
    pub fn sequences(&self) -> Vec<Vec<u8>> {
        let mut state = 0x2545_f491_4f6c_dd1du64 ^ (self.n * 31 + self.len) as u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let bases = *b"ACGT";
        let reference: Vec<u8> = (0..self.len)
            .map(|_| bases[(next() % 4) as usize])
            .collect();
        let mut fasta = String::new();
        for sample in 0..self.n {
            let mut sequence = reference.clone();
            for base in sequence.iter_mut() {
                let roll = next() % 1000;
                if roll < 2 {
                    *base = bases[(next() % 4) as usize];
                }
            }
            // A masked run and a gap in some samples.
            if sample % 3 == 0 && self.len > 100 {
                let start = (next() % (self.len as u64 - 50)) as usize;
                sequence[start..start + 40].fill(b'N');
            }
            if sample % 5 == 1 && self.len > 100 {
                let start = (next() % (self.len as u64 - 20)) as usize;
                sequence[start..start + 12].fill(b'-');
            }
            fasta.push_str(&format!(">sample{sample}\n"));
            fasta.push_str(std::str::from_utf8(&sequence).unwrap());
            fasta.push('\n');
        }
        let transformer = Transformer::parse_from(["psdm"]);
        let mut reader = noodles_fasta::Reader::new(fasta.as_bytes());
        transformer.load_alignment(&mut reader, 0).unwrap().1
    }

    /// psdm's pairs: `combinations_with_replacement(2)`, so `i <= j`.
    fn pairs(&self) -> Vec<(usize, usize)> {
        (0..self.n)
            .flat_map(|i| (i..self.n).map(move |j| (i, j)))
            .collect()
    }
}

/// psdm's per-pair distance, as its binary computes it.
fn distance(sequences: &[Vec<u8>], (i, j): (usize, usize)) -> u64 {
    if i == j {
        0
    } else {
        hamming_distance(&sequences[i], &sequences[j])
    }
}

fn bytes(distances: Vec<u64>) -> Output {
    Output::Bytes(distances.into_iter().flat_map(u64::to_le_bytes).collect())
}

impl Case for SnpCase {
    fn name(&self) -> String {
        "psdm SNP distance matrix".into()
    }

    fn size(&self) -> (usize, usize) {
        (self.n, self.len)
    }

    fn shape(&self) -> String {
        format!("{} genomes x {} bp", self.n, self.len)
    }

    fn kernel_name(&self) -> &'static str {
        "snp_distances"
    }

    fn cpu(&self) -> Output {
        let sequences = self.sequences();
        bytes(
            self.pairs()
                .into_iter()
                .map(|pair| distance(&sequences, pair))
                .collect(),
        )
    }

    /// psdm's binary parallelizes over the pairs with rayon.
    fn cpu_parallel(&self) -> Output {
        let sequences = self.sequences();
        bytes(
            self.pairs()
                .into_par_iter()
                .map(|pair| distance(&sequences, pair))
                .collect(),
        )
    }

    fn gpu(&self, gpu: &mut Gpu) -> Output {
        let sequences = self.sequences();
        let flat: Vec<u8> = sequences.concat();
        let kernel = gpu.kernel("snp_distances");
        let mut out = vec![0u64; self.n * (self.n + 1) / 2];
        let mut job = gpu.memory.job().unwrap();
        let input = job.input(&flat).unwrap();
        let output = job.output(&mut out).unwrap();
        let n = self.n as u32;
        let args = [
            Arg::Buffer(output),
            Arg::Buffer(input),
            Arg::U32(n),
            Arg::U32(self.len as u32),
        ];
        job.launch(
            kernel,
            Grid::Plane {
                width: n,
                height: n,
            },
            &args,
        )
        .unwrap();
        job.finish().unwrap();
        bytes(out)
    }
}
