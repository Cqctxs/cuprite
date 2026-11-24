//! CPU and GPU timing curves fitted from calibration samples.

/// The measured cost at one length: `fixed + per_pass·p` microseconds for
/// `p` passes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    pub len: usize,
    pub fixed: f64,
    pub per_pass: f64,
}

impl Row {
    fn at(&self, passes: f64) -> f64 {
        self.fixed + self.per_pass * passes
    }
}

/// Predicted time of one complete call, from measurements at a few lengths.
///
/// At each measured length the time is linear in the passes: on the GPU,
/// `fixed` is the session and its transfers and `per_pass` a launch and its
/// work; on the CPU nearly all of it is `per_pass`. Between lengths the time
/// is interpolated linearly in the length, and beyond them extrapolated from
/// the nearest two. A single global formula cannot follow the CPU from cache
/// into main memory, where each element costs 2.5 times as much.
#[derive(Clone, Debug, PartialEq)]
pub struct Cost {
    /// At least two, by strictly increasing length.
    pub rows: Vec<Row>,
}

impl Cost {
    pub fn predict(&self, len: usize, passes: u32) -> f64 {
        let length = len as f64;
        let passes = f64::from(passes);
        // Use the nearest segment when extrapolating beyond the measurements.
        let upper_index = self
            .rows
            .partition_point(|row| row.len as f64 <= length)
            .clamp(1, self.rows.len() - 1);
        let lower = &self.rows[upper_index - 1];
        let upper = &self.rows[upper_index];
        let fraction = (length - lower.len as f64) / (upper.len - lower.len) as f64;
        let lower_time = lower.at(passes);
        (lower_time + fraction * (upper.at(passes) - lower_time)).max(0.0)
    }
}

/// One complete call: `len` elements and `passes` passes took `micros`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub len: usize,
    pub passes: u32,
    pub micros: f64,
}

/// Fits a [`Cost`] with one row per measured length. Returns `None` unless
/// there are at least two lengths, each with two pass counts, and every
/// time is positive.
pub fn fit(samples: &[Sample]) -> Option<Cost> {
    if samples
        .iter()
        .any(|s| !(s.micros > 0.0 && s.micros.is_finite()))
    {
        return None;
    }
    let mut lengths: Vec<usize> = samples.iter().map(|s| s.len).collect();
    lengths.sort_unstable();
    lengths.dedup();
    let rows: Vec<Row> = lengths
        .into_iter()
        .map(|len| {
            let points: Vec<(f64, f64)> = samples
                .iter()
                .filter(|s| s.len == len)
                .map(|s| (f64::from(s.passes), s.micros))
                .collect();
            fit_line(&points).map(|(fixed, per_pass)| Row {
                len,
                fixed,
                per_pass,
            })
        })
        .collect::<Option<_>>()?;
    (rows.len() >= 2).then_some(Cost { rows })
}

/// `y = a + b·x` with `a, b ≥ 0`, by least squares on relative error, so
/// that one pass, where the choice is closest, counts as much as many.
fn fit_line(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    // Weighted sums with weights 1/y².
    let sum = |f: &dyn Fn(f64, f64) -> f64| points.iter().map(|&(x, y)| f(x, y) / (y * y)).sum();
    let (s1, sx, sxx): (f64, f64, f64) = (sum(&|_, _| 1.0), sum(&|x, _| x), sum(&|x, _| x * x));
    let (sy, sxy): (f64, f64) = (sum(&|_, y| y), sum(&|x, y| x * y));
    let det = s1 * sxx - sx * sx;
    if det <= 1e-12 * s1 * sxx {
        return None; // Fewer than two distinct pass counts.
    }
    let a = (sxx * sy - sx * sxy) / det;
    let b = (s1 * sxy - sx * sy) / det;
    Some(if a < 0.0 {
        (0.0, sxy / sxx)
    } else if b < 0.0 {
        (sy / s1, 0.0)
    } else {
        (a, b)
    })
}

/// The largest relative difference between a model and its samples.
pub fn worst_error(cost: &Cost, samples: &[Sample]) -> f64 {
    samples
        .iter()
        .map(|s| (cost.predict(s.len, s.passes) - s.micros).abs() / s.micros)
        .fold(0.0, f64::max)
}
