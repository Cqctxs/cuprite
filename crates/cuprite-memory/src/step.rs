/// One element-wise map over a resident slice: `x = x * mul + add`, with
/// wrapping `u32` arithmetic, so it never panics and matches ordinary Rust
/// `wrapping_mul` and `wrapping_add` bit for bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapStep {
    pub mul: u32,
    pub add: u32,
}

impl MapStep {
    /// CPU reference for one element.
    pub fn apply(self, value: u32) -> u32 {
        value.wrapping_mul(self.mul).wrapping_add(self.add)
    }
}
