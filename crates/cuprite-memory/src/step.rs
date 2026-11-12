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

/// The name of the entry in [`MAP_PTX`].
pub(crate) const MAP_NAME: &str = "map_u32";

/// Handwritten PTX for [`MapStep`], compiled by the driver for the installed
/// GPU when first used, so no CUDA Toolkit is needed. Parameters: `values`,
/// `len` (elements), `mul`, `add`. Each thread handles one element:
/// `values[i] = values[i] * mul + add` (`mad.lo` keeps the low 32 bits,
/// which is wrapping arithmetic).
pub(crate) const MAP_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry map_u32(
    .param .u64 values,
    .param .u64 len,
    .param .u32 mul,
    .param .u32 add
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [values];
    ld.param.u64 %rd2, [len];
    ld.param.u32 %r6, [mul];
    ld.param.u32 %r7, [add];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    cvt.u64.u32 %rd3, %r3;
    mad.wide.u32 %rd4, %r1, %r2, %rd3;
    setp.ge.u64 %p1, %rd4, %rd2;
    @%p1 bra DONE;
    cvta.to.global.u64 %rd5, %rd1;
    shl.b64 %rd6, %rd4, 2;
    add.s64 %rd7, %rd5, %rd6;
    ld.global.u32 %r4, [%rd7];
    mad.lo.u32 %r5, %r4, %r6, %r7;
    st.global.u32 [%rd7], %r5;
DONE:
    ret;
}
"#;
