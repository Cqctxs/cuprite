//! Slice element layouts: where the numbers of one element of a slice sit.

use crate::{Invalid, Ty};

/// The numbers in one element of a slice, by byte offset, in the order an
/// element function takes them; `size` is the distance between elements.
///
/// A valid layout covers every byte of the element exactly once, with
/// numbers of at most 64 bits aligned to their size. With no padding and
/// only numbers, any bytes are a valid element, so elements can move
/// between the CPU and the GPU as raw bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub size: u32,
    pub leaves: Vec<(u32, Ty)>,
}

impl Layout {
    /// A slice of one number type.
    pub fn number(ty: Ty) -> Self {
        Layout { size: ty.bytes(), leaves: vec![(0, ty)] }
    }

    /// The numbers' types, in order.
    pub fn types(&self) -> Vec<Ty> {
        self.leaves.iter().map(|&(_, ty)| ty).collect()
    }

    /// The type, if the element is a single number.
    pub fn as_number(&self) -> Option<Ty> {
        match self.leaves[..] {
            [(0, ty)] if ty.bytes() == self.size => Some(ty),
            _ => None,
        }
    }

    pub fn validate(&self) -> Result<(), Invalid> {
        let invalid = |reason: String| Err(Invalid(format!("element layout {}: {reason}", self.encode())));
        if self.leaves.is_empty() {
            return invalid("holds no numbers".into());
        }
        let mut covered = vec![false; self.size as usize];
        for &(offset, ty) in &self.leaves {
            if !(ty.is_int() || ty.is_float()) || ty.bits() > 64 {
                return invalid(format!("holds a {ty:?}"));
            }
            if offset % ty.bytes() != 0 {
                return invalid(format!("a {ty:?} at {offset} isn't aligned"));
            }
            let (start, end) = (offset as usize, (offset + ty.bytes()) as usize);
            let Some(bytes) = covered.get_mut(start..end) else {
                return invalid(format!("a {ty:?} at {offset} is past the end"));
            };
            if bytes.iter().any(|&b| b) {
                return invalid(format!("a {ty:?} at {offset} overlaps another number"));
            }
            bytes.fill(true);
        }
        if covered.contains(&false) {
            return invalid("has padding".into());
        }
        Ok(())
    }

    /// A number's name (`f32`), or the size then each number's offset and
    /// type (`16=0:f64,8:f64`).
    pub fn encode(&self) -> String {
        if let Some(ty) = self.as_number() {
            return ty.name().into();
        }
        let leaves: Vec<String> = self.leaves.iter().map(|(offset, ty)| format!("{offset}:{}", ty.name())).collect();
        format!("{}={}", self.size, leaves.join(","))
    }

    /// Reads [`Layout::encode`]'s text, and checks the layout.
    pub fn decode(text: &str) -> Result<Layout, String> {
        let layout = match text.split_once('=') {
            None => Layout::number(Ty::from_name(text).ok_or_else(|| format!("unknown element type `{text}`"))?),
            Some((size, leaves)) => {
                let size = size.parse().map_err(|_| format!("`{text}` has no element size"))?;
                let leaves = leaves
                    .split(',')
                    .map(|leaf| {
                        let parsed = leaf.split_once(':').and_then(|(offset, ty)| Some((offset.parse().ok()?, Ty::from_name(ty)?)));
                        parsed.ok_or_else(|| format!("`{leaf}` in `{text}` isn't an offset and a number type"))
                    })
                    .collect::<Result<_, _>>()?;
                Layout { size, leaves }
            }
        };
        layout.validate().map_err(|invalid| invalid.0)?;
        Ok(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_survive_encoding() {
        for layout in [
            Layout::number(Ty::F32),
            Layout::number(Ty::U8),
            Layout { size: 16, leaves: vec![(0, Ty::F64), (8, Ty::F64)] },
            Layout { size: 8, leaves: vec![(4, Ty::U32), (0, Ty::I16), (2, Ty::U8), (3, Ty::I8)] },
        ] {
            assert_eq!(Layout::decode(&layout.encode()), Ok(layout));
        }
        assert_eq!(Layout::number(Ty::F32).encode(), "f32");
        assert_eq!(Layout { size: 16, leaves: vec![(0, Ty::F64), (8, Ty::F64)] }.encode(), "16=0:f64,8:f64");
    }

    #[test]
    fn layouts_that_leave_bytes_or_misplace_numbers_are_refused() {
        for text in [
            "bool",       // Not a number every byte pattern of which is valid.
            "u128",       // Wider than a kernel loads.
            "8=0:u32",    // Padding after the number.
            "8=0:u32,2:u32", // Misaligned, and overlapping.
            "8=0:u32,4:u64", // Past the end.
            "8=0:u32,0:u32", // The same bytes twice.
            "8=",
            "x=0:u32",
            "4=0:f16",
        ] {
            assert!(Layout::decode(text).is_err(), "{text}");
        }
    }
}
