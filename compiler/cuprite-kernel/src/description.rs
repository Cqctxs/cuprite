//! Kernel descriptions: what the compiler embeds in a rewritten function
//! for the runtime, as text. A description names the function, the kernel's
//! entry point and signature, and holds its PTX.
//!
//! ```text
//! cuprite kernel 3
//! function cuprite_probes::scale
//! entry cuprite_probes__scale
//! element f32
//! inputs u8
//! tables u32
//! repeated no
//! scalars f32
//! ptx
//! <PTX until the end>
//! ```
//!
//! The kernel is [`crate::shapes::in_place_map`]. The call passes the
//! bits of each scalar in `scalars`, then the address and length of each
//! input slice, whose element types `inputs` lists. The runtime passes the
//! data, the shortest length, a zeroed `u32` failure flag, the input
//! pointers, then the stages' scalars: a pointer and length for each
//! table (a read-only slice the code indexes, whose element types `tables`
//! lists, passed like an input after them), then `scalars`. It discards the
//! results if the flag is set. With
//! `repeated yes`, the first scalar is the pass count, a `u32`.

use crate::Ty;

const HEADER: &str = "cuprite kernel 3";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Description {
    /// The Rust function, for traces and reports.
    pub function: String,
    pub entry: String,
    pub element: Ty,
    /// The element types of the read-only slices zipped with the data.
    pub inputs: Vec<Ty>,
    /// The element types of the read-only slices the code indexes.
    pub tables: Vec<Ty>,
    pub repeated: bool,
    /// The kernel's parameters after the data and its length.
    pub scalars: Vec<Ty>,
    pub ptx: String,
}

impl Description {
    pub fn encode(&self) -> String {
        format!(
            "{HEADER}\nfunction {}\nentry {}\nelement {}\ninputs {}\ntables {}\nrepeated {}\nscalars {}\nptx\n{}",
            self.function,
            self.entry,
            self.element.name(),
            names(&self.inputs),
            names(&self.tables),
            if self.repeated { "yes" } else { "no" },
            names(&self.scalars),
            self.ptx
        )
    }

    pub fn decode(text: &str) -> Result<Description, String> {
        let mut lines = text.splitn(10, '\n');
        let mut field = |name: &str| -> Result<&str, String> {
            let line = lines.next().ok_or_else(|| format!("no `{name}` line"))?;
            if name == HEADER || name == "ptx" {
                return if line == name { Ok("") } else { Err(format!("expected `{name}`, found `{line}`")) };
            }
            line.strip_prefix(name)
                .and_then(|rest| rest.strip_prefix(' ').or((rest.is_empty()).then_some("")))
                .ok_or_else(|| format!("expected `{name}`, found `{line}`"))
        };
        field(HEADER)?;
        let function = field("function")?.to_owned();
        let entry = field("entry")?.to_owned();
        let element = field("element")?;
        let element = Ty::from_name(element).ok_or_else(|| format!("unknown element type `{element}`"))?;
        let inputs = field("inputs")?
            .split_whitespace()
            .map(|name| match Ty::from_name(name) {
                Some(ty) if ty.is_int() || ty.is_float() => Ok(ty),
                _ => Err(format!("`{name}` is not an input element type")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tables = field("tables")?
            .split_whitespace()
            .map(|name| match Ty::from_name(name) {
                Some(ty) if ty.is_int() || ty.is_float() => Ok(ty),
                _ => Err(format!("`{name}` is not a table element type")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let repeated = match field("repeated")? {
            "yes" => true,
            "no" => false,
            other => return Err(format!("`repeated {other}` is not yes or no")),
        };
        let scalars = field("scalars")?
            .split_whitespace()
            .map(|name| match Ty::from_name(name) {
                Some(ty) if ty.is_passable() && ty != Ty::Ptr => Ok(ty),
                _ => Err(format!("`{name}` is not a scalar parameter type")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if repeated && scalars.first() != Some(&Ty::U32) {
            return Err("a repeated kernel's first scalar is its `u32` pass count".into());
        }
        field("ptx")?;
        let ptx = lines.next().unwrap_or_default().to_owned();
        Ok(Description { function, entry, element, inputs, tables, repeated, scalars, ptx })
    }
}

fn names(types: &[Ty]) -> String {
    types.iter().map(|ty| ty.name()).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Description {
        Description {
            function: "a::scale".into(),
            entry: "a__scale".into(),
            element: Ty::F32,
            inputs: vec![Ty::U8, Ty::F64],
            tables: vec![Ty::U32],
            repeated: true,
            scalars: vec![Ty::U32, Ty::F32, Ty::I64],
            ptx: ".version 7.0\n.target sm_70\n".into(),
        }
    }

    #[test]
    fn descriptions_survive_encoding() {
        let description = sample();
        assert_eq!(Description::decode(&description.encode()).unwrap(), description);
        let plain = Description { repeated: false, inputs: Vec::new(), tables: Vec::new(), scalars: Vec::new(), ..sample() };
        assert_eq!(Description::decode(&plain.encode()).unwrap(), plain);
    }

    #[test]
    fn malformed_descriptions_are_refused() {
        let text = sample().encode();
        for (from, to) in [
            ("cuprite kernel 3", "cuprite kernel 2"),
            ("element f32", "element f16"),
            ("inputs u8", "inputs bool"),
            ("tables u32", "tables ptr"),
            ("inputs u8 f64\n", "\n"),
            ("repeated yes", "repeated maybe"),
            ("scalars u32", "scalars ptr"),
            ("scalars u32 f32", "scalars f32 f32"),
            ("\nptx\n", "\nptc\n"),
        ] {
            assert!(Description::decode(&text.replacen(from, to, 1)).is_err(), "{to}");
        }
    }
}
