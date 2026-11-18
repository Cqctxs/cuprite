//! Kernel descriptions: what the compiler embeds in a rewritten function
//! for the runtime, as text. A description names the function, the kernel's
//! entry point and signature, and holds its PTX.
//!
//! ```text
//! cuprite kernel 4
//! function cuprite_probes::scale
//! entry cuprite_probes__scale
//! outputs f32 16=0:f64,8:f64
//! inputs u8
//! tables u32
//! repeated no
//! scalars f32
//! ptx
//! <PTX until the end>
//! ```
//!
//! The kernel is [`crate::shapes::in_place_map`]. `outputs` and `inputs`
//! are the [`Layout`]s of the slices it writes and reads, element by
//! element. The call passes the first output slice itself, then the bits
//! of each scalar in `scalars`, then the address and length of each other
//! output slice and each input slice. The runtime passes the first output,
//! the shortest length, a zeroed `u32` failure flag, the other outputs'
//! and the inputs' pointers, then the stages' scalars: a pointer and
//! length for each table (a read-only slice the code indexes, whose
//! element types `tables` lists, passed like an input after them), then
//! `scalars`. It discards the results if the flag is set. With
//! `repeated yes`, the first scalar is the pass count, a `u32`.

use crate::{layout::Layout, Ty};

const HEADER: &str = "cuprite kernel 4";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Description {
    /// The Rust function, for traces and reports.
    pub function: String,
    pub entry: String,
    /// The slices the kernel writes, element by element, the first being
    /// the one the call passes.
    pub outputs: Vec<Layout>,
    /// The read-only slices zipped with the outputs.
    pub inputs: Vec<Layout>,
    /// The element types of the read-only slices the code indexes.
    pub tables: Vec<Ty>,
    pub repeated: bool,
    /// The kernel's parameters after the data and its length.
    pub scalars: Vec<Ty>,
    pub ptx: String,
}

impl Description {
    pub fn encode(&self) -> String {
        let layouts = |layouts: &[Layout]| layouts.iter().map(Layout::encode).collect::<Vec<_>>().join(" ");
        format!(
            "{HEADER}\nfunction {}\nentry {}\noutputs {}\ninputs {}\ntables {}\nrepeated {}\nscalars {}\nptx\n{}",
            self.function,
            self.entry,
            layouts(&self.outputs),
            layouts(&self.inputs),
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
        let layouts = |line: &str| line.split_whitespace().map(Layout::decode).collect::<Result<Vec<_>, _>>();
        let outputs = layouts(field("outputs")?)?;
        if outputs.is_empty() {
            return Err("a kernel writes at least one slice".into());
        }
        let inputs = layouts(field("inputs")?)?;
        let tables = field("tables")?
            .split_whitespace()
            .map(|name| match Ty::from_name(name) {
                Some(ty) if (ty.is_int() || ty.is_float()) && ty.bits() <= 64 => Ok(ty),
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
        Ok(Description { function, entry, outputs, inputs, tables, repeated, scalars, ptx })
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
            outputs: vec![Layout::number(Ty::F32), Layout { size: 16, leaves: vec![(0, Ty::F64), (8, Ty::F64)] }],
            inputs: vec![Layout::number(Ty::U8), Layout::number(Ty::F64)],
            tables: vec![Ty::U32],
            repeated: true,
            scalars: vec![Ty::U32, Ty::F32, Ty::I64],
            ptx: ".version 7.0\n.target sm_70\n".into(),
        }
    }

    #[test]
    fn descriptions_survive_encoding() {
        let description = sample();
        assert!(description.encode().contains("\noutputs f32 16=0:f64,8:f64\n"));
        assert_eq!(Description::decode(&description.encode()).unwrap(), description);
        let plain = Description {
            outputs: vec![Layout::number(Ty::I16)],
            repeated: false,
            inputs: Vec::new(),
            tables: Vec::new(),
            scalars: Vec::new(),
            ..sample()
        };
        assert_eq!(Description::decode(&plain.encode()).unwrap(), plain);
    }

    #[test]
    fn malformed_descriptions_are_refused() {
        let text = sample().encode();
        for (from, to) in [
            ("cuprite kernel 4", "cuprite kernel 3"),
            ("outputs f32", "outputs f16"),
            ("outputs f32 16=0:f64,8:f64", "outputs"),
            ("16=0:f64,8:f64", "16=0:f64"),
            ("inputs u8", "inputs bool"),
            ("outputs f32", "outputs u128"),
            ("inputs u8", "inputs i128"),
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
