//! Has a map the driver would rewrite, but procedural macros are compiled
//! as plain Rust: they run while building, not in the program.

use proc_macro::TokenStream;

fn bump(values: &mut [u32]) {
    for value in values.iter_mut() {
        *value = value.wrapping_mul(3).wrapping_add(1);
    }
}

/// Expands to `42u32`.
#[proc_macro]
pub fn answer(_input: TokenStream) -> TokenStream {
    let mut values = [0, 13];
    bump(&mut values);
    format!("{}u32", values[1] + 2).parse().unwrap()
}
