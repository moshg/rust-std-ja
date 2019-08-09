// Test that `try!` macros are rewritten.

// run-rustfix
// build-pass (FIXME(62277): could be check-pass?)

#![warn(rust_2018_compatibility)]
#![allow(unused_variables)]
#![allow(dead_code)]

fn foo() -> Result<usize, ()> {
    let x: Result<usize, ()> = Ok(22);
    try!(x);
    Ok(44)
}

fn main() { }
