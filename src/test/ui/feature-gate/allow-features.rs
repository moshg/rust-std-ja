// compile-flags: -Z allow_features=lang_items
// Note: This test uses rustc internal flags because they will never stabilize.

#![feature(rustc_const_unstable)] //~ ERROR

#![feature(lang_items)]

#![feature(unknown_stdlib_feature)] //~ ERROR

fn main() {}
