// Test that we are able to introduce a negative constraint that
// `MyType: !MyTrait` along with other "fundamental" wrappers.

// check-pass
// aux-build:coherence_copy_like_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

extern crate coherence_copy_like_lib as lib;

struct MyType { x: i32 }

// naturally, legal
impl lib::MyCopy for MyType { }

fn main() { }
