// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
// aux-build:packed.rs


extern crate packed;

use std::mem;

macro_rules! check {
    ($t:ty, $align:expr, $size:expr) => ({
        assert_eq!(mem::align_of::<$t>(), $align);
        assert_eq!(mem::size_of::<$t>(), $size);
    });
}

pub fn main() {
    check!(packed::P1S5, 1, 5);
    check!(packed::P2S6, 2, 6);
    check!(packed::P2CS8, 2, 8);
}
