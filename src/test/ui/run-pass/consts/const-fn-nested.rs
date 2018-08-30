// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
// Test a call whose argument is the result of another call.

#![feature(min_const_fn)]

const fn sub(x: u32, y: u32) -> u32 {
    x - y
}

const X: u32 = sub(sub(88, 44), 22);

fn main() {
    assert_eq!(X, 22);
}
