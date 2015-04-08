// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(core)]

use std::intrinsics::assume;

unsafe fn f(x: i32) -> i32 {
    assume(x == 34);
    match x {
        34 => 42,
        _  => 30
    }
}

fn main() {
    let x = unsafe { f(34) };
    assert_eq!(x, 42);
}
