// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![forbid(warnings)]
#![feature(std_misc)]

// Pretty printing tests complain about `use std::predule::*`
#![allow(unused_imports)]

// We shouldn't need to rebind a moved upvar as mut if it's already
// marked as mut

use std::thunk::Thunk;

pub fn main() {
    let mut x = 1;
    let _thunk = Thunk::new(move|| { x = 2; });
}
