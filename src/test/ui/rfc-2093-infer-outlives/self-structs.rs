// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

#[rustc_outlives]
struct Foo<'a, 'b, T> { //~ ERROR 14:1: 16:2: rustc_outlives
    field1: Bar<'a, 'b, T>
}

trait Bar<'x, 's, U>
    where U: 'x,
    Self:'s
{}

fn main() {}
