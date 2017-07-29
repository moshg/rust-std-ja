// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Unnecessary path disambiguator is ok

#![feature(rustc_attrs)]
#![allow(unused)]

struct Foo<T> {
    _a: T,
}

fn f() {
    let f = Some(Foo { _a: 42 }).map(|a| a as Foo::<i32>);
    let g: Foo::<i32> = Foo { _a: 42 };
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
