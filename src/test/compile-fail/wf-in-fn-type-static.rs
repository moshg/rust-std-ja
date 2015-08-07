// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we enforce WF conditions related to regions also for
// types in fns.

#![allow(dead_code)]
#![feature(rustc_attrs)]

struct MustBeCopy<T:Copy> {
    t: T
}

struct Foo<T> { //~ WARN E0310
    // needs T: 'static
    x: fn() -> &'static T //~ WARN E0310
}

struct Bar<T> { //~ WARN E0310
    // needs T: Copy
    x: fn(&'static T) //~ WARN E0310
}

#[rustc_error]
fn main() { } //~ ERROR compilation successful
