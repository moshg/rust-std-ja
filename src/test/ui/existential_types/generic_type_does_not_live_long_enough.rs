// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(existential_type)]

fn main() {
    let y = 42;
    let x = wrong_generic(&y);
    let z: i32 = x; //~ ERROR mismatched types
}

existential type WrongGeneric<T>: 'static;
//~^ ERROR the parameter type `T` may not live long enough

fn wrong_generic<T>(t: T) -> WrongGeneric<T> {
    t
}
