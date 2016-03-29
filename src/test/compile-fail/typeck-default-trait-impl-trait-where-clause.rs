// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

// Test that when a `..` impl applies, we also check that any
// supertrait conditions are met.

#![feature(optin_builtin_traits)]

trait NotImplemented { }

trait MyTrait: Sized
    where Option<Self> : NotImplemented
{}

impl NotImplemented for i32 {}

impl MyTrait for .. {}

fn foo<T:MyTrait>() {
    //~^ ERROR `std::option::Option<T> : NotImplemented` is not satisfied
    // This should probably typecheck. This is #20671.
}

fn bar<T:NotImplemented>() { }

fn main() {
}
