// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Deref;

pub trait Foo {
    fn baz(_: Self::Target) where Self: Deref {}
    //~^ ERROR `<Self as std::ops::Deref>::Target` does not have a constant size known at
}

pub fn f(_: ToString) {}
//~^ ERROR `std::string::ToString + 'static` does not have a constant size known at compile-time

fn main() { }
