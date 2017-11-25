// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only

#![feature(generic_associated_types)]

impl<T> Baz for T where T: Foo {
    //FIXME(sunjay): This should parse successfully
    type Quux<'a> = <T as Foo>::Bar<'a, 'static>;
}

fn main() {}
