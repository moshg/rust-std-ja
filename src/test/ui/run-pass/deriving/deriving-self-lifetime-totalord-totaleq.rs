// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
use std::cmp::Ordering::{Less,Equal,Greater};

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct A<'a> {
    x: &'a isize
}
pub fn main() {
    let (a, b) = (A { x: &1 }, A { x: &2 });

    assert_eq!(a.cmp(&a), Equal);
    assert_eq!(b.cmp(&b), Equal);

    assert_eq!(a.cmp(&b), Less);
    assert_eq!(b.cmp(&a), Greater);
}
