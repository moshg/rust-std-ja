// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::*;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct AllTheRanges {
    a: Range<usize>,
    //~^ ERROR PartialOrd
    //~^^ ERROR Ord
    //~^^^ the trait bound
    b: RangeTo<usize>,
    //~^ ERROR PartialOrd
    //~^^ ERROR Ord
    //~^^^ no method named `partial_cmp`
    c: RangeFrom<usize>,
    //~^ ERROR PartialOrd
    //~^^ ERROR Ord
    //~^^^ the trait bound
    d: RangeFull,
    //~^ ERROR PartialOrd
    //~^^ ERROR Ord
    //~^^^ no method named `partial_cmp`
    e: RangeInclusive<usize>,
    //~^ ERROR PartialOrd
    //~^^ ERROR Ord
    //~^^^ the trait bound
    f: RangeToInclusive<usize>,
    //~^ ERROR PartialOrd
    //~^^ ERROR Ord
    //~^^^ no method named `partial_cmp`
}

fn main() {}
