// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we enforce WF conditions also for where clauses in fn items.

#![feature(rustc_attrs)]
#![allow(dead_code)]

trait MustBeCopy<T:Copy> {
}

fn bar<T,U>() //~ WARN E0277
    where T: MustBeCopy<U>
{
}

#[rustc_error]
fn main() { } //~ ERROR compilation successful
