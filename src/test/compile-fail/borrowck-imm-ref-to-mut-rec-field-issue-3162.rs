// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn each<T>(x: &[T], op: fn(elem: &T) -> bool) {
    uint::range(0, x.len(), |i| op(&x[i]));
}

fn main() {
    struct A {
        mut a: int
    }
    let x = ~[A {mut a: 0}];
    for each(x) |y| {
        let z = &y.a; //~ ERROR illegal borrow unless pure
        x[0].a = 10; //~ NOTE impure due to assigning to mutable field
        log(error, z);
    }
}
