// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn test_int() {
    fn f() -> int { 10 }
    fail_unless!((f() == 10));
}

fn test_vec() {
    fn f() -> ~[int] { ~[10, 11] }
    fail_unless!((f()[1] == 11));
}

fn test_generic() {
    fn f<T:Copy>(t: T) -> T { t }
    fail_unless!((f(10) == 10));
}

fn test_alt() {
    fn f() -> int { match true { false => { 10 } true => { 20 } } }
    fail_unless!((f() == 20));
}

fn test_if() {
    fn f() -> int { if true { 10 } else { 20 } }
    fail_unless!((f() == 10));
}

fn test_block() {
    fn f() -> int { { 10 } }
    fail_unless!((f() == 10));
}

fn test_ret() {
    fn f() -> int {
        return 10 // no semi

    }
    fail_unless!((f() == 10));
}


// From issue #372
fn test_372() {
    fn f() -> int { let x = { 3 }; x }
    fail_unless!((f() == 3));
}

fn test_nil() { () }

pub fn main() {
    test_int();
    test_vec();
    test_generic();
    test_alt();
    test_if();
    test_block();
    test_ret();
    test_372();
    test_nil();
}
