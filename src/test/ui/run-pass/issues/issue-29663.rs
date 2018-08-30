// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
// write_volatile causes an LLVM assert with composite types

#![feature(volatile)]
use std::ptr::{read_volatile, write_volatile};

#[derive(Debug, Eq, PartialEq)]
struct A(u32);
#[derive(Debug, Eq, PartialEq)]
struct B(u64);
#[derive(Debug, Eq, PartialEq)]
struct C(u32, u32);
#[derive(Debug, Eq, PartialEq)]
struct D(u64, u64);
#[derive(Debug, Eq, PartialEq)]
struct E([u64; 32]);

fn main() {
    unsafe {
        let mut x: u32 = 0;
        write_volatile(&mut x, 1);
        assert_eq!(read_volatile(&x), 1);
        assert_eq!(x, 1);

        let mut x: u64 = 0;
        write_volatile(&mut x, 1);
        assert_eq!(read_volatile(&x), 1);
        assert_eq!(x, 1);

        let mut x = A(0);
        write_volatile(&mut x, A(1));
        assert_eq!(read_volatile(&x), A(1));
        assert_eq!(x, A(1));

        let mut x = B(0);
        write_volatile(&mut x, B(1));
        assert_eq!(read_volatile(&x), B(1));
        assert_eq!(x, B(1));

        let mut x = C(0, 0);
        write_volatile(&mut x, C(1, 1));
        assert_eq!(read_volatile(&x), C(1, 1));
        assert_eq!(x, C(1, 1));

        let mut x = D(0, 0);
        write_volatile(&mut x, D(1, 1));
        assert_eq!(read_volatile(&x), D(1, 1));
        assert_eq!(x, D(1, 1));

        let mut x = E([0; 32]);
        write_volatile(&mut x, E([1; 32]));
        assert_eq!(read_volatile(&x), E([1; 32]));
        assert_eq!(x, E([1; 32]));
    }
}
