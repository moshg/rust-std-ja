// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(exhaustive_integer_patterns)]
#![feature(exclusive_range_pattern)]
#![deny(unreachable_patterns)]

use std::{char, usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128};

fn main() {
    let x: u8 = 0;

    // A single range covering the entire domain.
    match x {
        0 ..= 255 => {} // ok
    }

    // A combination of ranges and values.
    // These are currently allowed to be overlapping.
    match x {
        0 ..= 32 => {}
        33 => {}
        34 .. 128 => {}
        100 ..= 200 => {}
        200 => {} //~ ERROR unreachable pattern
        201 ..= 255 => {}
    }

    // An incomplete set of values.
    match x { //~ ERROR non-exhaustive patterns
        0 .. 128 => {}
    }

    // A more incomplete set of values.
    match x { //~ ERROR non-exhaustive patterns
        0 ..= 10 => {}
        20 ..= 30 => {}
        35 => {}
        70 .. 255 => {}
    }

    let x: i8 = 0;
    match x { //~ ERROR non-exhaustive patterns
        -7 => {}
        -5..=120 => {}
        -2..=20 => {} //~ ERROR unreachable pattern
        125 => {}
    }

    // Let's test other types too!
    let c: char = '\u{0}';
    match c {
        '\u{0}' ..= char::MAX => {} // ok
    }

    // We can actually get away with just covering the
    // following two ranges, which correspond to all
    // valid Unicode Scalar Values.
    match c {
        '\u{0000}' ..= '\u{D7FF}' => {}
        '\u{E000}' ..= '\u{10_FFFF}' => {}
    }

    match 0usize {
        0 ..= usize::MAX => {} // ok
    }

    match 0u16 {
        0 ..= u16::MAX => {} // ok
    }

    match 0u32 {
        0 ..= u32::MAX => {} // ok
    }

    match 0u64 {
        0 ..= u64::MAX => {} // ok
    }

    match 0u128 {
        0 ..= u128::MAX => {} // ok
    }

    match 0isize {
        isize::MIN ..= isize::MAX => {} // ok
    }

    match 0i8 {
        -128 ..= 127 => {} // ok
    }

    match 0i8 { //~ ERROR non-exhaustive patterns
        -127 ..= 127 => {}
    }

    match 0i16 {
        i16::MIN ..= i16::MAX => {} // ok
    }

    match 0i16 { //~ ERROR non-exhaustive patterns
        i16::MIN ..= -1 => {}
        1 ..= i16::MAX => {}
    }

    match 0i32 {
        i32::MIN ..= i32::MAX => {} // ok
    }

    match 0i64 {
        i64::MIN ..= i64::MAX => {} // ok
    }

    match 0i128 {
        i128::MIN ..= i128::MAX => {} // ok
    }

    // Make sure that guards don't factor into the exhaustiveness checks.
    match 0u8 { //~ ERROR non-exhaustive patterns
        0 .. 128 => {}
        128 ..= 255 if true => {}
    }

    match 0u8 {
        0 .. 128 => {}
        128 ..= 255 if false => {}
        128 ..= 255 => {} // ok, because previous arm was guarded
    }

    // Now things start getting a bit more interesting. Testing products!
    match (0u8, Some(())) { //~ ERROR non-exhaustive patterns
        (1, _) => {}
        (_, None) => {}
    }
}
