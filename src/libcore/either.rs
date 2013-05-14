// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A type that represents one of two alternatives

use container::Container;
use cmp::Eq;
use kinds::Copy;
use old_iter::BaseIter;
use result::Result;
use result;
use vec;
use vec::OwnedVector;

/// The either type
#[deriving(Clone, Eq)]
pub enum Either<T, U> {
    Left(T),
    Right(U)
}

#[inline(always)]
pub fn either<T, U, V>(f_left: &fn(&T) -> V,
                       f_right: &fn(&U) -> V, value: &Either<T, U>) -> V {
    /*!
     * Applies a function based on the given either value
     *
     * If `value` is left(T) then `f_left` is applied to its contents, if
     * `value` is right(U) then `f_right` is applied to its contents, and the
     * result is returned.
     */

    match *value {
      Left(ref l) => f_left(l),
      Right(ref r) => f_right(r)
    }
}

pub fn lefts<T:Copy,U>(eithers: &[Either<T, U>]) -> ~[T] {
    //! Extracts from a vector of either all the left values

    do vec::build_sized(eithers.len()) |push| {
        for eithers.each |elt| {
            match *elt {
                Left(ref l) => { push(*l); }
                _ => { /* fallthrough */ }
            }
        }
    }
}

pub fn rights<T, U: Copy>(eithers: &[Either<T, U>]) -> ~[U] {
    //! Extracts from a vector of either all the right values

    do vec::build_sized(eithers.len()) |push| {
        for eithers.each |elt| {
            match *elt {
                Right(ref r) => { push(*r); }
                _ => { /* fallthrough */ }
            }
        }
    }
}

pub fn partition<T, U>(eithers: ~[Either<T, U>])
    -> (~[T], ~[U]) {
    /*!
     * Extracts from a vector of either all the left values and right values
     *
     * Returns a structure containing a vector of left values and a vector of
     * right values.
     */

    let mut lefts: ~[T] = ~[];
    let mut rights: ~[U] = ~[];
    do vec::consume(eithers) |_i, elt| {
        match elt {
          Left(l) => lefts.push(l),
          Right(r) => rights.push(r)
        }
    }
    return (lefts, rights);
}

#[inline(always)]
pub fn flip<T, U>(eith: Either<T, U>) -> Either<U, T> {
    //! Flips between left and right of a given either

    match eith {
      Right(r) => Left(r),
      Left(l) => Right(l)
    }
}

#[inline(always)]
pub fn to_result<T, U>(eith: Either<T, U>)
    -> Result<U, T> {
    /*!
     * Converts either::t to a result::t
     *
     * Converts an `either` type to a `result` type, making the "right" choice
     * an ok result, and the "left" choice a fail
     */

    match eith {
      Right(r) => result::Ok(r),
      Left(l) => result::Err(l)
    }
}

#[inline(always)]
pub fn is_left<T, U>(eith: &Either<T, U>) -> bool {
    //! Checks whether the given value is a left

    match *eith { Left(_) => true, _ => false }
}

#[inline(always)]
pub fn is_right<T, U>(eith: &Either<T, U>) -> bool {
    //! Checks whether the given value is a right

    match *eith { Right(_) => true, _ => false }
}

#[inline(always)]
pub fn unwrap_left<T,U>(eith: Either<T,U>) -> T {
    //! Retrieves the value in the left branch. Fails if the either is Right.

    match eith {
        Left(x) => x,
        Right(_) => fail!("either::unwrap_left Right")
    }
}

#[inline(always)]
pub fn unwrap_right<T,U>(eith: Either<T,U>) -> U {
    //! Retrieves the value in the right branch. Fails if the either is Left.

    match eith {
        Right(x) => x,
        Left(_) => fail!("either::unwrap_right Left")
    }
}

pub impl<T, U> Either<T, U> {
    #[inline(always)]
    fn either<V>(&self, f_left: &fn(&T) -> V, f_right: &fn(&U) -> V) -> V {
        either(f_left, f_right, self)
    }

    #[inline(always)]
    fn flip(self) -> Either<U, T> { flip(self) }

    #[inline(always)]
    fn to_result(self) -> Result<U, T> { to_result(self) }

    #[inline(always)]
    fn is_left(&self) -> bool { is_left(self) }

    #[inline(always)]
    fn is_right(&self) -> bool { is_right(self) }

    #[inline(always)]
    fn unwrap_left(self) -> T { unwrap_left(self) }

    #[inline(always)]
    fn unwrap_right(self) -> U { unwrap_right(self) }
}

#[test]
fn test_either_left() {
    let val = Left(10);
    fn f_left(x: &int) -> bool { *x == 10 }
    fn f_right(_x: &uint) -> bool { false }
    assert!((either(f_left, f_right, &val)));
}

#[test]
fn test_either_right() {
    let val = Right(10u);
    fn f_left(_x: &int) -> bool { false }
    fn f_right(x: &uint) -> bool { *x == 10u }
    assert!((either(f_left, f_right, &val)));
}

#[test]
fn test_lefts() {
    let input = ~[Left(10), Right(11), Left(12), Right(13), Left(14)];
    let result = lefts(input);
    assert_eq!(result, ~[10, 12, 14]);
}

#[test]
fn test_lefts_none() {
    let input: ~[Either<int, int>] = ~[Right(10), Right(10)];
    let result = lefts(input);
    assert_eq!(result.len(), 0u);
}

#[test]
fn test_lefts_empty() {
    let input: ~[Either<int, int>] = ~[];
    let result = lefts(input);
    assert_eq!(result.len(), 0u);
}

#[test]
fn test_rights() {
    let input = ~[Left(10), Right(11), Left(12), Right(13), Left(14)];
    let result = rights(input);
    assert_eq!(result, ~[11, 13]);
}

#[test]
fn test_rights_none() {
    let input: ~[Either<int, int>] = ~[Left(10), Left(10)];
    let result = rights(input);
    assert_eq!(result.len(), 0u);
}

#[test]
fn test_rights_empty() {
    let input: ~[Either<int, int>] = ~[];
    let result = rights(input);
    assert_eq!(result.len(), 0u);
}

#[test]
fn test_partition() {
    let input = ~[Left(10), Right(11), Left(12), Right(13), Left(14)];
    let (lefts, rights) = partition(input);
    assert_eq!(lefts[0], 10);
    assert_eq!(lefts[1], 12);
    assert_eq!(lefts[2], 14);
    assert_eq!(rights[0], 11);
    assert_eq!(rights[1], 13);
}

#[test]
fn test_partition_no_lefts() {
    let input: ~[Either<int, int>] = ~[Right(10), Right(11)];
    let (lefts, rights) = partition(input);
    assert_eq!(lefts.len(), 0u);
    assert_eq!(rights.len(), 2u);
}

#[test]
fn test_partition_no_rights() {
    let input: ~[Either<int, int>] = ~[Left(10), Left(11)];
    let (lefts, rights) = partition(input);
    assert_eq!(lefts.len(), 2u);
    assert_eq!(rights.len(), 0u);
}

#[test]
fn test_partition_empty() {
    let input: ~[Either<int, int>] = ~[];
    let (lefts, rights) = partition(input);
    assert_eq!(lefts.len(), 0u);
    assert_eq!(rights.len(), 0u);
}
