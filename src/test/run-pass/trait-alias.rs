// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait SimpleAlias = Default;
trait GenericAlias<T> = Iterator<Item=T>;
trait Partial<T> = IntoIterator<Item=T>;

trait Things<T> {}
trait Romeo {}
struct The<T>(T);
struct Fore<T>(T);
impl<T, U> Things<T> for The<U> {}
impl<T> Romeo for Fore<T> {}

trait WithWhere<Art, Thou> = Romeo + Romeo where Fore<(Art, Thou)>: Romeo;
trait BareWhere<Wild, Are> = where The<Wild>: Things<Are>;

trait CD = Clone + Default;

fn foo<T: CD>() -> (T, T) {
    let one = T::default();
    let two = one.clone();
    (one, two)
}

fn main() {
    let both = foo();
    assert_eq!(both.0, 0);
    assert_eq!(both.1, 0);
    let both: (i32, i32) = foo();
    assert_eq!(both.0, 0);
    assert_eq!(both.1, 0);
}

