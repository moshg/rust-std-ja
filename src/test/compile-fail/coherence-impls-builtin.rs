// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(optin_builtin_traits)]

use std::marker::Send;

enum TestE {
  A
}

struct MyType;

struct NotSync;
impl !Sync for NotSync {}

unsafe impl Send for TestE {}
unsafe impl Send for MyType {}
unsafe impl Send for (MyType, MyType) {}
//~^ ERROR builtin traits can only be implemented on structs or enums

unsafe impl Send for &'static NotSync {}
//~^ ERROR builtin traits can only be implemented on structs or enums

unsafe impl Send for [MyType] {}
//~^ ERROR builtin traits can only be implemented on structs or enums

unsafe impl Send for &'static [NotSync] {}
//~^ ERROR builtin traits can only be implemented on structs or enums
//~^^ ERROR conflicting implementations for trait `core::marker::Send`

fn is_send<T: Send>() {}

fn main() {
    is_send::<(MyType, TestE)>();
}
