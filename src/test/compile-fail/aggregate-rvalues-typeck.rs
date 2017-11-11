// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//revisions: ast mir
//[mir] compile-flags: -Z emit-end-regions -Z borrowck-mir -Z nll

#![allow(unused_assignments)]

struct Wrap<'a> { w: &'a mut u32 }

fn foo() {
    let mut x = 22;
    let wrapper = Wrap { w: &mut x };
    //~^ ERROR cannot assign to `x` because it is borrowed (Mir) [E0506]
    //~^^ ERROR cannot use `x` because it was mutably borrowed (Mir) [E0503]
    x += 1; //[ast]~ ERROR cannot assign to `x` because it is borrowed [E0506]
    //[mir]~^ ERROR cannot assign to `x` because it is borrowed (Ast) [E0506]
    //[mir]~^^ ERROR cannot assign to `x` because it is borrowed (Mir) [E0506]
    //[mir]~^^^ ERROR cannot use `x` because it was mutably borrowed (Mir) [E0503]
    *wrapper.w += 1;
}

fn main() { }