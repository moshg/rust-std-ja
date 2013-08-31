// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern fn foopy() {}

static f: extern "C" fn() = foopy;
static s: S = S { f: foopy };

struct S {
    f: extern "C" fn()
}

pub fn main() {
    assert_eq!(foopy, f);
    assert_eq!(f, s.f);
}
