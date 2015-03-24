// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

use std::mem;

struct Cat {
    x: int
}

struct Kitty {
    x: int,
}

impl Drop for Kitty {
    fn drop(&mut self) {}
}

#[cfg(any(target_arch = "x86_64", target_arch="aarch64"))]
pub fn main() {
    assert_eq!(mem::size_of::<Cat>(), 8 as uint);
    assert_eq!(mem::size_of::<Kitty>(), 16 as uint);
}

#[cfg(any(target_arch = "x86", target_arch = "arm"))]
pub fn main() {
    assert_eq!(mem::size_of::<Cat>(), 4 as uint);
    assert_eq!(mem::size_of::<Kitty>(), 8 as uint);
}
