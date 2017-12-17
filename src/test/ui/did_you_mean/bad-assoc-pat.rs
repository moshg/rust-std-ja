// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    match 0u8 {
        [u8]::AssocItem => {}
        //~^ ERROR missing angle brackets in associated item path
        //~| ERROR no associated item named `AssocItem` found for type `[u8]` in the current scope
        (u8, u8)::AssocItem => {}
        //~^ ERROR missing angle brackets in associated item path
        //~| ERROR no associated item named `AssocItem` found for type `(u8, u8)` in the current sco
        _::AssocItem => {}
        //~^ ERROR missing angle brackets in associated item path
        //~| ERROR no associated item named `AssocItem` found for type `_` in the current scope
    }
}
