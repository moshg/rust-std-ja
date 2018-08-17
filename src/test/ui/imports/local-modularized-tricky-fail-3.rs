// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Crate-local macro expanded `macro_export` macros cannot be accessed with module-relative paths.

macro_rules! define_exported { () => {
    #[macro_export]
    macro_rules! exported {
        () => ()
    }
}}

define_exported!();

mod m {
    use exported;
    //~^ ERROR macro-expanded `macro_export` macros from the current crate cannot
}

fn main() {
    ::exported!();
    //~^ ERROR macro-expanded `macro_export` macros from the current crate cannot
}
