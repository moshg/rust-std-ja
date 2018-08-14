// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! m {
    () => {
        // Avoid having more than one `$crate`-named item in the same module,
        // as even though they error, they still parse as `$crate` and conflict.
        mod foo {
            struct $crate {} //~ ERROR expected identifier, found reserved identifier `$crate`
        }

        use $crate; // OK
                    //~^ WARN `$crate` may not be imported
        use $crate as $crate; //~ ERROR expected identifier, found reserved identifier `$crate`
                              //~^ WARN `$crate` may not be imported
    }
}

m!();

fn main() {}
