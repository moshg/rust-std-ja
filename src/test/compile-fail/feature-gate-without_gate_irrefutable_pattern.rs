// gate-test-irrefutable_let_pattern

// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    #[allow(irrefutable_let_pattern)]
    if let _ = 5 {}
    //~^ ERROR 15:12: 15:13: irrefutable if-let pattern [E0162]
}
