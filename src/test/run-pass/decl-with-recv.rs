// -*- rust -*-
// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn main() {
    let po = core::comm::Port();
    let ch = core::comm::Chan(&po);
    core::comm::send(ch, 10);
    let i = core::comm::recv(po);
    assert (i == 10);
    core::comm::send(ch, 11);
    let j = core::comm::recv(po);
    assert (j == 11);
}
