// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "cdylib"]
#![deny(warnings)]

extern crate foo;

#[link(wasm_import_module = "./me")]
extern {
    #[link_name = "me_in_dep"]
    fn dep();
}

#[no_mangle]
pub extern fn foo() {
    unsafe {
        foo::dep();
        dep();
    }
}
