// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:edition-lint-paths.rs
// run-rustfix

#![feature(rust_2018_preview)]
#![deny(absolute_paths_not_starting_with_crate)]
#![allow(unused)]

extern crate edition_lint_paths;

pub mod foo {
    use edition_lint_paths;
    use ::bar::Bar;
    //~^ ERROR absolute
    //~| WARN this was previously accepted
    use super::bar::Bar2;
    use crate::bar::Bar3;

    use bar;
    //~^ ERROR absolute
    //~| WARN this was previously accepted
    use crate::{bar as something_else};

    use {Bar as SomethingElse, main};
    //~^ ERROR absolute
    //~| WARN this was previously accepted

    use crate::{Bar as SomethingElse2, main as another_main};

    pub fn test() {
    }

    pub trait SomeTrait { }
}

use bar::Bar;
//~^ ERROR absolute
//~| WARN this was previously accepted

pub mod bar {
    use edition_lint_paths as foo;
    pub struct Bar;
    pub type Bar2 = Bar;
    pub type Bar3 = Bar;
}

mod baz {
    use *;
    //~^ ERROR absolute
    //~| WARN this was previously accepted
}

impl ::foo::SomeTrait for u32 { }
//~^ ERROR absolute
//~| WARN this was previously accepted

fn main() {
    let x = ::bar::Bar;
    //~^ ERROR absolute
    //~| WARN this was previously accepted
    let x = bar::Bar;
    let x = crate::bar::Bar;
    let x = self::bar::Bar;
    foo::test();

    {
        use edition_lint_paths as bar;
        edition_lint_paths::foo();
        bar::foo();
        ::edition_lint_paths::foo();
    }
}
