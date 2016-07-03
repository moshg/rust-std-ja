// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

// Error messages for EXXXX errors.
// Each message should start and end with a new line, and be wrapped to 80 characters.
// In vim you can `:set tw=80` and use `gq` to wrap paragraphs. Use `:set tw=0` to disable.
register_long_diagnostics! {

E0533: r##"
The export_name attribute was badly formatted.

Erroneous code example:

```compile_fail,E0533
#[export_name] // error: export_name attribute has invalid format
pub fn something() {}

fn main() {}
```

The export_name attribute expects a string in order to determine the name of
the exported symbol. Example:

```
#[export_name = "some function"] // ok!
pub fn something() {}

fn main() {}
```
"##,

E0534: r##"
The inline attribute was badly used.

Erroneous code example:

```compile_fail,E0534
#[inline()] // error: expected one argument
pub fn something() {}

fn main() {}
```

The inline attribute can be used without arguments:

```
#[inline] // ok!
pub fn something() {}

fn main() {}
```

Or with arguments (and parens have to be used for this case!):

```
#[inline(always)] // ok!
pub fn something() {}

fn main() {}
```

For more information about the inline attribute, take a look here:
https://doc.rust-lang.org/reference.html#inline-attributes
"##,

E0535: r##"
An unknown argument was given to inline attribute.

Erroneous code example:

```compile_fail,E0535
#[inline(unknown)] // error: invalid argument
pub fn something() {}

fn main() {}
```

The inline attribute only knows two arguments:

 * always
 * never

All other arguments given to the inline attribute will return this error.
Example:

```
#[inline(never)] // ok!
pub fn something() {}

fn main() {}
```

For more information about the inline attribute, take a look here:
https://doc.rust-lang.org/reference.html#inline-attributes
"##,

E0536: r##"
No cfg-pattern was found for `not` statement.

Erroneous code example:

```compile_fail,E0536
#[cfg(not())] // error: expected 1 cfg-pattern
pub fn something() {}

pub fn main() {}
```

The `not` statement expects at least one cfg-pattern. Example:

```
#[cfg(not(target_os = "linux"))] // ok!
pub fn something() {}

pub fn main() {}
```

For more information about the cfg attribute, take a look here:
https://doc.rust-lang.org/reference.html#conditional-compilation
"##,

}

register_diagnostics! {
    E0537, // invalid predicate
    E0538, // multiple [same] items
    E0539, // incorrect meta item
    E0540, // multiple rustc_deprecated attributes
    E0541, // unknown meta item
    E0542, // missing 'since'
    E0543, // missing 'reason'
    E0544, // multiple stability levels
    E0545, // incorrect 'issue'
    E0546, // missing 'feature'
    E0547, // missing 'issue'
    E0548, // incorrect stability attribute type
    E0549, // rustc_deprecated attribute must be paired with either stable or unstable attribute
    E0550, // multiple deprecated attributes
    E0551, // incorrect meta item
    E0552, // unrecognized representation hint
    E0553, // unrecognized enum representation hint
    E0554, // #[feature] may not be used on the [] release channel
    E0555, // malformed feature attribute, expected #![feature(...)]
    E0556, // malformed feature, expected just one word
    E0557, // feature has been removed
}
