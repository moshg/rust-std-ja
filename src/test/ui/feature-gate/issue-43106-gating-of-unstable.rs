// Testing gating of `#[unstable]` in "weird" places.
//
// This file sits on its own because these signal errors, making
// this test incompatible with the "warnings only" nature of
// issue-43106-gating-of-builtin-attrs.rs

#![unstable                   = "1200"]
//~^ ERROR stability attributes may not be used outside of the standard library

#[unstable = "1200"]
//~^ ERROR stability attributes may not be used outside of the standard library
mod unstable {
    mod inner { #![unstable="1200"] }
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[unstable = "1200"] fn f() { }
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[unstable = "1200"] struct S;
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[unstable = "1200"] type T = S;
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[unstable = "1200"] impl S { }
    //~^ ERROR stability attributes may not be used outside of the standard library
}

fn main() {}
