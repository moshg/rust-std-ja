// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unions_with_drop_fields)]

// Some traits can be derived for unions.

#![feature(untagged_unions)]

#[derive(
    Copy,
    Clone,
    Eq,
)]
union U {
    a: u8,
    b: u16,
}

impl PartialEq for U { fn eq(&self, rhs: &Self) -> bool { true } }

#[derive(
    Clone,
    Copy,
    Eq
)]
union W<T> {
    a: T,
}

impl<T> PartialEq for W<T> { fn eq(&self, rhs: &Self) -> bool { true } }

fn main() {
    let u = U { b: 0 };
    let u1 = u;
    let u2 = u.clone();
    assert!(u1 == u2);

    let w = W { a: 0 };
    let w1 = w.clone();
    assert!(w == w1);
}
