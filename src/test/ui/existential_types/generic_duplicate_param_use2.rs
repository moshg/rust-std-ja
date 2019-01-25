// compile-pass
#![feature(existential_type)]

fn main() {}

// test that unused generic parameters are ok
existential type Two<T, U>: 'static;

fn one<T: 'static>(t: T) -> Two<T, T> {
    t
}

fn two<T: 'static, U: 'static>(t: T, _: U) -> Two<T, U> {
    t
}
