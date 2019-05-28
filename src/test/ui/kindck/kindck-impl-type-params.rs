// Issue #14061: tests the interaction between generic implementation
// parameter bounds and trait objects.

#![feature(box_syntax)]

use std::marker;

struct S<T>(marker::PhantomData<T>);

trait Gettable<T> {
    fn get(&self) -> T { panic!() }
}

impl<T: Send + Copy + 'static> Gettable<T> for S<T> {}

fn f<T>(val: T) {
    let t: S<T> = S(marker::PhantomData);
    let a = &t as &dyn Gettable<T>;
    //~^ ERROR `T` cannot be sent between threads safely
    //~| ERROR : std::marker::Copy` is not satisfied
}

fn g<T>(val: T) {
    let t: S<T> = S(marker::PhantomData);
    let a: &dyn Gettable<T> = &t;
    //~^ ERROR `T` cannot be sent between threads safely
    //~| ERROR : std::marker::Copy` is not satisfied
}

fn foo<'a>() {
    let t: S<&'a isize> = S(marker::PhantomData);
    let a = &t as &dyn Gettable<&'a isize>;
    //~^ ERROR does not fulfill
}

fn foo2<'a>() {
    let t: Box<S<String>> = box S(marker::PhantomData);
    let a = t as Box<dyn Gettable<String>>;
    //~^ ERROR : std::marker::Copy` is not satisfied
}

fn foo3<'a>() {
    struct Foo; // does not impl Copy

    let t: Box<S<Foo>> = box S(marker::PhantomData);
    let a: Box<dyn Gettable<Foo>> = t;
    //~^ ERROR : std::marker::Copy` is not satisfied
}

fn main() { }
