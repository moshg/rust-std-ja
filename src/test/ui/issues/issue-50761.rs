// Confirm that we don't accidentally divide or mod by zero in llvm_type

// compile-pass

mod a {
    pub trait A {}
}

mod b {
    pub struct Builder {}

    pub fn new() -> Builder {
        Builder {}
    }

    impl Builder {
        pub fn with_a(&mut self, _a: fn() -> ::a::A) {}
    }
}

pub use self::b::new;

fn main() {}
