fn foo(x: isize, ...) {
    //~^ ERROR: only foreign functions are allowed to be variadic
}

fn main() {}
