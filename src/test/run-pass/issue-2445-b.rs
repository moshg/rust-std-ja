struct c1<T: Copy> {
    x: T,
}

impl<T: Copy> c1<T> {
    fn f1(x: int) {
    }
}

fn c1<T: Copy>(x: T) -> c1<T> {
    c1 {
        x: x
    }
}

impl<T: Copy> c1<T> {
    fn f2(x: int) {
    }
}


fn main() {
    c1::<int>(3).f1(4);
    c1::<int>(3).f2(4);
}
