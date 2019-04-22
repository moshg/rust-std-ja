#![feature(box_syntax)]

trait X {
    fn get_i(&self) -> isize;
}


struct B {
    i: isize
}

impl X for B {
    fn get_i(&self) -> isize {
        self.i
    }
}

struct A<'a> {
    p: &'a (X+'a)
}

fn make_a<'a>(p: &'a X) -> A<'a> {
    A { p: p }
}

fn make_make_a<'a>() -> A<'a> {
    let b: Box<B> = box B {i:1};
    let bb: &B = &*b;
    make_a(bb)  //~ ERROR cannot return value referencing local data `*b`
}

fn main() {
    let _a = make_make_a();
}
