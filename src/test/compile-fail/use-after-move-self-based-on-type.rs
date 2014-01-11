struct S {
    x: int,
}

impl Drop for S {
    fn drop(&mut self) {}
}

impl S {
    pub fn foo(self) -> int {
        self.bar();
        return self.x;  //~ ERROR use of moved value: `self`
    }

    pub fn bar(self) {}
}

fn main() {
    let x = S { x: 1 };
    println!("{}", x.foo());
}
