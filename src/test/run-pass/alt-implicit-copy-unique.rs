fn main() {
    let x = ~{mut a: ~10, b: ~20};
    match x {
      ~{a: ref mut a, b: ref mut b} => {
        assert **a == 10; (*x).a = ~30; assert **a == 30;
      }
    }
}
