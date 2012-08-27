


// -*- rust -*-
type compare<T> = fn@(T, T) -> bool;

fn test_generic<T: copy>(expected: T, eq: compare<T>) {
  let actual: T = match true { true => { expected }, _ => fail ~"wat" };
    assert (eq(expected, actual));
}

fn test_bool() {
    fn compare_bool(&&b1: bool, &&b2: bool) -> bool { return b1 == b2; }
    test_generic::<bool>(true, compare_bool);
}

type t = {a: int, b: int};
impl t : cmp::Eq {
    pure fn eq(&&other: t) -> bool {
        self.a == other.a && self.b == other.b
    }
}

fn test_rec() {
    fn compare_rec(t1: t, t2: t) -> bool { return t1 == t2; }
    test_generic::<t>({a: 1, b: 2}, compare_rec);
}

fn main() { test_bool(); test_rec(); }
