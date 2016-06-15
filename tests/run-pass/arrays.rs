fn empty_array() -> [u16; 0] {
    []
}

fn mini_array() -> [u16; 1] {
    [42]
}

fn big_array() -> [u16; 5] {
    [5, 4, 3, 2, 1]
}

fn array_array() -> [[u8; 2]; 3] {
    [[5, 4], [3, 2], [1, 0]]
}

fn index_unsafe() -> i32 {
    let a = [0, 10, 20, 30];
    unsafe { *a.get_unchecked(2) }
}

fn index() -> i32 {
    let a = [0, 10, 20, 30];
    a[2]
}

fn array_repeat() -> [u8; 8] {
    [42; 8]
}

fn slice_index() -> u8 {
    let arr: &[_] = &[101, 102, 103, 104, 105, 106];
    arr[5]
}

fn main() {
    assert_eq!(empty_array(), []);
    assert_eq!(index_unsafe(), 20);
    assert_eq!(index(), 20);
    assert_eq!(slice_index(), 106);
    assert_eq!(big_array(), [5, 4, 3, 2, 1]);
    assert_eq!(array_array(), [[5, 4], [3, 2], [1, 0]]);
    assert_eq!(array_repeat(), [42; 8]);
    assert_eq!(mini_array(), [42]);
}
