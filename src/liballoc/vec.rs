//! ヒープアロケートされたデータを伴う、連続する拡張可能な配列型。`Vec<T>`と書かれます。
//!
//! <!-- A contiguous growable array type with heap-allocated contents, written
//! `Vec<T>`. -->
//!
//! ベクターは`O(1)`のインデクシングができ、償却`O(1)`の (最後への) プッシュと`O(1)`の
//! (最後からの) ポップができます。
//!
//! <!-- Vectors have `O(1)` indexing, amortized `O(1)` push (to the end) and
//! `O(1)` pop (from the end). -->
//!
//! # Examples
//!
//! [`new`]を使って明示的に[`Vec<T>`]を作成することができます:
//!
//! <!-- You can explicitly create a [`Vec<T>`] with [`new`]: -->
//!
//! ```
//! let v: Vec<i32> = Vec::new();
//! ```
//!
//! ...または[`vec!`]マクロを使って:
//!
//! <!-- ...or by using the [`vec!`] macro: -->
//!
//! ```
//! let v: Vec<i32> = vec![];
//!
//! let v = vec![1, 2, 3, 4, 5];
//!
//! let v = vec![0; 10]; // 10個のゼロ
//! ```
//!
//! <!-- ``` -->
//! <!-- let v: Vec<i32> = vec![];
//!
//! let v = vec![1, 2, 3, 4, 5];
//!
//! let v = vec![0; 10]; // ten zeroes -->
//! <!-- ``` -->
//!
//! 値をベクターの最後に[`push`]することができます (必要に応じてベクターを拡張します):
//!
//! <!-- You can [`push`] values onto the end of a vector (which will grow the vector
//! as needed): -->
//!
//! ```
//! let mut v = vec![1, 2];
//!
//! v.push(3);
//! ```
//!
//! 値のポップはほとんど同じようにできます:
//!
//! <!-- Popping values works in much the same way: -->
//!
//! ```
//! let mut v = vec![1, 2];
//!
//! let two = v.pop();
//! ```
//!
//! ベクターはインデクシングもサポートします ([`Index`]と[`IndexMut`]トレイトを通して):
//!
//! <!-- Vectors also support indexing (through the [`Index`] and [`IndexMut`] traits): -->
//!
//! ```
//! let mut v = vec![1, 2, 3];
//! let three = v[2];
//! v[1] = v[1] + 5;
//! ```
//!
//! [`Vec<T>`]: ../../std/vec/struct.Vec.html
//! [`new`]: ../../std/vec/struct.Vec.html#method.new
//! [`push`]: ../../std/vec/struct.Vec.html#method.push
//! [`Index`]: ../../std/ops/trait.Index.html
//! [`IndexMut`]: ../../std/ops/trait.IndexMut.html
//! [`vec!`]: ../../std/macro.vec.html

#![stable(feature = "rust1", since = "1.0.0")]

use core::cmp::{self, Ordering};
use core::fmt;
use core::hash::{self, Hash};
use core::intrinsics::{arith_offset, assume};
use core::iter::{FromIterator, FusedIterator, TrustedLen};
use core::marker::PhantomData;
use core::mem;
use core::ops::{self, Index, IndexMut, RangeBounds};
use core::ops::Bound::{Excluded, Included, Unbounded};
use core::ptr::{self, NonNull};
use core::slice::{self, SliceIndex};

use crate::borrow::{ToOwned, Cow};
use crate::collections::CollectionAllocErr;
use crate::boxed::Box;
use crate::raw_vec::RawVec;

/// 連続する拡張可能な配列型。`Vec<T>`と書かれますが「ベクター」と発音されます。
///
/// <!-- A contiguous growable array type, written `Vec<T>` but pronounced 'vector'. -->
///
/// # Examples
///
/// ```
/// let mut vec = Vec::new();
/// vec.push(1);
/// vec.push(2);
///
/// assert_eq!(vec.len(), 2);
/// assert_eq!(vec[0], 1);
///
/// assert_eq!(vec.pop(), Some(2));
/// assert_eq!(vec.len(), 1);
///
/// vec[0] = 7;
/// assert_eq!(vec[0], 7);
///
/// vec.extend([1, 2, 3].iter().cloned());
///
/// for x in &vec {
///     println!("{}", x);
/// }
/// assert_eq!(vec, [7, 1, 2, 3]);
/// ```
///
/// 初期化を便利にするために[`vec!`]マクロが提供されています:
///
/// <!-- The [`vec!`] macro is provided to make initialization more convenient: -->
///
/// ```
/// let mut vec = vec![1, 2, 3];
/// vec.push(4);
/// assert_eq!(vec, [1, 2, 3, 4]);
/// ```
///
/// 与えられた値から各要素を初期化することも[`vec`]マクロでできます。これはメモリを確保し、別々に初期化するよりも効率的かもしれません。特にベクターをゼロで初期化するときはそうかもしれません。
///
/// <!-  It can also initialize each element of a `Vec<T>` with a given value.
/// This may be more efficient than performing allocation and initialization
/// in separate steps, especially when initializing a vector of zeros: -->
///
/// ```
/// let vec = vec![0; 5];
/// assert_eq!(vec, [0, 0, 0, 0, 0]);
///
/// // 次の方法は同値ですが、もしかするとより遅いかもしれません:
/// let mut vec1 = Vec::with_capacity(5);
/// vec1.resize(5, 0);
/// ```
///
/// <!-- ``` -->
/// <!-- let vec = vec![0; 5];
/// assert_eq!(vec, [0, 0, 0, 0, 0]);
///
/// // The following is equivalent, but potentially slower:
/// let mut vec1 = Vec::with_capacity(5);
/// vec1.resize(5, 0); -->
/// <!-- ``` -->
///
/// `Vec<T>`を効率的なスタックとして使ってください:
///
/// <!-- Use a `Vec<T>` as an efficient stack: -->
///
/// ```
/// let mut stack = Vec::new();
///
/// stack.push(1);
/// stack.push(2);
/// stack.push(3);
///
/// while let Some(top) = stack.pop() {
///     // 3, 2, 1をプリントします
///     println!("{}", top);
/// }
/// ```
///
/// <!-- ``` -->
/// <!-- let mut stack = Vec::new();
///
/// stack.push(1);
/// stack.push(2);
/// stack.push(3);
///
/// while let Some(top) = stack.pop() {
///     // Prints 3, 2, 1
///     println!("{}", top);
/// } -->
/// <!-- ``` -->
///
/// # インデクシング
///
/// <!-- # Indexing -->
///
/// `Vec`型は[`Index`]トレイトを実装しているので、インデックスを使って値にアクセスすることができます。次の例はより明白でしょう:
///
/// <!-- The `Vec` type allows to access values by index, because it implements the
/// [`Index`] trait. An example will be more explicit: -->
///
/// ```
/// let v = vec![0, 2, 4, 6];
/// println!("{}", v[1]); // 「2」を表示します
/// ```
///
/// <!-- ``` -->
/// <!-- let v = vec![0, 2, 4, 6];
/// println!("{}", v[1]); // it will display '2' -->
/// <!-- ``` -->
///
/// しかし注意してください: `Vec`に含まれないインデックスにアクセスしようとすると、あなたのソフトウェアはパニックします！このようなことはできません:
///
/// <!-- However be careful: if you try to access an index which isn't in the `Vec`,
/// your software will panic! You cannot do this: -->
///
/// ```should_panic
/// let v = vec![0, 2, 4, 6];
/// println!("{}", v[6]); // パニックします！
/// ```
///
/// <!-- ```should_panic -->
/// <!-- let v = vec![0, 2, 4, 6];
/// println!("{}", v[6]); // it will panic! -->
/// <!-- ``` -->
///
/// 結論: インデクシングの前にそのインデックスが本当に存在するかを常に確認してください。
///
/// <!-- In conclusion: always check if the index you want to get really exists
/// before doing it. -->
///
/// # スライシング
///
/// <!-- # Slicing -->
///
/// `Vec`はミュータブルになり得ます。一方、スライスは読み取り専用オブジェクトです。スライスを得るには、`&`を使ってください。例:
///
/// <!-- A `Vec` can be mutable. Slices, on the other hand, are read-only objects.
/// To get a slice, use `&`. Example: -->
///
/// ```
/// fn read_slice(slice: &[usize]) {
///     // ...
/// }
///
/// let v = vec![0, 1];
/// read_slice(&v);
///
/// // ... そしてこれだけです！
/// // このようにもできます:
/// let x : &[usize] = &v;
/// ```
///
/// <!-- ``` -->
/// <!-- fn read_slice(slice: &[usize]) {
///     // ...
/// }
///
/// let v = vec![0, 1];
/// read_slice(&v);
///
/// // ... and that's all!
/// // you can also do it like this:
/// let x : &[usize] = &v; -->
/// <!-- ``` -->
///
/// Rustにおいて、単に読み取りアクセスできるようにしたいときはベクターよりもスライスを引数として渡すことが一般的です。[`String`]と[`&str`]についても同様です。
///
/// <!-- In Rust, it's more common to pass slices as arguments rather than vectors
/// when you just want to provide a read access. The same goes for [`String`] and
/// [`&str`]. -->
///
/// # 容量とメモリの再確保
///
/// <!-- # Capacity and reallocation -->
///
/// ベクターの容量 (capacity) とは将来ベクターに追加される要素のためにアロケートされる領域の量のことです。これをベクターの*長さ (length)* と混同しないでください。ベクターの長さとはそのベクターに入っている実際の要素の個数のことです。ベクターの長さが容量を超えると、容量は自動的に増えますが、その要素は再確保されなければなりません。
///
/// <!-- The capacity of a vector is the amount of space allocated for any future
/// elements that will be added onto the vector. This is not to be confused with
/// the *length* of a vector, which specifies the number of actual elements
/// within the vector. If a vector's length exceeds its capacity, its capacity
/// will automatically be increased, but its elements will have to be
/// reallocated. -->
///
/// 例えば、容量10で長さ0のベクターは追加10要素分の領域をもった空のベクターです。10またはそれ以下の要素をベクターにプッシュしてもベクターの容量は変わりませんし、メモリの再確保も起きません。しかし、ベクターの長さが11まで増加すると、ベクターはメモリを再確保しなければならず、遅いです。このため、ベクターがどれだけ大きくなるかが予期できるときは常に[`Vec::with_capacity`]を利用することが推奨されます。
///
/// <!-- For example, a vector with capacity 10 and length 0 would be an empty vector
/// with space for 10 more elements. Pushing 10 or fewer elements onto the
/// vector will not change its capacity or cause reallocation to occur. However,
/// if the vector's length is increased to 11, it will have to reallocate, which
/// can be slow. For this reason, it is recommended to use [`Vec::with_capacity`]
/// whenever possible to specify how big the vector is expected to get. -->
///
/// # 保証
///
/// <!-- # Guarantees -->
///
/// そのとてつもなく基本的な性質のために、`Vec`はデザインについて多くのことを保証します。`Vec`は可能な限りロー・オーバーヘッドであり、アンセーフコードからプリミティブな方法で正しく操作することができます。これらの保証は制限のない`Vec<T>`を指すことに注意してください。もし追加の型パラメータが追加されれば (例えばカスタムアロケータのサポートのために)、`Vec`のデフォルトを上書きすることで動作が変わるかもしれません。
///
/// <!-- Due to its incredibly fundamental nature, `Vec` makes a lot of guarantees
/// about its design. This ensures that it's as low-overhead as possible in
/// the general case, and can be correctly manipulated in primitive ways
/// by unsafe code. Note that these guarantees refer to an unqualified `Vec<T>`.
/// If additional type parameters are added (e.g., to support custom allocators),
/// overriding their defaults may change the behavior. -->
///
/// 最も基本的なこととして、`Vec`は (ポインタ, 容量, 長さ) の三つ組であり将来的にも常にそうです。それ以上でも以下でもありません。これらのフィールドの順序は完全に未規定であり、その値を変更するためには適切なメソッドを使うべきです。ポインタは決してヌルにはなりません。ですので、この型はヌルポインタ最適化されます。
///
/// <!-- Most fundamentally, `Vec` is and always will be a (pointer, capacity, length)
/// triplet. No more, no less. The order of these fields is completely
/// unspecified, and you should use the appropriate methods to modify these.
/// The pointer will never be null, so this type is null-pointer-optimized. -->
///
/// しかし、ポインタは実際には確保されたメモリを指さないかもしれません。特に、空のベクターの作成を[`Vec::new`]や[`vec![]`][`vec!`]や[`Vec::with_capacity(0)`][`Vec::with_capacity`]により行ったり、[`shrink_to_fit`]の空のベクターでの呼び出しから行ったりするとき、メモリを確保しません。同様に、ゼロサイズ型を`Vec`に格納するとき、それらのための領域を確保しません。*この場合`Vec`は[`capacity`]がゼロであると伝えないかもしれないことに注意してください。*`Vec`は[`mem::size_of::<T>`]`() * capacity() > 0`のとき、またそのときに限りメモリを確保します。一般に、`Vec`のアロケーションの詳細はとても微妙です &mdash; もし`Vec`を使ってメモリを確保し他の何か (アンセーフコードに渡す、またはメモリが背後にあるあなた自身のコレクションのいずれか) に使うつもりなら、必ず`from_raw_parts`を使って`Vec`を復元しドロップすることでそのメモリを解放してください。
///
/// <!-- However, the pointer may not actually point to allocated memory. In particular,
/// if you construct a `Vec` with capacity 0 via [`Vec::new`], [`vec![]`][`vec!`],
/// [`Vec::with_capacity(0)`][`Vec::with_capacity`], or by calling [`shrink_to_fit`]
/// on an empty Vec, it will not allocate memory. Similarly, if you store zero-sized
/// types inside a `Vec`, it will not allocate space for them. *Note that in this case
/// the `Vec` may not report a [`capacity`] of 0*. `Vec` will allocate if and only
/// if [`mem::size_of::<T>`]`() * capacity() > 0`. In general, `Vec`'s allocation
/// details are very subtle &mdash; if you intend to allocate memory using a `Vec`
/// and use it for something else (either to pass to unsafe code, or to build your
/// own memory-backed collection), be sure to deallocate this memory by using
/// `from_raw_parts` to recover the `Vec` and then dropping it. -->
///
/// `Vec`がメモリを確保*している*とき、`Vec`が指すメモリはヒープにあり (Rustがデフォルトで使うよう設定されたアロケータによって定義されるように) 、ポインタは[`len`]個の初期化された、連続する (スライスに強制したときと同じ) 順に並んだ要素を指し、[`capacity`]` - `[`len`]個の論理的な初期化がされていない、連続する要素が後続します。
///
/// <!-- If a `Vec` *has* allocated memory, then the memory it points to is on the heap
/// (as defined by the allocator Rust is configured to use by default), and its
/// pointer points to [`len`] initialized, contiguous elements in order (what
/// you would see if you coerced it to a slice), followed by [`capacity`]` -
/// `[`len`] logically uninitialized, contiguous elements. -->
///
/// `Vec`は要素を実際にはスタックに格納する"small optimization"を決して行いません。それは2つの理由のためです:
///
/// <!-- `Vec` will never perform a "small optimization" where elements are actually
/// stored on the stack for two reasons: -->
///
/// * アンセーフコードが`Vec`を扱うことを難しくします。`Vec`が単にムーブされるとき`Vec`の中身は安定したアドレスを持たないでしょう。そして`Vec`が実際に確保されたメモリを持っているかを決定することが難しくなるでしょう。
///
/// <!-- * It would make it more difficult for unsafe code to correctly manipulate
///   a `Vec`. The contents of a `Vec` wouldn't have a stable address if it were
///   only moved, and it would be more difficult to determine if a `Vec` had
///   actually allocated memory. -->
///
/// * アクセス毎に追加の分岐を招くことで、一般的な状況で不利になるでしょう。
///
/// <!-- * It would penalize the general case, incurring an additional branch
///   on every access. -->
///
/// `Vec`は決して自動で縮みません。まったく空のときでさえもです。これによりメモリの不要な確保や解放が発生しないことが確実になります。`Vec`を空にし、それから同じ[`len`]以下まで埋め直すことがアロケータへの呼び出しを招くことは決してありません。もし使われていないメモリを解放したいなら、[`shrink_to_fit`][`shrink_to_fit`]を使ってください。
///
/// <!-- `Vec` will never automatically shrink itself, even if completely empty. This
/// ensures no unnecessary allocations or deallocations occur. Emptying a `Vec`
/// and then filling it back up to the same [`len`] should incur no calls to
/// the allocator. If you wish to free up unused memory, use
/// [`shrink_to_fit`][`shrink_to_fit`]. -->
///
/// 通知された容量が十分なとき[`push`]と[`insert`]は絶対にメモリを(再)確保しません。[`push`]と[`insert`]は[`len`]` == `[`capacity`]のときメモリを(再)確保*します*。つまり、知らされる容量は完全に正確で、信頼することができます。必要に応じて`Vec`によって確保されたメモリを手動で解放することもできます。バラバラに挿入するメソッドは必要がない場合もメモリの再確保をしてしまう*かも*しれません。
///
/// <!-- [`push`] and [`insert`] will never (re)allocate if the reported capacity is
/// sufficient. [`push`] and [`insert`] *will* (re)allocate if
/// [`len`]` == `[`capacity`]. That is, the reported capacity is completely
/// accurate, and can be relied on. It can even be used to manually free the memory
/// allocated by a `Vec` if desired. Bulk insertion methods *may* reallocate, even
/// when not necessary. -->
///
/// `Vec`は満杯になったときや[`reserve`]が呼び出されたときにメモリの再確保をする際の特定の拡張戦略をまったく保証しません。現在の戦略は基本的であり、定数でない増大係数が望ましいことが証明されるかもしれません。どのような戦略を使うとしても当然[`push`]が償却`O(1)`であることは保証されます。
///
/// <!-- `Vec` does not guarantee any particular growth strategy when reallocating
/// when full, nor when [`reserve`] is called. The current strategy is basic
/// and it may prove desirable to use a non-constant growth factor. Whatever
/// strategy is used will of course guarantee `O(1)` amortized [`push`]. -->
///
/// `vec![x; n]`と`vec![a, b, c, d]`と[`Vec::with_capacity(n)`][`Vec::with_capacity`]は全て正確に要求した容量を持つ`Vec`を提供します。([`vec`]マクロの場合のように) [`len`]` == `[`capacity`]ならば、`Vec<T>`はメモリの再確保や要素の移動なしに[`Box<[T]>`][owned slice]と相互に変換できます。
///
/// <!-- `vec![x; n]`, `vec![a, b, c, d]`, and
/// [`Vec::with_capacity(n)`][`Vec::with_capacity`], will all produce a `Vec`
/// with exactly the requested capacity. If [`len`]` == `[`capacity`],
/// (as is the case for the [`vec!`] macro), then a `Vec<T>` can be converted to
/// and from a [`Box<[T]>`][owned slice] without reallocating or moving the elements. -->
///
/// `Vec`は特に取り除かれたデータを上書きするとは限りませんが、特に取っておくとも限りません。未初期化メモリは必要ならばいつでも`Vec`が使ってよい一時領域です。`Vec`は一般に最も効率がいいような、または実装しやすいような動作をします。セキュリティ目的で取り除かれたデータが消去されることを頼ってはいけません。`Vec`をドロップしたとしても、その`Vec`のバッファは他の`Vec`に再利用されるかもしれません。`Vec`のメモリを最初にゼロにしたとしても、オプティマイザが保護すべき副作用だと考えないためにそれが実際には起こらないかもしれません。しかしながら、私達が破壊しないであろうケースが一つあります: `unsafe`コードを使って過剰分の容量に書き込んでから、それに合うように長さを増加させることは常に正当です.
///
/// <!-- `Vec` will not specifically overwrite any data that is removed from it,
/// but also won't specifically preserve it. Its uninitialized memory is
/// scratch space that it may use however it wants. It will generally just do
/// whatever is most efficient or otherwise easy to implement. Do not rely on
/// removed data to be erased for security purposes. Even if you drop a `Vec`, its
/// buffer may simply be reused by another `Vec`. Even if you zero a `Vec`'s memory
/// first, that may not actually happen because the optimizer does not consider
/// this a side-effect that must be preserved. There is one case which we will
/// not break, however: using `unsafe` code to write to the excess capacity,
/// and then increasing the length to match, is always valid. -->
///
/// `Vec`は現在、要素をドロップする順序を保証しません。過去に順序を変更したことがあり、また変更するかもしれません。
///
/// <!-- `Vec` does not currently guarantee the order in which elements are dropped.
/// The order has changed in the past and may change again. -->
///
/// [`vec!`]: ../../std/macro.vec.html
/// [`Index`]: ../../std/ops/trait.Index.html
/// [`String`]: ../../std/string/struct.String.html
/// [`&str`]: ../../std/primitive.str.html
/// [`Vec::with_capacity`]: ../../std/vec/struct.Vec.html#method.with_capacity
/// [`Vec::new`]: ../../std/vec/struct.Vec.html#method.new
/// [`shrink_to_fit`]: ../../std/vec/struct.Vec.html#method.shrink_to_fit
/// [`capacity`]: ../../std/vec/struct.Vec.html#method.capacity
/// [`mem::size_of::<T>`]: ../../std/mem/fn.size_of.html
/// [`len`]: ../../std/vec/struct.Vec.html#method.len
/// [`push`]: ../../std/vec/struct.Vec.html#method.push
/// [`insert`]: ../../std/vec/struct.Vec.html#method.insert
/// [`reserve`]: ../../std/vec/struct.Vec.html#method.reserve
/// [owned slice]: ../../std/boxed/struct.Box.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Vec<T> {
    buf: RawVec<T>,
    len: usize,
}

////////////////////////////////////////////////////////////////////////////////
// Inherent methods
////////////////////////////////////////////////////////////////////////////////

impl<T> Vec<T> {
    /// 新しい空の`Vec<T>`を作成します。
    ///
    /// <!-- Constructs a new, empty `Vec<T>`. -->
    ///
    /// ベクターは要素をプッシュされるまでメモリを確保しません。
    ///
    /// <!-- The vector will not allocate until elements are pushed onto it. -->
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(unused_mut)]
    /// let mut vec: Vec<i32> = Vec::new();
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_vec_new")]
    pub const fn new() -> Vec<T> {
        Vec {
            buf: RawVec::new(),
            len: 0,
        }
    }

    /// 新しい空の`Vec<T>`を指定された容量で作成します。
    ///
    /// <!-- Constructs a new, empty `Vec<T>` with the specified capacity. -->
    ///
    /// 返されたベクターはメモリの再確保なしにちょうど`capacity`個の要素を格納することができます。`capacity`が0ならばメモリを確保しません。
    ///
    /// <!-- The vector will be able to hold exactly `capacity` elements without
    /// reallocating. If `capacity` is 0, the vector will not allocate. -->
    ///
    /// 返されたベクターは指定された*容量*を持ちますが、長さが0であることに注意することが大切です。長さと容量の違いの説明については*[容量とメモリの再確保]*を見てください。
    ///
    /// <!-- It is important to note that although the returned vector has the
    /// *capacity* specified, the vector will have a zero *length*. For an
    /// explanation of the difference between length and capacity, see
    /// *[Capacity and reallocation]*. -->
    ///
    /// [容量とメモリの再確保]: #容量とメモリの再確保
    ///
    /// <!-- [Capacity and reallocation]: #capacity-and-reallocation -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = Vec::with_capacity(10);
    ///
    /// // 0より大きい容量を持ちますが、要素は持ちません。
    /// assert_eq!(vec.len(), 0);
    ///
    /// これらは全てメモリの再確保なしに行われます...
    /// for i in 0..10 {
    ///     vec.push(i)
    /// }
    ///
    /// // ...しかしこれはメモリを再確保するかもしれません
    /// vec.push(11);
    /// ```
    ///
    /// <!-- ``` -->
    /// <!-- let mut vec = Vec::with_capacity(10);
    ///
    /// // The vector contains no items, even though it has capacity for more
    /// assert_eq!(vec.len(), 0);
    ///
    /// // These are all done without reallocating...
    /// for i in 0..10 {
    ///     vec.push(i);
    /// }
    ///
    /// // ...but this may make the vector reallocate
    /// vec.push(11); -->
    /// <!-- ``` -->
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(capacity: usize) -> Vec<T> {
        Vec {
            buf: RawVec::with_capacity(capacity),
            len: 0,
        }
    }

    /// `Vec<T>`を他のベクターの生の構成要素から直接作成します。
    ///
    /// <!-- Creates a `Vec<T>` directly from the raw components of another vector. -->
    ///
    /// # Safety
    ///
    /// このメソッドは非常にアンセーフです。いくつものチェックされない不変量があるためです:
    ///
    /// <!-- This is highly unsafe, due to the number of invariants that aren't
    /// checked: -->
    ///
    /// * `ptr`は以前に[`String`]/`Vec<T>`で確保されいる必要があります (少なくともそうでなければ非常に不適切です)。
    /// * `ptr`の`T`はアロケートされたときと同じサイズ、同じアラインメントである必要があります。
    /// * `length`は`capacity`以下である必要があります。
    /// * `capacity`はポインタがアロケートされたときの容量である必要があります。
    ///
    /// <!-- * `ptr` needs to have been previously allocated via [`String`]/`Vec<T>`
    ///   (at least, it's highly likely to be incorrect if it wasn't).
    /// * `ptr`'s `T` needs to have the same size and alignment as it was allocated with.
    /// * `length` needs to be less than or equal to `capacity`.
    /// * `capacity` needs to be the capacity that the pointer was allocated with. -->
    ///
    /// これらに違反すると、アロケータの内部データ構造を破壊することになるかもしれません。例えば`Vec<u8>`をCの`char`配列と`size_t`から作成することは安全では**ありません**。
    ///
    /// <!-- Violating these may cause problems like corrupting the allocator's
    /// internal data structures. For example it is **not** safe
    /// to build a `Vec<u8>` from a pointer to a C `char` array and a `size_t`. -->
    ///
    /// `ptr`の所有権は有効に`Vec<T>`に移り、その`Vec<T>`は思うままにメモリの破棄や再確保やポインタの指すメモリの内容の変更する権利を得ます。この関数を呼んだ後にポインタを使うことがないことを確実にしてください。
    ///
    /// <!-- The ownership of `ptr` is effectively transferred to the
    /// `Vec<T>` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function. -->
    ///
    /// [`String`]: ../../std/string/struct.String.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ptr;
    /// use std::mem;
    ///
    /// fn main() {
    ///     let mut v = vec![1, 2, 3];
    ///
    ///     // さまざまな`v`の情報の重要な断片を抜き出します
    ///     let p = v.as_mut_ptr();
    ///     let len = v.len();
    ///     let cap = v.capacity();
    ///
    ///     unsafe {
    ///         // `v`をvoidにキャストします: デストラクタは走りません。
    ///         // よって`p`が指す確保されたメモリを完全に管理することになります。
    ///         mem::forget(v);
    ///
    ///         // メモリを4, 5, 6で上書きします
    ///         for i in 0..len as isize {
    ///             ptr::write(p.offset(i), 4 + i);
    ///         }
    ///
    ///         // 全てを合わせてVecに戻します
    ///         let rebuilt = Vec::from_raw_parts(p, len, cap);
    ///         assert_eq!(rebuilt, [4, 5, 6]);
    ///     }
    /// }
    /// ```
    ///
    /// <!-- ``` -->
    /// <!-- use std::ptr;
    /// use std::mem;
    ///
    /// fn main() {
    ///     let mut v = vec![1, 2, 3];
    ///
    ///     // Pull out the various important pieces of information about `v`
    ///     let p = v.as_mut_ptr();
    ///     let len = v.len();
    ///     let cap = v.capacity();
    ///
    ///     unsafe {
    ///         // Cast `v` into the void: no destructor run, so we are in
    ///         // complete control of the allocation to which `p` points.
    ///         mem::forget(v);
    ///
    ///         // Overwrite memory with 4, 5, 6
    ///         for i in 0..len as isize {
    ///             ptr::write(p.offset(i), 4 + i);
    ///         }
    ///
    ///         // Put everything back together into a Vec
    ///         let rebuilt = Vec::from_raw_parts(p, len, cap);
    ///         assert_eq!(rebuilt, [4, 5, 6]);
    ///     }
    /// } -->
    /// <!-- ``` -->
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Vec<T> {
        Vec {
            buf: RawVec::from_raw_parts(ptr, capacity),
            len: length,
        }
    }

    /// ベクターがメモリの再確保なしに保持することのできる要素の数を返します。
    ///
    /// <!-- Returns the number of elements the vector can hold without
    /// reallocating. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let vec: Vec<i32> = Vec::with_capacity(10);
    /// assert_eq!(vec.capacity(), 10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn capacity(&self) -> usize {
        self.buf.cap()
    }

    /// 少なくとも`additional`個の要素与えられた`Vec<T>`に挿入できるように容量を確保します。コレクションは頻繁なメモリの再確保を避けるために領域を多めに確保するかもしれません。`reserve`を呼び出した後、容量は`self.len() + addtional`以上になります。容量が既に十分なときは何もしません。
    ///
    /// <!-- Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `Vec<T>`. The collection may reserve more space to avoid
    /// frequent reallocations. After calling `reserve`, capacity will be
    /// greater than or equal to `self.len() + additional`. Does nothing if
    /// capacity is already sufficient. -->
    ///
    /// # Panics
    ///
    /// 新たな容量が`usize`に収まらないときパニックします。
    ///
    /// <!-- Panics if the new capacity overflows `usize`. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1];
    /// vec.reserve(10);
    /// assert!(vec.capacity() >= 11);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve(&mut self, additional: usize) {
        self.buf.reserve(self.len, additional);
    }

    /// ちょうど`additional`個の要素を与えられた`Vec<T>`に挿入できるように最低限の容量を確保します。`reserve_exact`を呼び出した後、容量は`self.len() + additional`以上になります。容量が既に十分なときは何もしません。
    ///
    /// <!-- Reserves the minimum capacity for exactly `additional` more elements to
    /// be inserted in the given `Vec<T>`. After calling `reserve_exact`,
    /// capacity will be greater than or equal to `self.len() + additional`.
    /// Does nothing if the capacity is already sufficient. -->
    ///
    /// アロケータは要求したより多くの領域を確保するかもしれないことに注意してください。そのためキャパシティが正確に最低限であることに依存することはできません。将来の挿入が予期される場合`reserve`のほうが好ましいです。
    ///
    /// <!-- Note that the allocator may give the collection more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer `reserve` if future insertions are expected. -->
    ///
    /// # Panics
    ///
    /// 新たな容量が`usize`に収まらないときパニックします。
    ///
    /// <!-- Panics if the new capacity overflows `usize`. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1];
    /// vec.reserve_exact(10);
    /// assert!(vec.capacity() >= 11);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.buf.reserve_exact(self.len, additional);
    }

    /// 少なくとも`additional`個の要素を与えられた`Vec<T>`に挿入できるように容量を確保することを試みます。コレクションは頻繁なリメモリの再確保を避けるために領域を多めに確保するかもしれません。`reserve`を呼び出した後、容量は`self.len() + addtional`以上になります。容量が既に十分なときは何もしません。
    ///
    /// <!-- Tries to reserve capacity for at least `additional` more elements to be inserted
    /// in the given `Vec<T>`. The collection may reserve more space to avoid
    /// frequent reallocations. After calling `reserve`, capacity will be
    /// greater than or equal to `self.len() + additional`. Does nothing if
    /// capacity is already sufficient. -->
    ///
    /// # Errors
    ///
    /// 容量がオーバーフローする、またはアロケータが失敗を通知するときエラーを返します。
    ///
    /// <!-- If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned. -->
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(try_reserve)]
    /// use std::collections::CollectionAllocErr;
    ///
    /// fn process_data(data: &[u32]) -> Result<Vec<u32>, CollectionAllocErr> {
    ///     let mut output = Vec::new();
    ///
    ///     // 予めメモリを確保し、できなければ脱出する
    ///     output.try_reserve(data.len())?;
    ///
    ///     // 今、これが複雑な作業の途中でOOMし得ないことがわかっています
    ///     // Now we know this can't OOM in the middle of our complex work
    ///     output.extend(data.iter().map(|&val| {
    ///         val * 2 + 5 // すごく複雑
    ///     }));
    ///
    ///     Ok(output)
    /// }
    /// # process_data(&[1, 2, 3]).expect("なんでテストフレームワークが12バイトでOOMしてるんだ？");
    /// ```
    ///
    /// <!-- ``` -->
    /// <!-- #![feature(try_reserve)]
    /// use std::collections::CollectionAllocErr;
    ///
    /// fn process_data(data: &[u32]) -> Result<Vec<u32>, CollectionAllocErr> {
    ///     let mut output = Vec::new();
    ///
    ///     // Pre-reserve the memory, exiting if we can't
    ///     output.try_reserve(data.len())?;
    ///
    ///     // Now we know this can't OOM in the middle of our complex work
    ///     output.extend(data.iter().map(|&val| {
    ///         val * 2 + 5 // very complicated
    ///     }));
    ///
    ///     Ok(output)
    /// }
    /// # process_data(&[1, 2, 3]).expect("why is the test harness OOMing on 12 bytes?"); -->
    /// <!-- ``` -->
    #[unstable(feature = "try_reserve", reason = "new API", issue="48043")]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), CollectionAllocErr> {
        self.buf.try_reserve(self.len, additional)
    }

    /// ちょうど`additional`個の要素与えられた`Vec<T>`に挿入できるように最低限の容量を確保することを試みます。コレクションは頻繁なメモリの再確保を避けるために領域を多めに確保するかもしれません。`reserve_exact`を呼び出した後、容量は`self.len() + addtional`以上になります。容量が既に十分なときは何もしません。
    ///
    /// <!-- Tries to reserves the minimum capacity for exactly `additional` more elements to
    /// be inserted in the given `Vec<T>`. After calling `reserve_exact`,
    /// capacity will be greater than or equal to `self.len() + additional`.
    /// Does nothing if the capacity is already sufficient. -->
    ///
    /// アロケータは要求したより多くの領域を確保するかもしれないことに注意してください。そのためキャパシティが正確に最小であることに依存することはできません。将来の挿入が予期される場合`reserve`のほうが好ましいです。
    ///
    /// <!-- Note that the allocator may give the collection more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer `reserve` if future insertions are expected. -->
    ///
    /// # Errors
    ///
    /// 容量がオーバーフローする、またはアロケータが失敗を通知するときエラーを返します。
    ///
    /// <!-- If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned. -->
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(try_reserve)]
    /// use std::collections::CollectionAllocErr;
    ///
    /// fn process_data(data: &[u32]) -> Result<Vec<u32>, CollectionAllocErr> {
    ///     let mut output = Vec::new();
    ///
    ///     // 予めメモリを確保し、できなければ脱出する
    ///     output.try_reserve(data.len())?;
    ///
    ///     // 今、これが複雑な作業の途中でOOMし得ないことがわかっています
    ///     // Now we know this can't OOM in the middle of our complex work
    ///     output.extend(data.iter().map(|&val| {
    ///         val * 2 + 5 // すごく複雑
    ///     }));
    ///
    ///     Ok(output)
    /// }
    /// # process_data(&[1, 2, 3]).expect("なんでテストフレームワークが12バイトでOOMしてるんだ？");
    /// ```
    ///
    /// <!-- ``` -->
    /// <!-- #![feature(try_reserve)]
    /// use std::collections::CollectionAllocErr;
    ///
    /// fn process_data(data: &[u32]) -> Result<Vec<u32>, CollectionAllocErr> {
    ///     let mut output = Vec::new();
    ///
    ///     // Pre-reserve the memory, exiting if we can't
    ///     output.try_reserve(data.len())?;
    ///
    ///     // Now we know this can't OOM in the middle of our complex work
    ///     output.extend(data.iter().map(|&val| {
    ///         val * 2 + 5 // very complicated
    ///     }));
    ///
    ///     Ok(output)
    /// }
    /// # process_data(&[1, 2, 3]).expect("why is the test harness OOMing on 12 bytes?"); -->
    /// <!-- ``` -->
    #[unstable(feature = "try_reserve", reason = "new API", issue="48043")]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), CollectionAllocErr>  {
        self.buf.try_reserve_exact(self.len, additional)
    }

    /// ベクターの容量を可能な限り縮小します。
    ///
    /// <!-- Shrinks the capacity of the vector as much as possible. -->
    ///
    /// 可能な限り長さの近くまで領域を破棄しますが、アロケータはまだ少し要素を格納できる領域があるとベクターに通知するかもしれません。
    ///
    /// <!-- It will drop down as close as possible to the length but the allocator
    /// may still inform the vector that there is space for a few more elements. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = Vec::with_capacity(10);
    /// vec.extend([1, 2, 3].iter().cloned());
    /// assert_eq!(vec.capacity(), 10);
    /// vec.shrink_to_fit();
    /// assert!(vec.capacity() >= 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn shrink_to_fit(&mut self) {
        if self.capacity() != self.len {
            self.buf.shrink_to_fit(self.len);
        }
    }

    /// 下限付きでベクターを縮小します。
    ///
    /// <!-- Shrinks the capacity of the vector with a lower bound. -->
    ///
    /// 容量は最低でも長さと与えられた値以上になります。
    ///
    /// <!-- The capacity will remain at least as large as both the length
    /// and the supplied value. -->
    ///
    /// 現在の容量が与えられた値より小さい場合パニックします。
    ///
    /// <!-- Panics if the current capacity is smaller than the supplied
    /// minimum capacity. -->
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(shrink_to)]
    /// let mut vec = Vec::with_capacity(10);
    /// vec.extend([1, 2, 3].iter().cloned());
    /// assert_eq!(vec.capacity(), 10);
    /// vec.shrink_to(4);
    /// assert!(vec.capacity() >= 4);
    /// vec.shrink_to(0);
    /// assert!(vec.capacity() >= 3);
    /// ```
    #[unstable(feature = "shrink_to", reason = "new API", issue="56431")]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.buf.shrink_to_fit(cmp::max(self.len, min_capacity));
    }

    /// ベクターを[`Box<[T]>`][owned slice]に変換します。
    ///
    /// <!-- Converts the vector into [`Box<[T]>`][owned slice]. -->
    ///
    /// このメソッドが余剰の容量を落とすことに注意してください。
    ///
    /// <!-- Note that this will drop any excess capacity. -->
    ///
    /// [owned slice]: ../../std/boxed/struct.Box.html
    ///
    /// # Examples
    ///
    /// ```
    /// let v = vec![1, 2, 3];
    ///
    /// let slice = v.into_boxed_slice();
    /// ```
    ///
    /// 余剰の容量は取り除かれます:
    ///
    /// <!-- Any excess capacity is removed: -->
    ///
    /// ```
    /// let mut vec = Vec::with_capacity(10);
    /// vec.extend([1, 2, 3].iter().cloned());
    ///
    /// assert_eq!(vec.capacity(), 10);
    /// let slice = vec.into_boxed_slice();
    /// assert_eq!(slice.into_vec().capacity(), 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_boxed_slice(mut self) -> Box<[T]> {
        unsafe {
            self.shrink_to_fit();
            let buf = ptr::read(&self.buf);
            mem::forget(self);
            buf.into_box()
        }
    }

    /// 初めの`len`個の要素を残し、残りを捨てることでベクターを短くします。
    ///
    /// <!-- Shortens the vector, keeping the first `len` elements and dropping
    /// the rest. -->
    ///
    /// `len`がベクターの現在の長さより大きいときは何の効果もありません。
    ///
    /// <!-- If `len` is greater than the vector's current length, this has no
    /// effect. -->
    ///
    /// [`drain`]メソッドは`truncate`をエミュレートできますが、余剰の要素を捨てる代わりに返すことになります。
    ///
    /// <!-- The [`drain`] method can emulate `truncate`, but causes the excess
    /// elements to be returned instead of dropped. -->
    ///
    /// このメソッドはベクターの確保された容量に影響しないことに注意してください。
    ///
    /// <!-- Note that this method has no effect on the allocated capacity
    /// of the vector. -->
    ///
    /// # Examples
    ///
    /// 五要素のベクターの二要素への切り詰め:
    ///
    /// <!-- Truncating a five element vector to two elements: -->
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3, 4, 5];
    /// vec.truncate(2);
    /// assert_eq!(vec, [1, 2]);
    /// ```
    ///
    /// `len`がベクターの現在の長さより大きいときは切り詰めが起きません:
    ///
    /// <!-- No truncation occurs when `len` is greater than the vector's current
    /// length: -->
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3];
    /// vec.truncate(8);
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    ///
    /// `len == 0`のときの切り詰めは[`clear`]の呼び出しと同値です。
    ///
    /// <!-- Truncating when `len == 0` is equivalent to calling the [`clear`]
    /// method. -->
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3];
    /// vec.truncate(0);
    /// assert_eq!(vec, []);
    /// ```
    ///
    /// [`clear`]: #method.clear
    /// [`drain`]: #method.drain
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn truncate(&mut self, len: usize) {
        let current_len = self.len;
        unsafe {
            let mut ptr = self.as_mut_ptr().add(self.len);
            // Set the final length at the end, keeping in mind that
            // dropping an element might panic. Works around a missed
            // optimization, as seen in the following issue:
            // https://github.com/rust-lang/rust/issues/51802
            let mut local_len = SetLenOnDrop::new(&mut self.len);

            // drop any extra elements
            for _ in len..current_len {
                local_len.decrement_len(1);
                ptr = ptr.offset(-1);
                ptr::drop_in_place(ptr);
            }
        }
    }

    /// ベクター全体を含むスライスを抜き出します。
    ///
    /// <!-- Extracts a slice containing the entire vector. -->
    ///
    /// `&s[..]`と同値です。
    ///
    /// <!-- Equivalent to `&s[..]`. -->
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{self, Write};
    /// let buffer = vec![1, 2, 3, 5, 8];
    /// io::sink().write(buffer.as_slice()).unwrap();
    /// ```
    #[inline]
    #[stable(feature = "vec_as_slice", since = "1.7.0")]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// ベクター全体のミュータブルなスライスを抜き出します。
    ///
    /// <!-- Extracts a mutable slice of the entire vector. -->
    ///
    /// `&mut s[..]`と同値です。
    ///
    /// <!-- Equivalent to `&mut s[..]`. -->
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{self, Read};
    /// let mut buffer = vec![0; 3];
    /// io::repeat(0b101).read_exact(buffer.as_mut_slice()).unwrap();
    /// ```
    #[inline]
    #[stable(feature = "vec_as_slice", since = "1.7.0")]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// ベクターの長さを`new_len`に強制します。
    ///
    /// <!-- Forces the length of the vector to `new_len`. -->
    ///
    /// これは型の通常の不変量を全く保存しない低レベル操作です。通常、ベクターの長さの変更はこのメソッドよりも[`truncate`]や[`resize`]や[`extend`]や[`clear`]のような安全な操作を利用することで行われます。
    ///
    /// <!-- This is a low-level operation that maintains none of the normal
    /// invariants of the type. Normally changing the length of a vector
    /// is done using one of the safe operations instead, such as
    /// [`truncate`], [`resize`], [`extend`], or [`clear`]. -->
    ///
    /// [`truncate`]: #method.truncate
    /// [`resize`]: #method.resize
    /// [`extend`]: #method.extend-1
    /// [`clear`]: #method.clear
    ///
    /// # Safety
    ///
    /// - `new_len`は[`capacity()`]以下でなければなりません。
    /// - `old_len..new_len`の要素は初期化されていなければなりません。
    ///
    /// <!-- - `new_len` must be less than or equal to [`capacity()`]. -->
    /// <!-- - The elements at `old_len..new_len` must be initialized. -->
    ///
    /// [`capacity()`]: #method.capacity
    ///
    /// # Examples
    ///
    /// このメソッドはベクターが他のコードでバッファになっているとき、特にFFIで役に立つことがあります:
    ///
    /// <!-- This method can be useful for situations in which the vector
    /// is serving as a buffer for other code, particularly over FFI: -->
    ///
    /// ```no_run
    /// # #![allow(dead_code)]
    /// # // これは単なるドキュメントの例のための最小の枠組みです。
    /// # // 実際のライブラリのスタート地点としては利用しないでください。
    /// # pub struct StreamWrapper { strm: *mut std::ffi::c_void }
    /// # const Z_OK: i32 = 0;
    /// # extern "C" {
    /// #     fn deflateGetDictionary(
    /// #         strm: *mut std::ffi::c_void,
    /// #         dictionary: *mut u8,
    /// #         dictLength: *mut usize,
    /// #     ) -> i32;
    /// # }
    /// # impl StreamWrapper {
    /// pub fn get_dictionary(&self) -> Option<Vec<u8>> {
    ///     // FFIメソッドのドキュメント毎に、「32768バイトは常に十分」と書く。
    ///     let mut dict = Vec::with_capacity(32_768);
    ///     let mut dict_length = 0;
    ///     // SAFETY: `deflateGetDictionary`が`Z_OK`を返すとき、次が成り立つ:
    ///     // 1. `dict_length`個の要素が初期化された。
    ///     // 2. `dict_length` <= 容量 (32_768)
    ///     // このことから`set_len`は安全に呼び出すことができる。
    ///     unsafe {
    ///         // FFIを呼び出す。
    ///         let r = deflateGetDictionary(self.strm, dict.as_mut_ptr(), &mut dict_length);
    ///         if r == Z_OK {
    ///             // ...そして長さを初期化された長さに設定する。
    ///             dict.set_len(dict_length);
    ///             Some(dict)
    ///         } else {
    ///             None
    ///         }
    ///     }
    /// }
    /// # }
    /// ```
    ///
    /// <!-- ```no_run -->
    /// <!-- # #![allow(dead_code)]
    /// # // This is just a minimal skeleton for the doc example;
    /// # // don't use this as a starting point for a real library.
    /// # pub struct StreamWrapper { strm: *mut std::ffi::c_void }
    /// # const Z_OK: i32 = 0;
    /// # extern "C" {
    /// #     fn deflateGetDictionary(
    /// #         strm: *mut std::ffi::c_void,
    /// #         dictionary: *mut u8,
    /// #         dictLength: *mut usize,
    /// #     ) -> i32;
    /// # }
    /// # impl StreamWrapper {
    /// pub fn get_dictionary(&self) -> Option<Vec<u8>> {
    ///     // Per the FFI method's docs, "32768 bytes is always enough".
    ///     let mut dict = Vec::with_capacity(32_768);
    ///     let mut dict_length = 0;
    ///     // SAFETY: When `deflateGetDictionary` returns `Z_OK`, it holds that:
    ///     // 1. `dict_length` elements were initialized.
    ///     // 2. `dict_length` <= the capacity (32_768)
    ///     // which makes `set_len` safe to call.
    ///     unsafe {
    ///         // Make the FFI call...
    ///         let r = deflateGetDictionary(self.strm, dict.as_mut_ptr(), &mut dict_length);
    ///         if r == Z_OK {
    ///             // ...and update the length to what was initialized.
    ///             dict.set_len(dict_length);
    ///             Some(dict)
    ///         } else {
    ///             None
    ///         }
    ///     }
    /// }
    /// # } -->
    /// <!-- ``` -->
    ///
    /// 次の例は健全ですが、内側のベクターが`set_len`の呼び出しの前に解放されないのでメモリリークします:
    ///
    /// <!-- While the following example is sound, there is a memory leak since
    /// the inner vectors were not freed prior to the `set_len` call: -->
    ///
    /// ```
    /// let mut vec = vec![vec![1, 0, 0],
    ///                    vec![0, 1, 0],
    ///                    vec![0, 0, 1]];
    /// // SAFETY:
    /// // 1. `old_len..0`は空なので初期化する必要のある要素はない。
    /// // 2. `0 <= capacity`はどのような`capacity`に対しても成り立つ。
    /// unsafe {
    ///     vec.set_len(0);
    /// }
    /// ```
    ///
    /// <!-- ``` -->
    /// <!-- let mut vec = vec![vec![1, 0, 0],
    ///                    vec![0, 1, 0],
    ///                    vec![0, 0, 1]];
    /// // SAFETY:
    /// // 1. `old_len..0` is empty so no elements need to be initialized.
    /// // 2. `0 <= capacity` always holds whatever `capacity` is.
    /// unsafe {
    ///     vec.set_len(0);
    /// } -->
    /// <!-- ``` -->
    ///
    /// 通常、このようなときは要素を正しくドロップし、メモリをリークさせないために[`clear`]を代わりに使います。
    ///
    /// <!-- Normally, here, one would use [`clear`] instead to correctly drop
    /// the contents and thus not leak memory. -->
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity());

        self.len = new_len;
    }

    /// ベクターから要素を取り除き、その要素を返します。
    ///
    /// <!-- Removes an element from the vector and returns it. -->
    ///
    /// 取り除かれた要素はベクターの最後の要素に置き換えられます。
    ///
    /// <!-- The removed element is replaced by the last element of the vector. -->
    ///
    /// このメソッドは順序を保ちませんが、O(1)です。
    ///
    /// <!-- This does not preserve ordering, but is O(1). -->
    ///
    /// # Panics
    ///
    /// `index`が境界の外にあるときパニックします。
    ///
    /// <!-- Panics if `index` is out of bounds. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vec!["foo", "bar", "baz", "qux"];
    ///
    /// assert_eq!(v.swap_remove(1), "bar");
    /// assert_eq!(v, ["foo", "qux", "baz"]);
    ///
    /// assert_eq!(v.swap_remove(0), "foo");
    /// assert_eq!(v, ["baz", "qux"]);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn swap_remove(&mut self, index: usize) -> T {
        unsafe {
            // We replace self[index] with the last element. Note that if the
            // bounds check on hole succeeds there must be a last element (which
            // can be self[index] itself).
            let hole: *mut T = &mut self[index];
            let last = ptr::read(self.get_unchecked(self.len - 1));
            self.len -= 1;
            ptr::replace(hole, last)
        }
    }

    /// 後続する全ての要素を右側に移動して、要素をベクターの`index`の位置に挿入します。
    ///
    /// <!-- Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right. -->
    ///
    /// # Panics
    ///
    /// `index > len`のときパニックします。
    ///
    /// <!-- Panics if `index > len`. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3];
    /// vec.insert(1, 4);
    /// assert_eq!(vec, [1, 4, 2, 3]);
    /// vec.insert(4, 5);
    /// assert_eq!(vec, [1, 4, 2, 3, 5]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, index: usize, element: T) {
        let len = self.len();
        assert!(index <= len);

        // space for the new element
        if len == self.buf.cap() {
            self.reserve(1);
        }

        unsafe {
            // infallible
            // The spot to put the new value
            {
                let p = self.as_mut_ptr().add(index);
                // Shift everything over to make space. (Duplicating the
                // `index`th element into two consecutive places.)
                ptr::copy(p, p.offset(1), len - index);
                // Write it in, overwriting the first copy of the `index`th
                // element.
                ptr::write(p, element);
            }
            self.set_len(len + 1);
        }
    }

    /// ベクターの`index`の位置にある要素を取り除き、返します。後続するすべての要素は左に移動します。
    ///
    /// <!-- Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left. -->
    ///
    /// # Panics
    ///
    /// `index`が教会の外にあるときパニックします。
    ///
    /// <!-- Panics if `index` is out of bounds. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vec![1, 2, 3];
    /// assert_eq!(v.remove(1), 2);
    /// assert_eq!(v, [1, 3]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(index < len);
        unsafe {
            // infallible
            let ret;
            {
                // the place we are taking from.
                let ptr = self.as_mut_ptr().add(index);
                // copy it out, unsafely having a copy of the value on
                // the stack and in the vector at the same time.
                ret = ptr::read(ptr);

                // Shift everything down to fill in that spot.
                ptr::copy(ptr.offset(1), ptr, len - index - 1);
            }
            self.set_len(len - 1);
            ret
        }
    }

    /// 命題で指定された要素だけを残します。
    ///
    /// <!-- Retains only the elements specified by the predicate. -->
    ///
    /// 言い換えると、`f(&e)`が`false`を返すような全ての`e`を取り除きます。このメソッドはインプレースで動作し、残った要素の順序を保ちます。
    ///
    /// <!-- In other words, remove all elements `e` such that `f(&e)` returns `false`.
    /// This method operates in place and preserves the order of the retained
    /// elements. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3, 4];
    /// vec.retain(|&x| x%2 == 0);
    /// assert_eq!(vec, [2, 4]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn retain<F>(&mut self, mut f: F)
        where F: FnMut(&T) -> bool
    {
        self.drain_filter(|x| !f(x));
    }

    /// ベクター内の同じキーが解決される連続した要素から先頭以外全てを取り除きます。
    ///
    /// <!-- Removes all but the first of consecutive elements in the vector that resolve to the same
    /// key. -->
    ///
    /// ベクターがソートされているとき、このメソッドは全ての重複を取り除きます。
    ///
    /// <!-- If the vector is sorted, this removes all duplicates. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![10, 20, 21, 30, 20];
    ///
    /// vec.dedup_by_key(|i| *i / 10);
    ///
    /// assert_eq!(vec, [10, 20, 30, 20]);
    /// ```
    #[stable(feature = "dedup_by", since = "1.16.0")]
    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, mut key: F) where F: FnMut(&mut T) -> K, K: PartialEq {
        self.dedup_by(|a, b| key(a) == key(b))
    }

    /// ベクター内の与えられた等価関係を満たす連続する要素から先頭以外全てを取り除きます。
    ///
    /// <!-- Removes all but the first of consecutive elements in the vector satisfying a given equality
    /// relation. -->
    ///
    /// `same_bucket`関数はベクターから要素へ二つの参照を渡され、それらの要素同士が等しいかを決定しなければなりません。
    /// 要素はスライス内での順と逆の順で渡されるので、`same_bucket(a, b)`が`true`を返すとき、`a`が取り除かれます。
    ///
    /// <!-- The `same_bucket` function is passed references to two elements from the vector and
    /// must determine if the elements compare equal. The elements are passed in opposite order
    /// from their order in the slice, so if `same_bucket(a, b)` returns `true`, `a` is removed. -->
    ///
    /// ベクターがソートされているとき、このメソッドは全ての重複を取り除きます。
    ///
    /// <!-- If the vector is sorted, this removes all duplicates. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec!["foo", "bar", "Bar", "baz", "bar"];
    ///
    /// vec.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    ///
    /// assert_eq!(vec, ["foo", "bar", "baz", "bar"]);
    /// ```
    #[stable(feature = "dedup_by", since = "1.16.0")]
    pub fn dedup_by<F>(&mut self, same_bucket: F) where F: FnMut(&mut T, &mut T) -> bool {
        let len = {
            let (dedup, _) = self.as_mut_slice().partition_dedup_by(same_bucket);
            dedup.len()
        };
        self.truncate(len);
    }

    /// 要素をコレクションの後方に加えます。
    ///
    /// <!-- Appends an element to the back of a collection. -->
    ///
    /// # Panics
    ///
    /// ベクター内の要素の数が`usize`に収まらない場合パニックします。
    ///
    /// <!-- Panics if the number of elements in the vector overflows a `usize`. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2];
    /// vec.push(3);
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push(&mut self, value: T) {
        // This will panic or abort if we would allocate > isize::MAX bytes
        // or if the length increment would overflow for zero-sized types.
        if self.len == self.buf.cap() {
            self.reserve(1);
        }
        unsafe {
            let end = self.as_mut_ptr().add(self.len);
            ptr::write(end, value);
            self.len += 1;
        }
    }

    /// Removes the last element from a vector and returns it, or [`None`] if it
    /// is empty.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3];
    /// assert_eq!(vec.pop(), Some(3));
    /// assert_eq!(vec, [1, 2]);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                self.len -= 1;
                Some(ptr::read(self.get_unchecked(self.len())))
            }
        }
    }

    /// `other`の要素を全て`Self`に移動し、`other`を空にします。
    ///
    /// <!-- Moves all the elements of `other` into `Self`, leaving `other` empty. -->
    ///
    /// # Panics
    ///
    /// ベクターの要素の数が`usize`に収まらない場合パニックします。
    ///
    /// <!-- Panics if the number of elements in the vector overflows a `usize`. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3];
    /// let mut vec2 = vec![4, 5, 6];
    /// vec.append(&mut vec2);
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6]);
    /// assert_eq!(vec2, []);
    /// ```
    #[inline]
    #[stable(feature = "append", since = "1.4.0")]
    pub fn append(&mut self, other: &mut Self) {
        unsafe {
            self.append_elements(other.as_slice() as _);
            other.set_len(0);
        }
    }

    /// 他のバッファから`Self`に要素を追加します。
    ///
    /// <!-- Appends elements to `Self` from other buffer. -->
    #[inline]
    unsafe fn append_elements(&mut self, other: *const [T]) {
        let count = (*other).len();
        self.reserve(count);
        let len = self.len();
        ptr::copy_nonoverlapping(other as *const T, self.get_unchecked_mut(len), count);
        self.len += count;
    }

    /// ベクター内の指定された区間を取り除き、取り除かれた要素を与える排出イテレータを作成します。
    ///
    /// <!-- Creates a draining iterator that removes the specified range in the vector
    /// and yields the removed items. -->
    ///
    /// 注1: イテレータが部分的にだけ消費される、またはまったく消費されない場合も区間内の要素は取り除かれます。
    ///
    /// <!-- Note 1: The element range is removed even if the iterator is only
    /// partially consumed or not consumed at all. -->
    ///
    /// 注2: `Drain`の値がリークしたとき、ベクターから要素がいくつ取り除かれるかは未規定です。
    ///
    /// <!-- Note 2: It is unspecified how many elements are removed from the vector
    /// if the `Drain` value is leaked. -->
    ///
    /// # Panics
    ///
    /// 始点が終点より大きい、または終了位置がベクターの長さより大きいときパニックします。
    ///
    /// <!-- Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vec![1, 2, 3];
    /// let u: Vec<_> = v.drain(1..).collect();
    /// assert_eq!(v, &[1]);
    /// assert_eq!(u, &[2, 3]);
    ///
    /// // 全区間でベクターをクリアします
    /// v.drain(..);
    /// assert_eq!(v, &[]);
    /// ```
    ///
    /// <!-- ``` -->
    /// <!-- let mut v = vec![1, 2, 3];
    /// let u: Vec<_> = v.drain(1..).collect();
    /// assert_eq!(v, &[1]);
    /// assert_eq!(u, &[2, 3]);
    ///
    /// // A full range clears the vector
    /// v.drain(..);
    /// assert_eq!(v, &[]); -->
    /// <!-- ``` -->
    #[stable(feature = "drain", since = "1.6.0")]
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T>
        where R: RangeBounds<usize>
    {
        // Memory safety
        //
        // When the Drain is first created, it shortens the length of
        // the source vector to make sure no uninitialized or moved-from elements
        // are accessible at all if the Drain's destructor never gets to run.
        //
        // Drain will ptr::read out the values to remove.
        // When finished, remaining tail of the vec is copied back to cover
        // the hole, and the vector length is restored to the new length.
        //
        let len = self.len();
        let start = match range.start_bound() {
            Included(&n) => n,
            Excluded(&n) => n + 1,
            Unbounded    => 0,
        };
        let end = match range.end_bound() {
            Included(&n) => n + 1,
            Excluded(&n) => n,
            Unbounded    => len,
        };
        assert!(start <= end);
        assert!(end <= len);

        unsafe {
            // set self.vec length's to start, to be safe in case Drain is leaked
            self.set_len(start);
            // Use the borrow in the IterMut to indicate borrowing behavior of the
            // whole Drain iterator (like &mut T).
            let range_slice = slice::from_raw_parts_mut(self.as_mut_ptr().add(start),
                                                        end - start);
            Drain {
                tail_start: end,
                tail_len: len - end,
                iter: range_slice.iter(),
                vec: NonNull::from(self),
            }
        }
    }

    /// 全ての値を取り除き、ベクターを空にします。
    ///
    /// <!-- Clears the vector, removing all values. -->
    ///
    /// このメソッドは確保された容量に影響を持たないことに注意してください。
    ///
    /// <!-- Note that this method has no effect on the allocated capacity
    /// of the vector. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vec![1, 2, 3];
    ///
    /// v.clear();
    ///
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) {
        self.truncate(0)
    }

    /// ベクター内の要素の数を返します。ベクターの長さとも呼ばれるものです。
    ///
    /// <!-- Returns the number of elements in the vector, also referred to
    /// as its 'length'. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let a = vec![1, 2, 3];
    /// assert_eq!(a.len(), 3);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize {
        self.len
    }

    /// ベクターが要素を持たないとき`true`を返します。
    ///
    /// <!-- Returns `true` if the vector contains no elements. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = Vec::new();
    /// assert!(v.is_empty());
    ///
    /// v.push(1);
    /// assert!(!v.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// コレクションを与えられたインデックスで二つに分割します。
    ///
    /// <!-- Splits the collection into two at the given index. -->
    ///
    /// 新たにアロケートされた`Self`を返します。`self`は`[0, at)`の要素を含み、返された`Self`は`[at, len)`の要素を含みます。
    ///
    /// <!-- Returns a newly allocated `Self`. `self` contains elements `[0, at)`,
    /// and the returned `Self` contains elements `[at, len)`. -->
    ///
    /// `self`の容量は変わらないことに注意してください。
    ///
    /// <!-- Note that the capacity of `self` does not change. -->
    ///
    /// # Panics
    ///
    /// `at > len`のときパニックします。
    ///
    /// <!-- Panics if `at > len`. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1,2,3];
    /// let vec2 = vec.split_off(1);
    /// assert_eq!(vec, [1]);
    /// assert_eq!(vec2, [2, 3]);
    /// ```
    #[inline]
    #[stable(feature = "split_off", since = "1.4.0")]
    pub fn split_off(&mut self, at: usize) -> Self {
        assert!(at <= self.len(), "`at` out of bounds");

        let other_len = self.len - at;
        let mut other = Vec::with_capacity(other_len);

        // Unsafely `set_len` and copy items to `other`.
        unsafe {
            self.set_len(at);
            other.set_len(other_len);

            ptr::copy_nonoverlapping(self.as_ptr().add(at),
                                     other.as_mut_ptr(),
                                     other.len());
        }
        other
    }

    /// `Vec`を`len`と`new_len`が等しくなるようにインプレースでリサイズします。
    ///
    /// <!-- Resizes the `Vec` in-place so that `len` is equal to `new_len`. -->
    ///
    /// `new_len`が`len`よりも大きいとき`Vec`は差の分だけ拡張され、追加された場所はクロージャ`f`を呼び出した結果で埋められます。`f`の戻り値は生成された順に`Vec`に入ります。
    ///
    /// <!-- If `new_len` is greater than `len`, the `Vec` is extended by the
    /// difference, with each additional slot filled with the result of
    /// calling the closure `f`. The return values from `f` will end up
    /// in the `Vec` in the order they have been generated. -->
    ///
    /// `new_len`が`len`より小さいとき`Vec`は単に切り詰められます。
    ///
    /// <!-- If `new_len` is less than `len`, the `Vec` is simply truncated. -->
    ///
    /// このメソッドはプッシュ毎にクロージャを使用して新しい値を作成します。与えられた値を[`Clone`]するほうが望ましい場合は[`resize`]を使用してください。値を生成するのに[`Default`]を使いたい場合は[`Default::default()`]を第二引数として渡すことができます。
    ///
    /// <!-- This method uses a closure to create new values on every push. If
    /// you'd rather [`Clone`] a given value, use [`resize`]. If you want
    /// to use the [`Default`] trait to generate values, you can pass
    /// [`Default::default()`] as the second argument.. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3];
    /// vec.resize_with(5, Default::default);
    /// assert_eq!(vec, [1, 2, 3, 0, 0]);
    ///
    /// let mut vec = vec![];
    /// let mut p = 1;
    /// vec.resize_with(4, || { p *= 2; p });
    /// assert_eq!(vec, [2, 4, 8, 16]);
    /// ```
    ///
    /// [`resize`]: #method.resize
    /// [`Clone`]: ../../std/clone/trait.Clone.html
    #[stable(feature = "vec_resize_with", since = "1.33.0")]
    pub fn resize_with<F>(&mut self, new_len: usize, f: F)
        where F: FnMut() -> T
    {
        let len = self.len();
        if new_len > len {
            self.extend_with(new_len - len, ExtendFunc(f));
        } else {
            self.truncate(new_len);
        }
    }
}

impl<T: Clone> Vec<T> {
    /// `Vec`を`len`と`new_len`が等しくなるようにインプレースでリサイズします。
    ///
    /// <!-- Resizes the `Vec` in-place so that `len` is equal to `new_len`. -->
    ///
    /// `new_len`が`len`よりも大きいとき`Vec`は差の分だけ拡張され、追加分の位置は`value`で埋められます。`new_len`が`len`より小さいとき`Vec`は単に切り詰められます。
    ///
    /// <!-- If `new_len` is greater than `len`, the `Vec` is extended by the
    /// difference, with each additional slot filled with `value`.
    /// If `new_len` is less than `len`, the `Vec` is simply truncated. -->
    ///
    /// このメソッドは与えられた値を複製するために[`Clone`]を要求します。もっと柔軟であることを求めるなら (または[`Clone`]の代わりに[`Default`]に依存したいなら)、[`resize_with`]を使用してください。
    ///
    /// <!-- This method requires [`Clone`] to be able clone the passed value. If
    /// you need more flexibility (or want to rely on [`Default`] instead of
    /// [`Clone`]), use [`resize_with`]. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec!["hello"];
    /// vec.resize(3, "world");
    /// assert_eq!(vec, ["hello", "world", "world"]);
    ///
    /// let mut vec = vec![1, 2, 3, 4];
    /// vec.resize(2, 0);
    /// assert_eq!(vec, [1, 2]);
    /// ```
    ///
    /// [`Clone`]: ../../std/clone/trait.Clone.html
    /// [`Default`]: ../../std/default/trait.Default.html
    /// [`resize_with`]: #method.resize_with
    #[stable(feature = "vec_resize", since = "1.5.0")]
    pub fn resize(&mut self, new_len: usize, value: T) {
        let len = self.len();

        if new_len > len {
            self.extend_with(new_len - len, ExtendElement(value))
        } else {
            self.truncate(new_len);
        }
    }

    /// スライスのすべての要素を複製し`Vec`に追加します。
    ///
    /// <!-- Clones and appends all elements in a slice to the `Vec`. -->
    ///
    /// スライス`other`の各要素を複製し、そしてそれを`Vec`に追加します。`other`は順番に反復されます。
    ///
    /// <!-- Iterates over the slice `other`, clones each element, and then appends
    /// it to this `Vec`. The `other` vector is traversed in-order. -->
    ///
    /// この関数はスライスと共に動作することに特殊化していることを除いて[`extend`]と同じであることに注意してください。
    /// もしRustが特殊化 (訳注: [specialization](https://github.com/rust-lang/rust/issues/31844)) を得た場合、この関数は恐らく非推奨になります (しかしそれでも利用は可能です)。
    ///
    /// <!-- Note that this function is same as [`extend`] except that it is
    /// specialized to work with slices instead. If and when Rust gets
    /// specialization this function will likely be deprecated (but still
    /// available). -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1];
    /// vec.extend_from_slice(&[2, 3, 4]);
    /// assert_eq!(vec, [1, 2, 3, 4]);
    /// ```
    ///
    /// [`extend`]: #method.extend
    #[stable(feature = "vec_extend_from_slice", since = "1.6.0")]
    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.spec_extend(other.iter())
    }
}

impl<T: Default> Vec<T> {
    /// `Vec`を`len`と`new_len`が等しくなるようにインプレースでリサイズします。
    ///
    /// <!-- Resizes the `Vec` in-place so that `len` is equal to `new_len`. -->
    ///
    /// `new_len`が`len`よりも大きいとき`Vec`は差の分だけ拡張され、追加分の位置は[`Default::default()`]で埋められます。`new_len`が`len`より小さいとき`Vec`は単に切り詰められます。
    ///
    /// <!-- If `new_len` is greater than `len`, the `Vec` is extended by the
    /// difference, with each additional slot filled with [`Default::default()`].
    /// If `new_len` is less than `len`, the `Vec` is simply truncated. -->
    ///
    /// このメソッドはプッシュ毎に[`Default`]を使用して新しい値を作成します。[`Clone`]のほうが好ましい場合は[`resize`]を使用してください。
    ///
    /// <!-- This method uses [`Default`] to create new values on every push. If
    /// you'd rather [`Clone`] a given value, use [`resize`]. -->
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(deprecated)]
    /// #![feature(vec_resize_default)]
    ///
    /// let mut vec = vec![1, 2, 3];
    /// vec.resize_default(5);
    /// assert_eq!(vec, [1, 2, 3, 0, 0]);
    ///
    /// let mut vec = vec![1, 2, 3, 4];
    /// vec.resize_default(2);
    /// assert_eq!(vec, [1, 2]);
    /// ```
    ///
    /// [`resize`]: #method.resize
    /// [`Default::default()`]: ../../std/default/trait.Default.html#tymethod.default
    /// [`Default`]: ../../std/default/trait.Default.html
    /// [`Clone`]: ../../std/clone/trait.Clone.html
    #[unstable(feature = "vec_resize_default", issue = "41758")]
    #[rustc_deprecated(reason = "これは取り除かれていっており、代わりに`.resize_with(Default::default)`が使われています。同意できない場合、追跡イシューにコメントしてください。", since = "1.33.0")]
    // #[rustc_deprecated(reason = "This is moving towards being removed in favor \
    //     of `.resize_with(Default::default)`.  If you disagree, please comment \
    //     in the tracking issue.", since = "1.33.0")]
    pub fn resize_default(&mut self, new_len: usize) {
        let len = self.len();

        if new_len > len {
            self.extend_with(new_len - len, ExtendDefault);
        } else {
            self.truncate(new_len);
        }
    }
}

// This code generalises `extend_with_{element,default}`.
trait ExtendWith<T> {
    fn next(&mut self) -> T;
    fn last(self) -> T;
}

struct ExtendElement<T>(T);
impl<T: Clone> ExtendWith<T> for ExtendElement<T> {
    fn next(&mut self) -> T { self.0.clone() }
    fn last(self) -> T { self.0 }
}

struct ExtendDefault;
impl<T: Default> ExtendWith<T> for ExtendDefault {
    fn next(&mut self) -> T { Default::default() }
    fn last(self) -> T { Default::default() }
}

struct ExtendFunc<F>(F);
impl<T, F: FnMut() -> T> ExtendWith<T> for ExtendFunc<F> {
    fn next(&mut self) -> T { (self.0)() }
    fn last(mut self) -> T { (self.0)() }
}

impl<T> Vec<T> {
    /// 与えられたジェネレータを使って、ベクターを`n`個の値で拡張します。
    ///
    /// <!-- Extend the vector by `n` values, using the given generator. -->
    fn extend_with<E: ExtendWith<T>>(&mut self, n: usize, mut value: E) {
        self.reserve(n);

        unsafe {
            let mut ptr = self.as_mut_ptr().add(self.len());
            // Use SetLenOnDrop to work around bug where compiler
            // may not realize the store through `ptr` through self.set_len()
            // don't alias.
            let mut local_len = SetLenOnDrop::new(&mut self.len);

            // Write all elements except the last one
            for _ in 1..n {
                ptr::write(ptr, value.next());
                ptr = ptr.offset(1);
                // Increment the length in every step in case next() panics
                local_len.increment_len(1);
            }

            if n > 0 {
                // We can write the last element directly without cloning needlessly
                ptr::write(ptr, value.last());
                local_len.increment_len(1);
            }

            // len set by scope guard
        }
    }
}

// Set the length of the vec when the `SetLenOnDrop` value goes out of scope.
//
// The idea is: The length field in SetLenOnDrop is a local variable
// that the optimizer will see does not alias with any stores through the Vec's data
// pointer. This is a workaround for alias analysis issue #32155
struct SetLenOnDrop<'a> {
    len: &'a mut usize,
    local_len: usize,
}

impl<'a> SetLenOnDrop<'a> {
    #[inline]
    fn new(len: &'a mut usize) -> Self {
        SetLenOnDrop { local_len: *len, len: len }
    }

    #[inline]
    fn increment_len(&mut self, increment: usize) {
        self.local_len += increment;
    }

    #[inline]
    fn decrement_len(&mut self, decrement: usize) {
        self.local_len -= decrement;
    }
}

impl Drop for SetLenOnDrop<'_> {
    #[inline]
    fn drop(&mut self) {
        *self.len = self.local_len;
    }
}

impl<T: PartialEq> Vec<T> {
    /// [`PartialEq`]トレイトの実装によって連続して繰り返される要素を取り除きます。
    ///
    /// <!-- Removes consecutive repeated elements in the vector according to the
    /// [`PartialEq`] trait implementation. -->
    ///
    /// ベクターがソートされているときこのメソッドは全ての重複を取り除きます。
    ///
    /// <!-- If the vector is sorted, this removes all duplicates. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 2, 3, 2];
    ///
    /// vec.dedup();
    ///
    /// assert_eq!(vec, [1, 2, 3, 2]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn dedup(&mut self) {
        self.dedup_by(|a, b| a == b)
    }

    /// ベクターから最初の`item`のインスタンスをもし存在するなら取り除きます。
    ///
    /// <!-- Removes the first instance of `item` from the vector if the item exists. -->
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(vec_remove_item)]
    /// let mut vec = vec![1, 2, 3, 1];
    ///
    /// vec.remove_item(&1);
    ///
    /// assert_eq!(vec, vec![2, 3, 1]);
    /// ```
    #[unstable(feature = "vec_remove_item", reason = "recently added", issue = "40062")]
    pub fn remove_item(&mut self, item: &T) -> Option<T> {
        let pos = self.iter().position(|x| *x == *item)?;
        Some(self.remove(pos))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Internal methods and functions
////////////////////////////////////////////////////////////////////////////////

#[doc(hidden)]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn from_elem<T: Clone>(elem: T, n: usize) -> Vec<T> {
    <T as SpecFromElem>::from_elem(elem, n)
}

// Specialization trait used for Vec::from_elem
trait SpecFromElem: Sized {
    fn from_elem(elem: Self, n: usize) -> Vec<Self>;
}

impl<T: Clone> SpecFromElem for T {
    default fn from_elem(elem: Self, n: usize) -> Vec<Self> {
        let mut v = Vec::with_capacity(n);
        v.extend_with(n, ExtendElement(elem));
        v
    }
}

impl SpecFromElem for u8 {
    #[inline]
    fn from_elem(elem: u8, n: usize) -> Vec<u8> {
        if elem == 0 {
            return Vec {
                buf: RawVec::with_capacity_zeroed(n),
                len: n,
            }
        }
        unsafe {
            let mut v = Vec::with_capacity(n);
            ptr::write_bytes(v.as_mut_ptr(), elem, n);
            v.set_len(n);
            v
        }
    }
}

impl<T: Clone + IsZero> SpecFromElem for T {
    #[inline]
    fn from_elem(elem: T, n: usize) -> Vec<T> {
        if elem.is_zero() {
            return Vec {
                buf: RawVec::with_capacity_zeroed(n),
                len: n,
            }
        }
        let mut v = Vec::with_capacity(n);
        v.extend_with(n, ExtendElement(elem));
        v
    }
}

unsafe trait IsZero {
    /// この値がゼロかどうか
    ///
    /// <!-- Whether this value is zero -->
    fn is_zero(&self) -> bool;
}

macro_rules! impl_is_zero {
    ($t: ty, $is_zero: expr) => {
        unsafe impl IsZero for $t {
            #[inline]
            fn is_zero(&self) -> bool {
                $is_zero(*self)
            }
        }
    }
}

impl_is_zero!(i8, |x| x == 0);
impl_is_zero!(i16, |x| x == 0);
impl_is_zero!(i32, |x| x == 0);
impl_is_zero!(i64, |x| x == 0);
impl_is_zero!(i128, |x| x == 0);
impl_is_zero!(isize, |x| x == 0);

impl_is_zero!(u16, |x| x == 0);
impl_is_zero!(u32, |x| x == 0);
impl_is_zero!(u64, |x| x == 0);
impl_is_zero!(u128, |x| x == 0);
impl_is_zero!(usize, |x| x == 0);

impl_is_zero!(bool, |x| x == false);
impl_is_zero!(char, |x| x == '\0');

impl_is_zero!(f32, |x: f32| x.to_bits() == 0);
impl_is_zero!(f64, |x: f64| x.to_bits() == 0);

unsafe impl<T: ?Sized> IsZero for *const T {
    #[inline]
    fn is_zero(&self) -> bool {
        (*self).is_null()
    }
}

unsafe impl<T: ?Sized> IsZero for *mut T {
    #[inline]
    fn is_zero(&self) -> bool {
        (*self).is_null()
    }
}


////////////////////////////////////////////////////////////////////////////////
// Common trait implementations for Vec
////////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone> Clone for Vec<T> {
    #[cfg(not(test))]
    fn clone(&self) -> Vec<T> {
        <[T]>::to_vec(&**self)
    }

    // HACK(japaric): with cfg(test) the inherent `[T]::to_vec` method, which is
    // required for this method definition, is not available. Instead use the
    // `slice::to_vec`  function which is only available with cfg(test)
    // NB see the slice::hack module in slice.rs for more information
    #[cfg(test)]
    fn clone(&self) -> Vec<T> {
        crate::slice::to_vec(&**self)
    }

    fn clone_from(&mut self, other: &Vec<T>) {
        other.as_slice().clone_into(self);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Hash> Hash for Vec<T> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(
    message="vector indices are of type `usize` or ranges of `usize`",
    label="vector indices are of type `usize` or ranges of `usize`",
)]
impl<T, I: SliceIndex<[T]>> Index<I> for Vec<T> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&**self, index)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(
    message="vector indices are of type `usize` or ranges of `usize`",
    label="vector indices are of type `usize` or ranges of `usize`",
)]
impl<T, I: SliceIndex<[T]>> IndexMut<I> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut **self, index)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::Deref for Vec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe {
            let p = self.buf.ptr();
            assume(!p.is_null());
            slice::from_raw_parts(p, self.len)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ops::DerefMut for Vec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            let ptr = self.buf.ptr();
            assume(!ptr.is_null());
            slice::from_raw_parts_mut(ptr, self.len)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> FromIterator<T> for Vec<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Vec<T> {
        <Self as SpecExtend<T, I::IntoIter>>::from_iter(iter.into_iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> IntoIterator for Vec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// `Vec<T>`を消費するイテレータを作成します。すなわちベクターの各値を (最初から最後まで) ムーブします。このメソッドを呼び出した後、ベクターは使用できません。
    ///
    /// <!-- Creates a consuming iterator, that is, one that moves each value out of
    /// the vector (from start to end). The vector cannot be used after calling
    /// this. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let v = vec!["a".to_string(), "b".to_string()];
    /// for s in v.into_iter() {
    ///     // sはString型であって、&Stringではない
    ///     println!("{}", s);
    /// }
    /// ```
    ///
    /// <!-- ``` -->
    /// <!-- let v = vec!["a".to_string(), "b".to_string()];
    /// for s in v.into_iter() {
    ///     // s has type String, not &String
    ///     println!("{}", s);
    /// } -->
    /// <!-- ``` -->
    #[inline]
    fn into_iter(mut self) -> IntoIter<T> {
        unsafe {
            let begin = self.as_mut_ptr();
            assume(!begin.is_null());
            let end = if mem::size_of::<T>() == 0 {
                arith_offset(begin as *const i8, self.len() as isize) as *const T
            } else {
                begin.add(self.len()) as *const T
            };
            let cap = self.buf.cap();
            mem::forget(self);
            IntoIter {
                buf: NonNull::new_unchecked(begin),
                phantom: PhantomData,
                cap,
                ptr: begin,
                end,
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a Vec<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a mut Vec<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Extend<T> for Vec<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        <Self as SpecExtend<T, I::IntoIter>>::spec_extend(self, iter.into_iter())
    }
}

// Specialization trait used for Vec::from_iter and Vec::extend
trait SpecExtend<T, I> {
    fn from_iter(iter: I) -> Self;
    fn spec_extend(&mut self, iter: I);
}

impl<T, I> SpecExtend<T, I> for Vec<T>
    where I: Iterator<Item=T>,
{
    default fn from_iter(mut iterator: I) -> Self {
        // Unroll the first iteration, as the vector is going to be
        // expanded on this iteration in every case when the iterable is not
        // empty, but the loop in extend_desugared() is not going to see the
        // vector being full in the few subsequent loop iterations.
        // So we get better branch prediction.
        let mut vector = match iterator.next() {
            None => return Vec::new(),
            Some(element) => {
                let (lower, _) = iterator.size_hint();
                let mut vector = Vec::with_capacity(lower.saturating_add(1));
                unsafe {
                    ptr::write(vector.get_unchecked_mut(0), element);
                    vector.set_len(1);
                }
                vector
            }
        };
        <Vec<T> as SpecExtend<T, I>>::spec_extend(&mut vector, iterator);
        vector
    }

    default fn spec_extend(&mut self, iter: I) {
        self.extend_desugared(iter)
    }
}

impl<T, I> SpecExtend<T, I> for Vec<T>
    where I: TrustedLen<Item=T>,
{
    default fn from_iter(iterator: I) -> Self {
        let mut vector = Vec::new();
        vector.spec_extend(iterator);
        vector
    }

    default fn spec_extend(&mut self, iterator: I) {
        // This is the case for a TrustedLen iterator.
        let (low, high) = iterator.size_hint();
        if let Some(high_value) = high {
            debug_assert_eq!(low, high_value,
                             "TrustedLen iterator's size hint is not exact: {:?}",
                             (low, high));
        }
        if let Some(additional) = high {
            self.reserve(additional);
            unsafe {
                let mut ptr = self.as_mut_ptr().add(self.len());
                let mut local_len = SetLenOnDrop::new(&mut self.len);
                iterator.for_each(move |element| {
                    ptr::write(ptr, element);
                    ptr = ptr.offset(1);
                    // NB can't overflow since we would have had to alloc the address space
                    local_len.increment_len(1);
                });
            }
        } else {
            self.extend_desugared(iterator)
        }
    }
}

impl<T> SpecExtend<T, IntoIter<T>> for Vec<T> {
    fn from_iter(iterator: IntoIter<T>) -> Self {
        // A common case is passing a vector into a function which immediately
        // re-collects into a vector. We can short circuit this if the IntoIter
        // has not been advanced at all.
        if iterator.buf.as_ptr() as *const _ == iterator.ptr {
            unsafe {
                let vec = Vec::from_raw_parts(iterator.buf.as_ptr(),
                                              iterator.len(),
                                              iterator.cap);
                mem::forget(iterator);
                vec
            }
        } else {
            let mut vector = Vec::new();
            vector.spec_extend(iterator);
            vector
        }
    }

    fn spec_extend(&mut self, mut iterator: IntoIter<T>) {
        unsafe {
            self.append_elements(iterator.as_slice() as _);
        }
        iterator.ptr = iterator.end;
    }
}

impl<'a, T: 'a, I> SpecExtend<&'a T, I> for Vec<T>
    where I: Iterator<Item=&'a T>,
          T: Clone,
{
    default fn from_iter(iterator: I) -> Self {
        SpecExtend::from_iter(iterator.cloned())
    }

    default fn spec_extend(&mut self, iterator: I) {
        self.spec_extend(iterator.cloned())
    }
}

impl<'a, T: 'a> SpecExtend<&'a T, slice::Iter<'a, T>> for Vec<T>
    where T: Copy,
{
    fn spec_extend(&mut self, iterator: slice::Iter<'a, T>) {
        let slice = iterator.as_slice();
        self.reserve(slice.len());
        unsafe {
            let len = self.len();
            self.set_len(len + slice.len());
            self.get_unchecked_mut(len..).copy_from_slice(slice);
        }
    }
}

impl<T> Vec<T> {
    fn extend_desugared<I: Iterator<Item = T>>(&mut self, mut iterator: I) {
        // This is the case for a general iterator.
        //
        // This function should be the moral equivalent of:
        //
        //      for item in iterator {
        //          self.push(item);
        //      }
        while let Some(element) = iterator.next() {
            let len = self.len();
            if len == self.capacity() {
                let (lower, _) = iterator.size_hint();
                self.reserve(lower.saturating_add(1));
            }
            unsafe {
                ptr::write(self.get_unchecked_mut(len), element);
                // NB can't overflow since we would have had to alloc the address space
                self.set_len(len + 1);
            }
        }
    }

    /// ベクター内の指定された区間を与えられた`replace_with`イテレータで置き換え、取り除かれた要素を与える置換イテレータを作成します。`replace_with`は`range`と同じ長さでなくてもかまいません。
    ///
    /// <!-- Creates a splicing iterator that replaces the specified range in the vector
    /// with the given `replace_with` iterator and yields the removed items.
    /// `replace_with` does not need to be the same length as `range`. -->
    ///
    /// 注1: イテレータが最後まで消費されないとしても区間内の要素は取り除かれます。
    ///
    /// <!-- Note 1: The element range is removed even if the iterator is not
    /// consumed until the end. -->
    ///
    /// `Splice`の値がリークした場合、いくつの要素がベクターから取り除かれるかは未規定です。
    ///
    /// <!-- Note 2: It is unspecified how many elements are removed from the vector,
    /// if the `Splice` value is leaked. -->
    ///
    /// 注3: `Splice`がドロップされたとき、入力イテレータ`replace_with`は消費だけされます。
    ///
    /// <!-- Note 3: The input iterator `replace_with` is only consumed
    /// when the `Splice` value is dropped. -->
    ///
    /// 注4: このメソッドは次の場合最適です:
    ///
    /// * 後部 (ベクター内の`range`の後の要素) が空のとき
    /// * または`replace_with`が`range`の長さ以下の個数の要素を与えるとき
    /// * または`size_hint()`の下限が正確であるとき。
    ///
    /// <!-- Note 4: This is optimal if:
    ///
    /// * The tail (elements in the vector after `range`) is empty,
    /// * or `replace_with` yields fewer elements than `range`’s length
    /// * or the lower bound of its `size_hint()` is exact. -->
    ///
    /// さもなくば一時ベクターがアロケートされ、後部は二度移動されます。
    ///
    /// <!-- Otherwise, a temporary vector is allocated and the tail is moved twice. -->
    ///
    /// # Panics
    ///
    /// 始点が終点より大きい場合、または終点がベクターの長さより大きい場合パニックします。
    ///
    /// <!-- Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vec![1, 2, 3];
    /// let new = [7, 8];
    /// let u: Vec<_> = v.splice(..2, new.iter().cloned()).collect();
    /// assert_eq!(v, &[7, 8, 3]);
    /// assert_eq!(u, &[1, 2]);
    /// ```
    #[inline]
    #[stable(feature = "vec_splice", since = "1.21.0")]
    pub fn splice<R, I>(&mut self, range: R, replace_with: I) -> Splice<'_, I::IntoIter>
        where R: RangeBounds<usize>, I: IntoIterator<Item=T>
    {
        Splice {
            drain: self.drain(range),
            replace_with: replace_with.into_iter(),
        }
    }

    /// 要素を取り除くべきかの判定にクロージャを利用するイテレータを作成します。
    ///
    /// <!-- Creates an iterator which uses a closure to determine if an element should be removed. -->
    ///
    /// クロージャがtrueを返すとき、要素は取り除かれ、イテレータによって与えられます。
    /// クロージャがfalseを返すとき、要素はベクターに残り、イテレータによって与えられません。
    ///
    /// <!-- If the closure returns true, then the element is removed and yielded.
    /// If the closure returns false, the element will remain in the vector and will not be yielded
    /// by the iterator. -->
    ///
    /// このメソッドを使うことは次のコードと同値です:
    ///
    /// <!-- Using this method is equivalent to the following code: -->
    ///
    /// ```
    /// # let some_predicate = |x: &mut i32| { *x == 2 || *x == 3 || *x == 6 };
    /// # let mut vec = vec![1, 2, 3, 4, 5, 6];
    /// let mut i = 0;
    /// while i != vec.len() {
    ///     if some_predicate(&mut vec[i]) {
    ///         let val = vec.remove(i);
    ///         // ここにあなたのコード
    ///     } else {
    ///         i += 1;
    ///     }
    /// }
    ///
    /// # assert_eq!(vec, vec![1, 4, 5]);
    /// ```
    ///
    /// <!-- ``` -->
    /// <!-- # let some_predicate = |x: &mut i32| { *x == 2 || *x == 3 || *x == 6 };
    /// # let mut vec = vec![1, 2, 3, 4, 5, 6];
    /// let mut i = 0;
    /// while i != vec.len() {
    ///     if some_predicate(&mut vec[i]) {
    ///         let val = vec.remove(i);
    ///         // your code here
    ///     } else {
    ///         i += 1;
    ///     }
    /// }
    ///
    /// # assert_eq!(vec, vec![1, 4, 5]); -->
    /// <!-- ``` -->
    ///
    /// しかし`drain_filter`はより簡単に使えます。`drain_filter`は配列の要素をまとめて移動できるので、効率的でもあります。
    ///
    /// <!-- But `drain_filter` is easier to use. `drain_filter` is also more efficient,
    /// because it can backshift the elements of the array in bulk. -->
    ///
    /// `drain_filter`では保持か除外かの選択に関わらず、filterクロージャの中で各要素を変化させることもできるので注意してください。
    ///
    /// <!-- Note that `drain_filter` also lets you mutate every element in the filter closure,
    /// regardless of whether you choose to keep or remove it. -->
    ///
    ///
    /// # Examples
    ///
    /// 配列を偶数と奇数に分割し、元の確保されたメモリを再利用します:
    ///
    /// <!-- Splitting an array into evens and odds, reusing the original allocation: -->
    ///
    /// ```
    /// #![feature(drain_filter)]
    /// let mut numbers = vec![1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 15];
    ///
    /// let evens = numbers.drain_filter(|x| *x % 2 == 0).collect::<Vec<_>>();
    /// let odds = numbers;
    ///
    /// assert_eq!(evens, vec![2, 4, 6, 8, 14]);
    /// assert_eq!(odds, vec![1, 3, 5, 9, 11, 13, 15]);
    /// ```
    #[unstable(feature = "drain_filter", reason = "recently added", issue = "43244")]
    pub fn drain_filter<F>(&mut self, filter: F) -> DrainFilter<'_, T, F>
        where F: FnMut(&mut T) -> bool,
    {
        let old_len = self.len();

        // Guard against us getting leaked (leak amplification)
        unsafe { self.set_len(0); }

        DrainFilter {
            vec: self,
            idx: 0,
            del: 0,
            old_len,
            pred: filter,
        }
    }
}

/// Vecに要素をプッシュする前に参照からコピーするExtendの実装です。
///
/// <!-- Extend implementation that copies elements out of references before pushing them onto the Vec. -->
///
/// この実装はスライスイテレータに特化していて、一度に要素全体を追加するために[`copy_from_slice`]を使います。
///
/// <!-- This implementation is specialized for slice iterators, where it uses [`copy_from_slice`] to
/// append the entire slice at once. -->
///
/// [`copy_from_slice`]: ../../std/primitive.slice.html#method.copy_from_slice
#[stable(feature = "extend_ref", since = "1.2.0")]
impl<'a, T: 'a + Copy> Extend<&'a T> for Vec<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.spec_extend(iter.into_iter())
    }
}

macro_rules! __impl_slice_eq1 {
    ($Lhs: ty, $Rhs: ty) => {
        __impl_slice_eq1! { $Lhs, $Rhs, Sized }
    };
    ($Lhs: ty, $Rhs: ty, $Bound: ident) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, 'b, A: $Bound, B> PartialEq<$Rhs> for $Lhs where A: PartialEq<B> {
            #[inline]
            fn eq(&self, other: &$Rhs) -> bool { self[..] == other[..] }
            #[inline]
            fn ne(&self, other: &$Rhs) -> bool { self[..] != other[..] }
        }
    }
}

__impl_slice_eq1! { Vec<A>, Vec<B> }
__impl_slice_eq1! { Vec<A>, &'b [B] }
__impl_slice_eq1! { Vec<A>, &'b mut [B] }
__impl_slice_eq1! { Cow<'a, [A]>, &'b [B], Clone }
__impl_slice_eq1! { Cow<'a, [A]>, &'b mut [B], Clone }
__impl_slice_eq1! { Cow<'a, [A]>, Vec<B>, Clone }

macro_rules! array_impls {
    ($($N: expr)+) => {
        $(
            // NOTE: some less important impls are omitted to reduce code bloat
            __impl_slice_eq1! { Vec<A>, [B; $N] }
            __impl_slice_eq1! { Vec<A>, &'b [B; $N] }
            // __impl_slice_eq1! { Vec<A>, &'b mut [B; $N] }
            // __impl_slice_eq1! { Cow<'a, [A]>, [B; $N], Clone }
            // __impl_slice_eq1! { Cow<'a, [A]>, &'b [B; $N], Clone }
            // __impl_slice_eq1! { Cow<'a, [A]>, &'b mut [B; $N], Clone }
        )+
    }
}

array_impls! {
     0  1  2  3  4  5  6  7  8  9
    10 11 12 13 14 15 16 17 18 19
    20 21 22 23 24 25 26 27 28 29
    30 31 32
}

/// 辞書式にベクターの比較を実装します。
///
/// <!-- Implements comparison of vectors, lexicographically. -->
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialOrd> PartialOrd for Vec<T> {
    #[inline]
    fn partial_cmp(&self, other: &Vec<T>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq> Eq for Vec<T> {}

/// 辞書式順序でベクターの順序を実装します。
///
/// <!-- Implements ordering of vectors, lexicographically. -->
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Ord for Vec<T> {
    #[inline]
    fn cmp(&self, other: &Vec<T>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<#[may_dangle] T> Drop for Vec<T> {
    fn drop(&mut self) {
        unsafe {
            // use drop for [T]
            ptr::drop_in_place(&mut self[..]);
        }
        // RawVec handles deallocation
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for Vec<T> {
    /// 空の`Vec<T>`を作成します。
    ///
    /// <!-- Creates an empty `Vec<T>`. -->
    fn default() -> Vec<T> {
        Vec::new()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Debug> fmt::Debug for Vec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> AsRef<Vec<T>> for Vec<T> {
    fn as_ref(&self) -> &Vec<T> {
        self
    }
}

#[stable(feature = "vec_as_mut", since = "1.5.0")]
impl<T> AsMut<Vec<T>> for Vec<T> {
    fn as_mut(&mut self) -> &mut Vec<T> {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> AsRef<[T]> for Vec<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

#[stable(feature = "vec_as_mut", since = "1.5.0")]
impl<T> AsMut<[T]> for Vec<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: Clone> From<&'a [T]> for Vec<T> {
    #[cfg(not(test))]
    fn from(s: &'a [T]) -> Vec<T> {
        s.to_vec()
    }
    #[cfg(test)]
    fn from(s: &'a [T]) -> Vec<T> {
        crate::slice::to_vec(s)
    }
}

#[stable(feature = "vec_from_mut", since = "1.19.0")]
impl<'a, T: Clone> From<&'a mut [T]> for Vec<T> {
    #[cfg(not(test))]
    fn from(s: &'a mut [T]) -> Vec<T> {
        s.to_vec()
    }
    #[cfg(test)]
    fn from(s: &'a mut [T]) -> Vec<T> {
        crate::slice::to_vec(s)
    }
}

#[stable(feature = "vec_from_cow_slice", since = "1.14.0")]
impl<'a, T> From<Cow<'a, [T]>> for Vec<T> where [T]: ToOwned<Owned=Vec<T>> {
    fn from(s: Cow<'a, [T]>) -> Vec<T> {
        s.into_owned()
    }
}

// note: test pulls in libstd, which causes errors here
#[cfg(not(test))]
#[stable(feature = "vec_from_box", since = "1.18.0")]
impl<T> From<Box<[T]>> for Vec<T> {
    fn from(s: Box<[T]>) -> Vec<T> {
        s.into_vec()
    }
}

// note: test pulls in libstd, which causes errors here
#[cfg(not(test))]
#[stable(feature = "box_from_vec", since = "1.20.0")]
impl<T> From<Vec<T>> for Box<[T]> {
    fn from(v: Vec<T>) -> Box<[T]> {
        v.into_boxed_slice()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> From<&'a str> for Vec<u8> {
    fn from(s: &'a str) -> Vec<u8> {
        From::from(s.as_bytes())
    }
}

////////////////////////////////////////////////////////////////////////////////
// Clone-on-write
////////////////////////////////////////////////////////////////////////////////

#[stable(feature = "cow_from_vec", since = "1.8.0")]
impl<'a, T: Clone> From<&'a [T]> for Cow<'a, [T]> {
    fn from(s: &'a [T]) -> Cow<'a, [T]> {
        Cow::Borrowed(s)
    }
}

#[stable(feature = "cow_from_vec", since = "1.8.0")]
impl<'a, T: Clone> From<Vec<T>> for Cow<'a, [T]> {
    fn from(v: Vec<T>) -> Cow<'a, [T]> {
        Cow::Owned(v)
    }
}

#[stable(feature = "cow_from_vec_ref", since = "1.28.0")]
impl<'a, T: Clone> From<&'a Vec<T>> for Cow<'a, [T]> {
    fn from(v: &'a Vec<T>) -> Cow<'a, [T]> {
        Cow::Borrowed(v.as_slice())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> FromIterator<T> for Cow<'a, [T]> where T: Clone {
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Cow<'a, [T]> {
        Cow::Owned(FromIterator::from_iter(it))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Iterators
////////////////////////////////////////////////////////////////////////////////

/// ベクターから所有権を奪うイテレータ。
///
/// <!-- An iterator that moves out of a vector. -->
///
/// この構造体は[`Vec`][`Vec`]の`into_iter`メソッドによって作成されます ([`IntoIterator`]トレイトによって提供されます)。
///
/// <!-- This `struct` is created by the `into_iter` method on [`Vec`][`Vec`] (provided
/// by the [`IntoIterator`] trait). -->
///
/// [`Vec`]: struct.Vec.html
/// [`IntoIterator`]: ../../std/iter/trait.IntoIterator.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<T> {
    buf: NonNull<T>,
    phantom: PhantomData<T>,
    cap: usize,
    ptr: *const T,
    end: *const T,
}

#[stable(feature = "vec_intoiter_debug", since = "1.13.0")]
impl<T: fmt::Debug> fmt::Debug for IntoIter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter")
            .field(&self.as_slice())
            .finish()
    }
}

impl<T> IntoIter<T> {
    /// このイテレータの残りの要素をスライスとして返します。
    ///
    /// <!-- Returns the remaining items of this iterator as a slice. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let vec = vec!['a', 'b', 'c'];
    /// let mut into_iter = vec.into_iter();
    /// assert_eq!(into_iter.as_slice(), &['a', 'b', 'c']);
    /// let _ = into_iter.next().unwrap();
    /// assert_eq!(into_iter.as_slice(), &['b', 'c']);
    /// ```
    #[stable(feature = "vec_into_iter_as_slice", since = "1.15.0")]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.ptr, self.len())
        }
    }

    /// このイテレータの残りの要素をミュータブルなスライスとして返します。
    ///
    /// <!-- Returns the remaining items of this iterator as a mutable slice. -->
    ///
    /// # Examples
    ///
    /// ```
    /// let vec = vec!['a', 'b', 'c'];
    /// let mut into_iter = vec.into_iter();
    /// assert_eq!(into_iter.as_slice(), &['a', 'b', 'c']);
    /// into_iter.as_mut_slice()[2] = 'z';
    /// assert_eq!(into_iter.next().unwrap(), 'a');
    /// assert_eq!(into_iter.next().unwrap(), 'b');
    /// assert_eq!(into_iter.next().unwrap(), 'z');
    /// ```
    #[stable(feature = "vec_into_iter_as_slice", since = "1.15.0")]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(self.ptr as *mut T, self.len())
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send> Send for IntoIter<T> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Sync for IntoIter<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        unsafe {
            if self.ptr as *const _ == self.end {
                None
            } else {
                if mem::size_of::<T>() == 0 {
                    // purposefully don't use 'ptr.offset' because for
                    // vectors with 0-size elements this would return the
                    // same pointer.
                    self.ptr = arith_offset(self.ptr as *const i8, 1) as *mut T;

                    // Make up a value of this ZST.
                    Some(mem::zeroed())
                } else {
                    let old = self.ptr;
                    self.ptr = self.ptr.offset(1);

                    Some(ptr::read(old))
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = if mem::size_of::<T>() == 0 {
            (self.end as usize).wrapping_sub(self.ptr as usize)
        } else {
            unsafe { self.end.offset_from(self.ptr) as usize }
        };
        (exact, Some(exact))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        unsafe {
            if self.end == self.ptr {
                None
            } else {
                if mem::size_of::<T>() == 0 {
                    // See above for why 'ptr.offset' isn't used
                    self.end = arith_offset(self.end as *const i8, -1) as *mut T;

                    // Make up a value of this ZST.
                    Some(mem::zeroed())
                } else {
                    self.end = self.end.offset(-1);

                    Some(ptr::read(self.end))
                }
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for IntoIter<T> {
    fn is_empty(&self) -> bool {
        self.ptr == self.end
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for IntoIter<T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for IntoIter<T> {}

#[stable(feature = "vec_into_iter_clone", since = "1.8.0")]
impl<T: Clone> Clone for IntoIter<T> {
    fn clone(&self) -> IntoIter<T> {
        self.as_slice().to_owned().into_iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<#[may_dangle] T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        // destroy the remaining elements
        for _x in self.by_ref() {}

        // RawVec handles deallocation
        let _ = unsafe { RawVec::from_raw_parts(self.buf.as_ptr(), self.cap) };
    }
}

/// `Vec<T>`の排出イテレータ。
///
/// <!-- A draining iterator for `Vec<T>`. -->
///
/// この構造体は[`Vec`]の[`drain`]メソッドによって作成されます。
///
/// <!-- This `struct` is created by the [`drain`] method on [`Vec`]. -->
///
/// [`drain`]: struct.Vec.html#method.drain
/// [`Vec`]: struct.Vec.html
#[stable(feature = "drain", since = "1.6.0")]
pub struct Drain<'a, T: 'a> {
    /// Index of tail to preserve
    tail_start: usize,
    /// Length of tail
    tail_len: usize,
    /// Current remaining range to remove
    iter: slice::Iter<'a, T>,
    vec: NonNull<Vec<T>>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for Drain<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Drain")
         .field(&self.iter.as_slice())
         .finish()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<T: Sync> Sync for Drain<'_, T> {}
#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<T: Send> Send for Drain<'_, T> {}

#[stable(feature = "drain", since = "1.6.0")]
impl<T> Iterator for Drain<'_, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter.next().map(|elt| unsafe { ptr::read(elt as *const _) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<T> DoubleEndedIterator for Drain<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back().map(|elt| unsafe { ptr::read(elt as *const _) })
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<T> Drop for Drain<'_, T> {
    fn drop(&mut self) {
        // exhaust self first
        self.for_each(drop);

        if self.tail_len > 0 {
            unsafe {
                let source_vec = self.vec.as_mut();
                // memmove back untouched tail, update to new length
                let start = source_vec.len();
                let tail = self.tail_start;
                if tail != start {
                    let src = source_vec.as_ptr().add(tail);
                    let dst = source_vec.as_mut_ptr().add(start);
                    ptr::copy(src, dst, self.tail_len);
                }
                source_vec.set_len(start + self.tail_len);
            }
        }
    }
}


#[stable(feature = "drain", since = "1.6.0")]
impl<T> ExactSizeIterator for Drain<'_, T> {
    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Drain<'_, T> {}

/// `Vec`の置換イテレータ。
///
/// <!-- A splicing iterator for `Vec`. -->
///
/// この構造体は[`Vec`]の[`splice()`]メソッドによって作成されます。より詳しい情報は[`splice()`]のドキュメンテーションを参照してください。
///
/// <!-- This struct is created by the [`splice()`] method on [`Vec`]. See its
/// documentation for more. -->
///
/// [`splice()`]: struct.Vec.html#method.splice
/// [`Vec`]: struct.Vec.html
#[derive(Debug)]
#[stable(feature = "vec_splice", since = "1.21.0")]
pub struct Splice<'a, I: Iterator + 'a> {
    drain: Drain<'a, I::Item>,
    replace_with: I,
}

#[stable(feature = "vec_splice", since = "1.21.0")]
impl<I: Iterator> Iterator for Splice<'_, I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.drain.size_hint()
    }
}

#[stable(feature = "vec_splice", since = "1.21.0")]
impl<I: Iterator> DoubleEndedIterator for Splice<'_, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back()
    }
}

#[stable(feature = "vec_splice", since = "1.21.0")]
impl<I: Iterator> ExactSizeIterator for Splice<'_, I> {}


#[stable(feature = "vec_splice", since = "1.21.0")]
impl<I: Iterator> Drop for Splice<'_, I> {
    fn drop(&mut self) {
        self.drain.by_ref().for_each(drop);

        unsafe {
            if self.drain.tail_len == 0 {
                self.drain.vec.as_mut().extend(self.replace_with.by_ref());
                return
            }

            // First fill the range left by drain().
            if !self.drain.fill(&mut self.replace_with) {
                return
            }

            // There may be more elements. Use the lower bound as an estimate.
            // FIXME: Is the upper bound a better guess? Or something else?
            let (lower_bound, _upper_bound) = self.replace_with.size_hint();
            if lower_bound > 0  {
                self.drain.move_tail(lower_bound);
                if !self.drain.fill(&mut self.replace_with) {
                    return
                }
            }

            // Collect any remaining elements.
            // This is a zero-length vector which does not allocate if `lower_bound` was exact.
            let mut collected = self.replace_with.by_ref().collect::<Vec<I::Item>>().into_iter();
            // Now we have an exact count.
            if collected.len() > 0 {
                self.drain.move_tail(collected.len());
                let filled = self.drain.fill(&mut collected);
                debug_assert!(filled);
                debug_assert_eq!(collected.len(), 0);
            }
        }
        // Let `Drain::drop` move the tail back if necessary and restore `vec.len`.
    }
}

/// Private helper methods for `Splice::drop`
impl<T> Drain<'_, T> {
    /// The range from `self.vec.len` to `self.tail_start` contains elements
    /// that have been moved out.
    /// Fill that range as much as possible with new elements from the `replace_with` iterator.
    /// Returns `true` if we filled the entire range. (`replace_with.next()` didn’t return `None`.)
    unsafe fn fill<I: Iterator<Item=T>>(&mut self, replace_with: &mut I) -> bool {
        let vec = self.vec.as_mut();
        let range_start = vec.len;
        let range_end = self.tail_start;
        let range_slice = slice::from_raw_parts_mut(
            vec.as_mut_ptr().add(range_start),
            range_end - range_start);

        for place in range_slice {
            if let Some(new_item) = replace_with.next() {
                ptr::write(place, new_item);
                vec.len += 1;
            } else {
                return false
            }
        }
        true
    }

    /// Makes room for inserting more elements before the tail.
    unsafe fn move_tail(&mut self, extra_capacity: usize) {
        let vec = self.vec.as_mut();
        let used_capacity = self.tail_start + self.tail_len;
        vec.buf.reserve(used_capacity, extra_capacity);

        let new_tail_start = self.tail_start + extra_capacity;
        let src = vec.as_ptr().add(self.tail_start);
        let dst = vec.as_mut_ptr().add(new_tail_start);
        ptr::copy(src, dst, self.tail_len);
        self.tail_start = new_tail_start;
    }
}

/// Vecで`drain_filter`を呼び出すと得られるイテレータ。
///
/// <!-- An iterator produced by calling `drain_filter` on Vec. -->
#[unstable(feature = "drain_filter", reason = "recently added", issue = "43244")]
#[derive(Debug)]
pub struct DrainFilter<'a, T, F>
    where F: FnMut(&mut T) -> bool,
{
    vec: &'a mut Vec<T>,
    idx: usize,
    del: usize,
    old_len: usize,
    pred: F,
}

#[unstable(feature = "drain_filter", reason = "recently added", issue = "43244")]
impl<T, F> Iterator for DrainFilter<'_, T, F>
    where F: FnMut(&mut T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        unsafe {
            while self.idx != self.old_len {
                let i = self.idx;
                self.idx += 1;
                let v = slice::from_raw_parts_mut(self.vec.as_mut_ptr(), self.old_len);
                if (self.pred)(&mut v[i]) {
                    self.del += 1;
                    return Some(ptr::read(&v[i]));
                } else if self.del > 0 {
                    let del = self.del;
                    let src: *const T = &v[i];
                    let dst: *mut T = &mut v[i - del];
                    // This is safe because self.vec has length 0
                    // thus its elements will not have Drop::drop
                    // called on them in the event of a panic.
                    ptr::copy_nonoverlapping(src, dst, 1);
                }
            }
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.old_len - self.idx))
    }
}

#[unstable(feature = "drain_filter", reason = "recently added", issue = "43244")]
impl<T, F> Drop for DrainFilter<'_, T, F>
    where F: FnMut(&mut T) -> bool,
{
    fn drop(&mut self) {
        self.for_each(drop);
        unsafe {
            self.vec.set_len(self.old_len - self.del);
        }
    }
}
