//! This module contains the "cleaned" pieces of the AST, and the functions
//! that clean them.

use its = syntax::parse::token::ident_to_str;

use rustc::metadata::{csearch,decoder,cstore};
use syntax;
use syntax::ast;

use std;
use doctree;
use visit_ast;
use std::local_data;

pub trait Clean<T> {
    fn clean(&self) -> T;
}

impl<T: Clean<U>, U> Clean<~[U]> for ~[T] {
    fn clean(&self) -> ~[U] {
        self.iter().map(|x| x.clean()).collect()
    }
}
impl<T: Clean<U>, U> Clean<U> for @T {
    fn clean(&self) -> U {
        (**self).clean()
    }
}

impl<T: Clean<U>, U> Clean<Option<U>> for Option<T> {
    fn clean(&self) -> Option<U> {
        match self {
            &None => None,
            &Some(ref v) => Some(v.clean())
        }
    }
}

impl<T: Clean<U>, U> Clean<~[U]> for syntax::opt_vec::OptVec<T> {
    fn clean(&self) -> ~[U] {
        match self {
            &syntax::opt_vec::Empty => ~[],
            &syntax::opt_vec::Vec(ref v) => v.clean()
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Crate {
    name: ~str,
    module: Option<Item>,
}

impl Clean<Crate> for visit_ast::RustdocVisitor {
    fn clean(&self) -> Crate {
        use syntax::attr::{find_linkage_metas, last_meta_item_value_str_by_name};
        let maybe_meta = last_meta_item_value_str_by_name(find_linkage_metas(self.attrs), "name");

        Crate {
            name: match maybe_meta {
                Some(x) => x.to_owned(),
                None => fail!("rustdoc_ng requires a #[link(name=\"foo\")] crate attribute"),
            },
            module: Some(self.module.clean()),
        }
    }
}

/// Anything with a source location and set of attributes and, optionally, a
/// name. That is, anything that can be documented. This doesn't correspond
/// directly to the AST's concept of an item; it's a strict superset.
#[deriving(Clone, Encodable, Decodable)]
pub struct Item {
    /// Stringified span
    source: ~str,
    /// Not everything has a name. E.g., impls
    name: Option<~str>,
    attrs: ~[Attribute],
    inner: ItemEnum,
    visibility: Option<Visibility>,
    id: ast::NodeId,
}

#[deriving(Clone, Encodable, Decodable)]
pub enum ItemEnum {
    StructItem(Struct),
    EnumItem(Enum),
    FunctionItem(Function),
    ModuleItem(Module),
    TypedefItem(Typedef),
    StaticItem(Static),
    TraitItem(Trait),
    ImplItem(Impl),
    ViewItemItem(ViewItem),
    TyMethodItem(TyMethod),
    MethodItem(Method),
    StructFieldItem(StructField),
    VariantItem(Variant),
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Module {
    items: ~[Item],
}

impl Clean<Item> for doctree::Module {
    fn clean(&self) -> Item {
        let name = if self.name.is_some() {
            self.name.unwrap().clean()
        } else {
            ~""
        };
        Item {
            name: Some(name),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            visibility: self.vis.clean(),
            id: self.id,
            inner: ModuleItem(Module {
               items: std::vec::concat(&[self.structs.clean(),
                              self.enums.clean(), self.fns.clean(),
                              self.mods.clean(), self.typedefs.clean(),
                              self.statics.clean(), self.traits.clean(),
                              self.impls.clean(), self.view_items.clean()])
            })
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum Attribute {
    Word(~str),
    List(~str, ~[Attribute]),
    NameValue(~str, ~str)
}

impl Clean<Attribute> for ast::MetaItem {
    fn clean(&self) -> Attribute {
        match self.node {
            ast::MetaWord(s) => Word(s.to_owned()),
            ast::MetaList(ref s, ref l) => List(s.to_owned(), l.clean()),
            ast::MetaNameValue(s, ref v) => NameValue(s.to_owned(), lit_to_str(v))
        }
    }
}

impl Clean<Attribute> for ast::Attribute {
    fn clean(&self) -> Attribute {
        self.node.value.clean()
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct TyParam {
    name: ~str,
    id: ast::NodeId,
    bounds: ~[TyParamBound]
}

impl Clean<TyParam> for ast::TyParam {
    fn clean(&self) -> TyParam {
        TyParam {
            name: self.ident.clean(),
            id: self.id,
            bounds: self.bounds.clean(),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum TyParamBound {
    RegionBound,
    TraitBound(Type)
}

impl Clean<TyParamBound> for ast::TyParamBound {
    fn clean(&self) -> TyParamBound {
        match *self {
            ast::RegionTyParamBound => RegionBound,
            ast::TraitTyParamBound(ref t) => TraitBound(t.clean()),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Lifetime(~str);

impl Clean<Lifetime> for ast::Lifetime {
    fn clean(&self) -> Lifetime {
        Lifetime(self.ident.clean())
    }
}

// maybe use a Generic enum and use ~[Generic]?
#[deriving(Clone, Encodable, Decodable)]
pub struct Generics {
    lifetimes: ~[Lifetime],
    type_params: ~[TyParam]
}

impl Generics {
    fn new() -> Generics {
        Generics {
            lifetimes: ~[],
            type_params: ~[]
        }
    }
}

impl Clean<Generics> for ast::Generics {
    fn clean(&self) -> Generics {
        Generics {
            lifetimes: self.lifetimes.clean(),
            type_params: self.ty_params.clean(),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Method {
    generics: Generics,
    self_: SelfTy,
    purity: ast::purity,
    decl: FnDecl,
}

impl Clean<Item> for ast::method {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.ident.clean()),
            attrs: self.attrs.clean(),
            source: self.span.clean(),
            id: self.self_id.clone(),
            visibility: None,
            inner: MethodItem(Method {
                generics: self.generics.clean(),
                self_: self.explicit_self.clean(),
                purity: self.purity.clone(),
                decl: self.decl.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct TyMethod {
    purity: ast::purity,
    decl: FnDecl,
    generics: Generics,
    self_: SelfTy,
}

impl Clean<Item> for ast::TypeMethod {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.ident.clean()),
            attrs: self.attrs.clean(),
            source: self.span.clean(),
            id: self.id,
            visibility: None,
            inner: TyMethodItem(TyMethod {
                purity: self.purity.clone(),
                decl: self.decl.clean(),
                self_: self.explicit_self.clean(),
                generics: self.generics.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum SelfTy {
    SelfStatic,
    SelfValue,
    SelfBorrowed(Option<Lifetime>, Mutability),
    SelfManaged(Mutability),
    SelfOwned,
}

impl Clean<SelfTy> for ast::explicit_self {
    fn clean(&self) -> SelfTy {
        match self.node {
            ast::sty_static => SelfStatic,
            ast::sty_value => SelfValue,
            ast::sty_uniq => SelfOwned,
            ast::sty_region(lt, mt) => SelfBorrowed(lt.clean(), mt.clean()),
            ast::sty_box(mt) => SelfManaged(mt.clean()),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Function {
    decl: FnDecl,
    generics: Generics,
}

impl Clean<Item> for doctree::Function {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            visibility: self.vis.clean(),
            id: self.id,
            inner: FunctionItem(Function {
                decl: self.decl.clean(),
                generics: self.generics.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct ClosureDecl {
    sigil: ast::Sigil,
    region: Option<Lifetime>,
    lifetimes: ~[Lifetime],
    decl: FnDecl,
    onceness: ast::Onceness,
    purity: ast::purity,
    bounds: ~[TyParamBound]
}

impl Clean<ClosureDecl> for ast::TyClosure {
    fn clean(&self) -> ClosureDecl {
        ClosureDecl {
            sigil: self.sigil,
            region: self.region.clean(),
            lifetimes: self.lifetimes.clean(),
            decl: self.decl.clean(),
            onceness: self.onceness,
            purity: self.purity,
            bounds: match self.bounds {
                Some(ref x) => x.clean(),
                None        => ~[]
            },
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct FnDecl {
    inputs: ~[Argument],
    output: Type,
    cf: RetStyle,
    attrs: ~[Attribute]
}

impl Clean<FnDecl> for ast::fn_decl {
    fn clean(&self) -> FnDecl {
        FnDecl {
            inputs: self.inputs.iter().map(|x| x.clean()).collect(),
            output: (self.output.clean()),
            cf: self.cf.clean(),
            attrs: ~[]
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Argument {
    type_: Type,
    name: ~str,
    id: ast::NodeId
}

impl Clean<Argument> for ast::arg {
    fn clean(&self) -> Argument {
        Argument {
            name: name_from_pat(self.pat),
            type_: (self.ty.clean()),
            id: self.id
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum RetStyle {
    NoReturn,
    Return
}

impl Clean<RetStyle> for ast::ret_style {
    fn clean(&self) -> RetStyle {
        match *self {
            ast::return_val => Return,
            ast::noreturn => NoReturn
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Trait {
    methods: ~[TraitMethod],
    generics: Generics,
    parents: ~[Type],
}

impl Clean<Item> for doctree::Trait {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            id: self.id,
            visibility: self.vis.clean(),
            inner: TraitItem(Trait {
                methods: self.methods.clean(),
                generics: self.generics.clean(),
                parents: self.parents.clean(),
            }),
        }
    }
}

impl Clean<Type> for ast::trait_ref {
    fn clean(&self) -> Type {
        let t = Unresolved(self.path.clean(), None, self.ref_id);
        resolve_type(&t)
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum TraitMethod {
    Required(Item),
    Provided(Item),
}

impl TraitMethod {
    fn is_req(&self) -> bool {
        match self {
            &Required(*) => true,
            _ => false,
        }
    }
    fn is_def(&self) -> bool {
        match self {
            &Provided(*) => true,
            _ => false,
        }
    }
}

impl Clean<TraitMethod> for ast::trait_method {
    fn clean(&self) -> TraitMethod {
        match self {
            &ast::required(ref t) => Required(t.clean()),
            &ast::provided(ref t) => Provided(t.clean()),
        }
    }
}

/// A representation of a Type suitable for hyperlinking purposes. Ideally one can get the original
/// type out of the AST/ty::ctxt given one of these, if more information is needed. Most importantly
/// it does not preserve mutability or boxes.
#[deriving(Clone, Encodable, Decodable)]
pub enum Type {
    /// Most types start out as "Unresolved". It serves as an intermediate stage between cleaning
    /// and type resolution.
    Unresolved(Path, Option<~[TyParamBound]>, ast::NodeId),
    /// structs/enums/traits (anything that'd be an ast::ty_path)
    ResolvedPath { path: Path, typarams: Option<~[TyParamBound]>, id: ast::NodeId },
    /// Reference to an item in an external crate (fully qualified path)
    External(~str, ~str),
    // I have no idea how to usefully use this.
    TyParamBinder(ast::NodeId),
    /// For parameterized types, so the consumer of the JSON don't go looking
    /// for types which don't exist anywhere.
    Generic(ast::NodeId),
    /// For references to self
    Self(ast::NodeId),
    /// Primitives are just the fixed-size numeric types (plus int/uint/float), and char.
    Primitive(ast::prim_ty),
    Closure(~ClosureDecl),
    /// extern "ABI" fn
    BareFunction(~BareFunctionDecl),
    Tuple(~[Type]),
    Vector(~Type),
    FixedVector(~Type, ~str),
    String,
    Bool,
    /// aka ty_nil
    Unit,
    /// aka ty_bot
    Bottom,
    Unique(~Type),
    Managed(Mutability, ~Type),
    RawPointer(Mutability, ~Type),
    BorrowedRef { lifetime: Option<Lifetime>, mutability: Mutability, type_: ~Type},
    // region, raw, other boxes, mutable
}

impl Clean<Type> for ast::Ty {
    fn clean(&self) -> Type {
        use syntax::ast::*;
        debug!("cleaning type `%?`", self);
        let codemap = local_data::get(super::ctxtkey, |x| *x.unwrap()).sess.codemap;
        debug!("span corresponds to `%s`", codemap.span_to_str(self.span));
        let t = match self.node {
            ty_nil => Unit,
            ty_ptr(ref m) =>  RawPointer(m.mutbl.clean(), ~resolve_type(&m.ty.clean())),
            ty_rptr(ref l, ref m) => 
                BorrowedRef {lifetime: l.clean(), mutability: m.mutbl.clean(),
                             type_: ~resolve_type(&m.ty.clean())},
            ty_box(ref m) => Managed(m.mutbl.clean(), ~resolve_type(&m.ty.clean())),
            ty_uniq(ref m) => Unique(~resolve_type(&m.ty.clean())),
            ty_vec(ref m) => Vector(~resolve_type(&m.ty.clean())),
            ty_fixed_length_vec(ref m, ref e) => FixedVector(~resolve_type(&m.ty.clean()),
                                                             e.span.to_src()),
            ty_tup(ref tys) => Tuple(tys.iter().map(|x| resolve_type(&x.clean())).collect()),
            ty_path(ref p, ref tpbs, id) => Unresolved(p.clean(), tpbs.clean(), id),
            ty_closure(ref c) => Closure(~c.clean()),
            ty_bare_fn(ref barefn) => BareFunction(~barefn.clean()),
            ty_bot => Bottom,
            ref x => fail!("Unimplemented type %?", x),
        };
        resolve_type(&t)
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct StructField {
    type_: Type,
}

impl Clean<Item> for ast::struct_field {
    fn clean(&self) -> Item {
        let (name, vis) = match self.node.kind {
            ast::named_field(id, vis) => (Some(id), Some(vis)),
            _ => (None, None)
        };
        Item {
            name: name.clean(),
            attrs: self.node.attrs.clean(),
            source: self.span.clean(),
            visibility: vis,
            id: self.node.id,
            inner: StructFieldItem(StructField {
                type_: self.node.ty.clean(),
            }),
        }
    }
}

pub type Visibility = ast::visibility;

impl Clean<Option<Visibility>> for ast::visibility {
    fn clean(&self) -> Option<Visibility> {
        Some(*self)
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Struct {
    struct_type: doctree::StructType,
    generics: Generics,
    fields: ~[Item],
}

impl Clean<Item> for doctree::Struct {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            id: self.id,
            visibility: self.vis.clean(),
            inner: StructItem(Struct {
                struct_type: self.struct_type,
                generics: self.generics.clean(),
                fields: self.fields.clean(),
            }),
        }
    }
}

/// This is a more limited form of the standard Struct, different in that it
/// it lacks the things most items have (name, id, parameterization). Found
/// only as a variant in an enum.
#[deriving(Clone, Encodable, Decodable)]
pub struct VariantStruct {
    struct_type: doctree::StructType,
    fields: ~[Item],
}

impl Clean<VariantStruct> for syntax::ast::struct_def {
    fn clean(&self) -> VariantStruct {
        VariantStruct {
            struct_type: doctree::struct_type_from_def(self),
            fields: self.fields.clean(),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Enum {
    variants: ~[Item],
    generics: Generics,
}

impl Clean<Item> for doctree::Enum {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            id: self.id,
            visibility: self.vis.clean(),
            inner: EnumItem(Enum {
                variants: self.variants.clean(),
                generics: self.generics.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Variant {
    kind: VariantKind,
}

impl Clean<Item> for doctree::Variant {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            visibility: self.vis.clean(),
            id: self.id,
            inner: VariantItem(Variant {
                kind: self.kind.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum VariantKind {
    CLikeVariant,
    TupleVariant(~[Type]),
    StructVariant(VariantStruct),
}

impl Clean<VariantKind> for ast::variant_kind {
    fn clean(&self) -> VariantKind {
        match self {
            &ast::tuple_variant_kind(ref args) => {
                if args.len() == 0 {
                    CLikeVariant
                } else {
                    TupleVariant(args.iter().map(|x| x.ty.clean()).collect())
                }
            },
            &ast::struct_variant_kind(ref sd) => StructVariant(sd.clean()),
        }
    }
}

impl Clean<~str> for syntax::codemap::span {
    fn clean(&self) -> ~str {
        let cm = local_data::get(super::ctxtkey, |x| x.unwrap().clone()).sess.codemap;
        cm.span_to_str(*self)
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Path {
    name: ~str,
    lifetime: Option<Lifetime>,
    typarams: ~[Type]
}

impl Clean<Path> for ast::Path {
    fn clean(&self) -> Path {
        Path {
            name: path_to_str(self),
            lifetime: self.rp.clean(),
            typarams: self.types.clean(),
        }
    }
}

fn path_to_str(p: &ast::Path) -> ~str {
    use syntax::parse::token::interner_get;

    let mut s = ~"";
    let mut first = true;
    for i in p.idents.iter().map(|x| interner_get(x.name)) {
        if !first || p.global {
            s.push_str("::");
        } else {
            first = false;
        }
        s.push_str(i);
    }
    s
}

impl Clean<~str> for ast::ident {
    fn clean(&self) -> ~str {
        its(self).to_owned()
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Typedef {
    type_: Type,
    generics: Generics,
}

impl Clean<Item> for doctree::Typedef {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            id: self.id.clone(),
            visibility: self.vis.clean(),
            inner: TypedefItem(Typedef {
                type_: self.ty.clean(),
                generics: self.gen.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct BareFunctionDecl {
    purity: ast::purity,
    generics: Generics,
    decl: FnDecl,
    abi: ~str
}

impl Clean<BareFunctionDecl> for ast::TyBareFn {
    fn clean(&self) -> BareFunctionDecl {
        BareFunctionDecl {
            purity: self.purity,
            generics: Generics {
                lifetimes: self.lifetimes.clean(),
                type_params: ~[],
            },
            decl: self.decl.clean(),
            abi: self.abis.to_str(),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Static {
    type_: Type,
    mutability: Mutability,
    /// It's useful to have the value of a static documented, but I have no
    /// desire to represent expressions (that'd basically be all of the AST,
    /// which is huge!). So, have a string.
    expr: ~str,
}

impl Clean<Item> for doctree::Static {
    fn clean(&self) -> Item {
        debug!("claning static %s: %?", self.name.clean(), self);
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            id: self.id,
            visibility: self.vis.clean(),
            inner: StaticItem(Static {
                type_: self.type_.clean(),
                mutability: self.mutability.clean(),
                expr: self.expr.span.to_src(),
            }),
        }
    }
}

#[deriving(ToStr, Clone, Encodable, Decodable)]
pub enum Mutability {
    Mutable,
    Immutable,
    Const,
}

impl Clean<Mutability> for ast::mutability {
    fn clean(&self) -> Mutability {
        match self {
            &ast::m_mutbl => Mutable,
            &ast::m_imm => Immutable,
            &ast::m_const => Const
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Impl {
    generics: Generics,
    trait_: Option<Type>,
    for_: Type,
    methods: ~[Item],
}

impl Clean<Item> for doctree::Impl {
    fn clean(&self) -> Item {
        Item {
            name: None,
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            id: self.id,
            visibility: self.vis.clean(),
            inner: ImplItem(Impl {
                generics: self.generics.clean(),
                trait_: self.trait_.clean(),
                for_: self.for_.clean(),
                methods: self.methods.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct ViewItem {
    inner: ViewItemInner
}

impl Clean<Item> for ast::view_item {
    fn clean(&self) -> Item {
        Item {
            name: None,
            attrs: self.attrs.clean(),
            source: self.span.clean(),
            id: 0,
            visibility: self.vis.clean(),
            inner: ViewItemItem(ViewItem {
                inner: self.node.clean()
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum ViewItemInner {
    ExternMod(~str, Option<~str>, ~[Attribute], ast::NodeId),
    Import(~[ViewPath])
}

impl Clean<ViewItemInner> for ast::view_item_ {
    fn clean(&self) -> ViewItemInner {
        match self {
            &ast::view_item_extern_mod(ref i, ref p, ref mi, ref id) =>
                ExternMod(i.clean(), p.map(|x| x.to_owned()),  mi.clean(), *id),
            &ast::view_item_use(ref vp) => Import(vp.clean())
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum ViewPath {
    SimpleImport(~str, Path, ast::NodeId),
    GlobImport(Path, ast::NodeId),
    ImportList(Path, ~[ViewListIdent], ast::NodeId)
}

impl Clean<ViewPath> for ast::view_path {
    fn clean(&self) -> ViewPath {
        match self.node {
            ast::view_path_simple(ref i, ref p, ref id) => SimpleImport(i.clean(), p.clean(), *id),
            ast::view_path_glob(ref p, ref id) => GlobImport(p.clean(), *id),
            ast::view_path_list(ref p, ref pl, ref id) => ImportList(p.clean(), pl.clean(), *id),
        }
    }
}

pub type ViewListIdent = ~str;

impl Clean<ViewListIdent> for ast::path_list_ident {
    fn clean(&self) -> ViewListIdent {
        self.node.name.clean()
    }
}

// Utilities

trait ToSource {
    fn to_src(&self) -> ~str;
}

impl ToSource for syntax::codemap::span {
    fn to_src(&self) -> ~str {
        debug!("converting span %s to snippet", self.clean());
        let cm = local_data::get(super::ctxtkey, |x| x.unwrap().clone()).sess.codemap.clone();
        let sn = match cm.span_to_snippet(*self) {
            Some(x) => x,
            None    => ~""
        };
        debug!("got snippet %s", sn);
        sn
    }
}

fn lit_to_str(lit: &ast::lit) -> ~str {
    match lit.node {
        ast::lit_str(st) => st.to_owned(),
        ast::lit_int(ch, ast::ty_char) => ~"'" + ch.to_str() + "'",
        ast::lit_int(i, _t) => i.to_str(),
        ast::lit_uint(u, _t) => u.to_str(),
        ast::lit_int_unsuffixed(i) => i.to_str(),
        ast::lit_float(f, _t) => f.to_str(),
        ast::lit_float_unsuffixed(f) => f.to_str(),
        ast::lit_bool(b) => b.to_str(),
        ast::lit_nil => ~"",
    }
}

fn name_from_pat(p: &ast::pat) -> ~str {
    use syntax::ast::*;
    match p.node {
        pat_wild => ~"_",
        pat_ident(_, ref p, _) => path_to_str(p),
        pat_enum(ref p, _) => path_to_str(p),
        pat_struct(*) => fail!("tried to get argument name from pat_struct, \
                                 which is not allowed in function arguments"),
        pat_tup(*) => ~"(tuple arg NYI)",
        pat_box(p) => name_from_pat(p),
        pat_uniq(p) => name_from_pat(p),
        pat_region(p) => name_from_pat(p),
        pat_lit(*) => fail!("tried to get argument name from pat_lit, \
                             which is not allowed in function arguments"),
        pat_range(*) => fail!("tried to get argument name from pat_range, \
                               which is not allowed in function arguments"),
        pat_vec(*) => fail!("tried to get argument name from pat_vec, \
                             which is not allowed in function arguments")
    }
}

fn remove_comment_tags(s: &str) -> ~str {
    if s.starts_with("/") {
        match s.slice(0,3) {
            &"///" => return s.slice(3, s.len()).trim().to_owned(),
            &"/**" | &"/*!" => return s.slice(3, s.len() - 2).trim().to_owned(),
            _ => return s.trim().to_owned()
        }
    } else {
        return s.to_owned();
    }
}

/*fn collapse_docs(attrs: ~[Attribute]) -> ~[Attribute] {
    let mut docstr = ~"";
    for at in attrs.iter() {
        match *at {
            //XXX how should these be separated?
            NameValue(~"doc", ref s) => docstr.push_str(fmt!("%s ", clean_comment_body(s.clone()))),
            _ => ()
        }
    }
    let mut a = attrs.iter().filter(|&a| match a {
        &NameValue(~"doc", _) => false,
        _ => true
    }).map(|x| x.clone()).collect::<~[Attribute]>();
    a.push(NameValue(~"doc", docstr.trim().to_owned()));
    a
}*/

/// Given a Type, resolve it using the def_map
fn resolve_type(t: &Type) -> Type {
    use syntax::ast::*;

    let (path, tpbs, id) = match t {
        &Unresolved(ref path, ref tbps, id) => (path, tbps, id),
        _ => return (*t).clone(),
    };

    let dm = local_data::get(super::ctxtkey, |x| *x.unwrap()).tycx.def_map;
    debug!("searching for %? in defmap", id);
    let d = match dm.find(&id) {
        Some(k) => k,
        None => {
            let ctxt = local_data::get(super::ctxtkey, |x| *x.unwrap());
            debug!("could not find %? in defmap (`%s`)", id,
                   syntax::ast_map::node_id_to_str(ctxt.tycx.items, id, ctxt.sess.intr()));
            fail!("Unexpected failure: unresolved id not in defmap (this is a bug!)")
        }
    };

    let def_id = match *d {
        def_fn(i, _) => i,
        def_self(i, _) | def_self_ty(i) => return Self(i),
        def_ty(i) => i,
        def_trait(i) => {
            debug!("saw def_trait in def_to_id");
            i
        },
        def_prim_ty(p) => match p {
            ty_str => return String,
            ty_bool => return Bool,
            _ => return Primitive(p)
        },
        def_ty_param(i, _) => return Generic(i.node),
        def_struct(i) => i,
        def_typaram_binder(i) => { 
            debug!("found a typaram_binder, what is it? %d", i);
            return TyParamBinder(i);
        },
        x => fail!("resolved type maps to a weird def %?", x),
    };

    if def_id.crate != ast::CRATE_NODE_ID {
        let sess = local_data::get(super::ctxtkey, |x| *x.unwrap()).sess;
        let mut path = ~"";
        let mut ty = ~"";
        do csearch::each_path(sess.cstore, def_id.crate) |pathstr, deflike, _vis| {
            match deflike {
                decoder::dl_def(di) => {
                    let d2 = match di {
                        def_fn(i, _) | def_ty(i) | def_trait(i) |
                            def_struct(i) | def_mod(i) => Some(i),
                        _ => None,
                    };
                    if d2.is_some() {
                        let d2 = d2.unwrap();
                        if def_id.node == d2.node {
                            debug!("found external def: %?", di);
                            path = pathstr.to_owned();
                            ty = match di {
                                def_fn(*) => ~"fn",
                                def_ty(*) => ~"enum",
                                def_trait(*) => ~"trait",
                                def_prim_ty(p) => match p {
                                    ty_str => ~"str",
                                    ty_bool => ~"bool",
                                    ty_int(t) => match t.to_str() {
                                        ~"" => ~"i",
                                        s => s
                                    },
                                    ty_uint(t) => t.to_str(),
                                    ty_float(t) => t.to_str()
                                },
                                def_ty_param(*) => ~"generic",
                                def_struct(*) => ~"struct",
                                def_typaram_binder(*) => ~"typaram_binder",
                                x => fail!("resolved external maps to a weird def %?", x),
                            };

                        }
                    }
                },
                _ => (),
            };
            true
        };
        let cname = cstore::get_crate_data(sess.cstore, def_id.crate).name.to_owned();
        External(cname + "::" + path, ty)
    } else {
        ResolvedPath {path: path.clone(), typarams: tpbs.clone(), id: def_id.node}
    }
}

#[cfg(test)]
mod tests {
    use super::NameValue;

    #[test]
    fn test_doc_collapsing() {
        assert_eq!(collapse_docs(~"// Foo\n//Bar\n // Baz\n"), ~"Foo\nBar\nBaz");
        assert_eq!(collapse_docs(~"* Foo\n *  Bar\n *Baz\n"), ~"Foo\n Bar\nBaz");
        assert_eq!(collapse_docs(~"* Short desc\n *\n * Bar\n *Baz\n"), ~"Short desc\n\nBar\nBaz");
        assert_eq!(collapse_docs(~" * Foo"), ~"Foo");
        assert_eq!(collapse_docs(~"\n *\n *\n * Foo"), ~"Foo");
    }

    fn collapse_docs(input: ~str) -> ~str {
        let attrs = ~[NameValue(~"doc", input)];
        let attrs_clean = super::collapse_docs(attrs);

        match attrs_clean[0] {
            NameValue(~"doc", s) => s,
            _ => (fail!("dude where's my doc?"))
        }
    }
}
