// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use CodeSuggestion;
use Substitution;
use Level;
use RenderSpan;
use std::fmt;
use syntax_pos::{MultiSpan, Span};
use snippet::Style;

#[must_use]
#[derive(Clone, Debug, PartialEq, RustcEncodable, RustcDecodable)]
pub struct Diagnostic {
    pub level: Level,
    pub message: Vec<(String, Style)>,
    pub code: Option<String>,
    pub span: MultiSpan,
    pub children: Vec<SubDiagnostic>,
    pub suggestions: Vec<CodeSuggestion>,
}

/// For example a note attached to an error.
#[derive(Clone, Debug, PartialEq, RustcEncodable, RustcDecodable)]
pub struct SubDiagnostic {
    pub level: Level,
    pub message: Vec<(String, Style)>,
    pub span: MultiSpan,
    pub render_span: Option<RenderSpan>,
}

#[derive(PartialEq, Eq)]
pub struct DiagnosticStyledString(pub Vec<StringPart>);

impl DiagnosticStyledString {
    pub fn new() -> DiagnosticStyledString {
        DiagnosticStyledString(vec![])
    }
    pub fn push_normal<S: Into<String>>(&mut self, t: S) {
        self.0.push(StringPart::Normal(t.into()));
    }
    pub fn push_highlighted<S: Into<String>>(&mut self, t: S) {
        self.0.push(StringPart::Highlighted(t.into()));
    }
    pub fn normal<S: Into<String>>(t: S) -> DiagnosticStyledString {
        DiagnosticStyledString(vec![StringPart::Normal(t.into())])
    }

    pub fn highlighted<S: Into<String>>(t: S) -> DiagnosticStyledString {
        DiagnosticStyledString(vec![StringPart::Highlighted(t.into())])
    }

    pub fn content(&self) -> String {
        self.0.iter().map(|x| x.content()).collect::<String>()
    }
}

#[derive(PartialEq, Eq)]
pub enum StringPart {
    Normal(String),
    Highlighted(String),
}

impl StringPart {
    pub fn content(&self) -> String {
        match self {
            &StringPart::Normal(ref s) | & StringPart::Highlighted(ref s) => s.to_owned()
        }
    }
}

impl Diagnostic {
    pub fn new(level: Level, message: &str) -> Self {
        Diagnostic::new_with_code(level, None, message)
    }

    pub fn new_with_code(level: Level, code: Option<String>, message: &str) -> Self {
        Diagnostic {
            level,
            message: vec![(message.to_owned(), Style::NoStyle)],
            code,
            span: MultiSpan::new(),
            children: vec![],
            suggestions: vec![],
        }
    }

    /// Cancel the diagnostic (a structured diagnostic must either be emitted or
    /// canceled or it will panic when dropped).
    /// BEWARE: if this DiagnosticBuilder is an error, then creating it will
    /// bump the error count on the Handler and canceling it won't undo that.
    /// If you want to decrement the error count you should use `Handler::cancel`.
    pub fn cancel(&mut self) {
        self.level = Level::Cancelled;
    }

    pub fn cancelled(&self) -> bool {
        self.level == Level::Cancelled
    }

    /// Add a span/label to be included in the resulting snippet.
    /// This is pushed onto the `MultiSpan` that was created when the
    /// diagnostic was first built. If you don't call this function at
    /// all, and you just supplied a `Span` to create the diagnostic,
    /// then the snippet will just include that `Span`, which is
    /// called the primary span.
    pub fn span_label<T: Into<String>>(&mut self, span: Span, label: T) -> &mut Self {
        self.span.push_span_label(span, label.into());
        self
    }

    pub fn note_expected_found(&mut self,
                               label: &fmt::Display,
                               expected: DiagnosticStyledString,
                               found: DiagnosticStyledString)
                               -> &mut Self
    {
        self.note_expected_found_extra(label, expected, found, &"", &"")
    }

    pub fn note_expected_found_extra(&mut self,
                                     label: &fmt::Display,
                                     expected: DiagnosticStyledString,
                                     found: DiagnosticStyledString,
                                     expected_extra: &fmt::Display,
                                     found_extra: &fmt::Display)
                                     -> &mut Self
    {
        let mut msg: Vec<_> = vec![(format!("expected {} `", label), Style::NoStyle)];
        msg.extend(expected.0.iter()
                   .map(|x| match *x {
                       StringPart::Normal(ref s) => (s.to_owned(), Style::NoStyle),
                       StringPart::Highlighted(ref s) => (s.to_owned(), Style::Highlight),
                   }));
        msg.push((format!("`{}\n", expected_extra), Style::NoStyle));
        msg.push((format!("   found {} `", label), Style::NoStyle));
        msg.extend(found.0.iter()
                   .map(|x| match *x {
                       StringPart::Normal(ref s) => (s.to_owned(), Style::NoStyle),
                       StringPart::Highlighted(ref s) => (s.to_owned(), Style::Highlight),
                   }));
        msg.push((format!("`{}", found_extra), Style::NoStyle));

        // For now, just attach these as notes
        self.highlighted_note(msg);
        self
    }

    pub fn note_trait_signature(&mut self, name: String, signature: String) -> &mut Self {
        self.highlighted_note(vec![
            (format!("`{}` from trait: `", name), Style::NoStyle),
            (signature, Style::Highlight),
            ("`".to_string(), Style::NoStyle)]);
        self
    }

    pub fn note(&mut self, msg: &str) -> &mut Self {
        self.sub(Level::Note, msg, MultiSpan::new(), None);
        self
    }

    pub fn highlighted_note(&mut self, msg: Vec<(String, Style)>) -> &mut Self {
        self.sub_with_highlights(Level::Note, msg, MultiSpan::new(), None);
        self
    }

    pub fn span_note<S: Into<MultiSpan>>(&mut self,
                                         sp: S,
                                         msg: &str)
                                         -> &mut Self {
        self.sub(Level::Note, msg, sp.into(), None);
        self
    }

    pub fn warn(&mut self, msg: &str) -> &mut Self {
        self.sub(Level::Warning, msg, MultiSpan::new(), None);
        self
    }

    pub fn span_warn<S: Into<MultiSpan>>(&mut self,
                                         sp: S,
                                         msg: &str)
                                         -> &mut Self {
        self.sub(Level::Warning, msg, sp.into(), None);
        self
    }

    pub fn help(&mut self , msg: &str) -> &mut Self {
        self.sub(Level::Help, msg, MultiSpan::new(), None);
        self
    }

    pub fn span_help<S: Into<MultiSpan>>(&mut self,
                                         sp: S,
                                         msg: &str)
                                         -> &mut Self {
        self.sub(Level::Help, msg, sp.into(), None);
        self
    }

    /// Prints out a message with a suggested edit of the code. If the suggestion is presented
    /// inline it will only show the text message and not the text.
    ///
    /// See `CodeSuggestion` for more information.
    pub fn span_suggestion_short(&mut self, sp: Span, msg: &str, suggestion: String) -> &mut Self {
        self.suggestions.push(CodeSuggestion {
            substitution_parts: vec![Substitution {
                span: sp,
                substitutions: vec![suggestion],
            }],
            msg: msg.to_owned(),
            show_code_when_inline: false,
        });
        self
    }

    /// Prints out a message with a suggested edit of the code.
    ///
    /// In case of short messages and a simple suggestion,
    /// rustc displays it as a label like
    ///
    /// "try adding parentheses: `(tup.0).1`"
    ///
    /// The message
    ///
    /// * should not end in any punctuation (a `:` is added automatically)
    /// * should not be a question
    /// * should not contain any parts like "the following", "as shown"
    /// * may look like "to do xyz, use" or "to do xyz, use abc"
    /// * may contain a name of a function, variable or type, but not whole expressions
    ///
    /// See `CodeSuggestion` for more information.
    pub fn span_suggestion(&mut self, sp: Span, msg: &str, suggestion: String) -> &mut Self {
        self.suggestions.push(CodeSuggestion {
            substitution_parts: vec![Substitution {
                span: sp,
                substitutions: vec![suggestion],
            }],
            msg: msg.to_owned(),
            show_code_when_inline: true,
        });
        self
    }

    /// Prints out a message with multiple suggested edits of the code.
    pub fn span_suggestions(&mut self, sp: Span, msg: &str, suggestions: Vec<String>) -> &mut Self {
        self.suggestions.push(CodeSuggestion {
            substitution_parts: vec![Substitution {
                span: sp,
                substitutions: suggestions,
            }],
            msg: msg.to_owned(),
            show_code_when_inline: true,
        });
        self
    }

    pub fn set_span<S: Into<MultiSpan>>(&mut self, sp: S) -> &mut Self {
        self.span = sp.into();
        self
    }

    pub fn code(&mut self, s: String) -> &mut Self {
        self.code = Some(s);
        self
    }

    pub fn message(&self) -> String {
        self.message.iter().map(|i| i.0.to_owned()).collect::<String>()
    }

    pub fn styled_message(&self) -> &Vec<(String, Style)> {
        &self.message
    }

    /// Used by a lint. Copies over all details *but* the "main
    /// message".
    pub fn copy_details_not_message(&mut self, from: &Diagnostic) {
        self.span = from.span.clone();
        self.code = from.code.clone();
        self.children.extend(from.children.iter().cloned())
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
    pub(crate) fn sub(&mut self,
           level: Level,
           message: &str,
           span: MultiSpan,
           render_span: Option<RenderSpan>) {
        let sub = SubDiagnostic {
            level,
            message: vec![(message.to_owned(), Style::NoStyle)],
            span,
            render_span,
        };
        self.children.push(sub);
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
    fn sub_with_highlights(&mut self,
                           level: Level,
                           message: Vec<(String, Style)>,
                           span: MultiSpan,
                           render_span: Option<RenderSpan>) {
        let sub = SubDiagnostic {
            level,
            message,
            span,
            render_span,
        };
        self.children.push(sub);
    }
}

impl SubDiagnostic {
    pub fn message(&self) -> String {
        self.message.iter().map(|i| i.0.to_owned()).collect::<String>()
    }

    pub fn styled_message(&self) -> &Vec<(String, Style)> {
        &self.message
    }
}
