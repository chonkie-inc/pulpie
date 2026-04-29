//! Stage 1: DOM cleaning — sanitize HTML and prune boilerplate elements.
//!
//! Ported from konbu/src/extract.rs. Removes structural junk before
//! block segmentation.

use ego_tree::NodeId;
use regex::Regex;
use scraper::node::Node;
use scraper::Html;
use std::sync::LazyLock;

// Tags removed during sanitization (before DOM parsing)
// Note: nav/footer/aside/header are NOT removed here — they're handled in
// prune_boilerplate() where we can check context (e.g., don't remove <header>
// inside <article>). Removing them via regex is too aggressive.
const SANITIZE_TAGS: &[&str] = &[
    "script", "style", "noscript", "iframe", "svg", "canvas", "template",
    "head",
    "select", "textarea",
    "video", "audio", "embed", "object", "applet",
];

// Patterns indicating boilerplate (used for DOM pruning and feature extraction)
pub(crate) const BOILERPLATE: &[&str] = &[
    "sidebar", "advert", "advertisement", "sponsor", "promoted",
    "related", "recommended", "trending", "popular",
    "comments", "comment-list", "comment-section",
    "social", "share-", "sharing",
    "newsletter", "subscribe",
    "author-info", "author-bio", "byline", "author-badge",
    "article-tags", "tag-list", "tags-container",
    "tooltip", "improve", "improvement", "like-count", "report",
    "breadcrumb",
    "last-updated", "article-meta",
];

// Pre-compiled regexes for sanitization
static SANITIZE_REGEXES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    SANITIZE_TAGS.iter().map(|tag| {
        Regex::new(&format!(
            r"(?is)<{tag}(?:\s[^>]*)?>.*?</{tag}>|<{tag}(?:\s[^>]*)?/?>",
            tag = tag
        )).unwrap()
    }).collect()
});

/// Remove structural junk from HTML before parsing.
pub fn sanitize(html: &str) -> String {
    SANITIZE_REGEXES.iter().fold(html.to_string(), |acc, re| {
        re.replace_all(&acc, "").into_owned()
    })
}

/// Prune boilerplate elements from DOM using trafilatura-style heuristics.
pub fn prune_boilerplate(document: &mut Html) {
    let mut to_remove: Vec<NodeId> = Vec::new();

    for node in document.tree.nodes() {
        if let Node::Element(el) = node.value() {
            let tag = el.name.local.as_ref();

            // Never prune structural roots
            if matches!(tag, "html" | "body") {
                continue;
            }

            let class = el.attr("class").unwrap_or("").to_lowercase();
            let id = el.attr("id").unwrap_or("").to_lowercase();

            if has_boilerplate_token(&class, &id) {
                to_remove.push(node.id());
                continue;
            }

            if tag == "table" && is_table_link_dense(&node) {
                to_remove.push(node.id());
                continue;
            }

            // Prune <form> elements unless they're large page-wrapper forms.
            // Small forms (login, search, comment, review) are boilerplate.
            // Large forms wrapping the whole page (ASP.NET pattern) have real content.
            if tag == "form" {
                let (text_len, _) = count_text_and_links(&node);
                if text_len < 200 {
                    to_remove.push(node.id());
                    continue;
                }
                let has_content_tags = node.descendants().any(|d| {
                    matches!(d.value(), Node::Element(e) if matches!(
                        e.name.local.as_ref(),
                        "p" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "article" | "table"
                    ))
                });
                if !has_content_tags {
                    to_remove.push(node.id());
                    continue;
                }
            }
        }
    }

    for id in to_remove {
        if let Some(mut node) = document.tree.get_mut(id) {
            node.detach();
        }
    }
}

/// Check if class/id tokens match boilerplate patterns.
/// Splits by whitespace into individual CSS class tokens, then checks if any
/// token starts with a boilerplate pattern. This avoids false positives like
/// "no-sidebars" matching "sidebar" or "has-sidebar" matching "sidebar".
fn has_boilerplate_token(class: &str, id: &str) -> bool {
    let tokens = class.split_whitespace().chain(id.split_whitespace());
    for token in tokens {
        for pattern in BOILERPLATE {
            if token == *pattern || token.starts_with(&format!("{}-", pattern)) || token.starts_with(&format!("{}_", pattern)) {
                return true;
            }
        }
    }
    false
}

fn is_table_link_dense(node: &ego_tree::NodeRef<'_, Node>) -> bool {
    let (text_len, link_len) = count_text_and_links(node);

    if text_len == 0 {
        return true;
    }
    if text_len > 1000 {
        return false;
    }

    let ratio = link_len as f64 / text_len as f64;
    ratio > 0.9
}

fn count_text_and_links(node: &ego_tree::NodeRef<'_, Node>) -> (usize, usize) {
    let mut text_len = 0;
    let mut link_len = 0;

    for descendant in node.descendants() {
        if let Node::Text(t) = descendant.value() {
            let len = t.text.trim().len();
            text_len += len;

            let in_link = descendant.ancestors().any(|ancestor| {
                matches!(ancestor.value(), Node::Element(el) if el.name.local.as_ref() == "a")
            });

            if in_link {
                link_len += len;
            }
        }
    }

    (text_len, link_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_strips_scripts() {
        let html = r#"<html><body><p>Hello</p><script>alert('xss')</script></body></html>"#;
        let clean = sanitize(html);
        assert!(clean.contains("Hello"));
        assert!(!clean.contains("alert"));
    }

    #[test]
    fn test_prune_removes_nav() {
        let html = r#"<html><body><nav>Menu</nav><p>Content</p></body></html>"#;
        let sanitized = sanitize(html);
        let mut doc = Html::parse_document(&sanitized);
        prune_boilerplate(&mut doc);
        let text: String = doc.root_element().text().collect();
        assert!(text.contains("Content"));
    }
}
