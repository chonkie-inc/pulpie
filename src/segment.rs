//! Stage 2: Block segmentation and feature extraction.
//!
//! Walks the cleaned DOM and produces a flat list of paragraph-level blocks,
//! each with features that the classifier uses to decide keep/discard.

use scraper::{ElementRef, Html, Selector};
use serde::Serialize;
use std::sync::LazyLock;

use crate::clean::BOILERPLATE;

static SEL_BODY: LazyLock<Selector> = LazyLock::new(|| Selector::parse("body").unwrap());
static SEL_A: LazyLock<Selector> = LazyLock::new(|| Selector::parse("a").unwrap());
static SEL_P: LazyLock<Selector> = LazyLock::new(|| Selector::parse("p").unwrap());
static SEL_HEADINGS: LazyLock<Selector> =
    LazyLock::new(|| Selector::parse("h1,h2,h3,h4,h5,h6").unwrap());
static SEL_LI: LazyLock<Selector> = LazyLock::new(|| Selector::parse("li").unwrap());
static SEL_IMG: LazyLock<Selector> = LazyLock::new(|| Selector::parse("img").unwrap());

/// Tags that are always leaf blocks (never recurse into children for segmentation).
const BLOCK_TAGS: &[&str] = &[
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "pre", "blockquote", "td", "th", "figcaption", "dt", "dd",
];

/// Tags that are containers — recurse if they have block-level descendants.
const CONTAINER_TAGS: &[&str] = &[
    "div", "section", "article", "main", "body", "form",
    "nav", "footer", "aside", "header",
];

/// Positive content signal keywords in class/id attributes.
const CONTENT_KEYWORDS: &[&str] = &[
    "content", "article", "post", "entry", "main", "body", "text", "story",
    "page", "hentry", "blog",
];

/// Common English stop words for stop_word_ratio.
const STOP_WORDS: &[&str] = &[
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "is", "are", "was", "were", "been", "being", "has", "had", "did", "does",
];

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum TagType {
    Paragraph,
    Heading,
    ListItem,
    Preformatted,
    TableCell,
    Blockquote,
    Other,
}

impl TagType {
    pub fn score(&self) -> f64 {
        match self {
            TagType::Paragraph => 3.0,
            TagType::Heading => 4.0,
            TagType::ListItem => 1.0,
            TagType::Preformatted => 3.0,
            TagType::TableCell => 0.0,
            TagType::Blockquote => 2.0,
            TagType::Other => -1.0,
        }
    }
}

/// Full feature set for block classification (~40 features).
#[derive(Debug, Serialize)]
pub struct Features {
    // --- Basic text stats ---
    pub text_len: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub comma_count: usize,
    pub avg_word_length: f64,
    pub stop_word_ratio: f64,
    pub capitalization_ratio: f64,
    pub punctuation_density: f64,
    pub has_copyright: bool,
    pub has_date_pattern: bool,

    // --- Structural ---
    pub link_len: usize,
    pub link_count: usize,
    pub link_ratio: f64,
    pub tag_count: usize,
    pub paragraph_count: usize,
    pub heading_count: usize,
    pub list_item_count: usize,
    pub image_count: usize,
    pub text_to_tag_ratio: f64,

    // --- Readability scoring ---
    pub class_id_score: f64,
    pub parent_class_id_score: f64,
    pub tag_type: TagType,
    pub tag_type_score: f64,

    // --- DOM / Position ---
    pub dom_depth: usize,
    pub position: f64,
    pub distance_from_end: f64,
    pub is_first_10pct: bool,
    pub is_last_10pct: bool,
    pub has_boilerplate_class: bool,
    pub parent_tag_type: f64,
    pub semantic_ancestor: f64,
    pub in_nav: bool,
    pub in_footer: bool,
    pub in_aside: bool,
    pub in_header: bool,

    // --- Context (filled in second pass) ---
    pub prev_block_text_len: usize,
    pub prev_block_link_ratio: f64,
    pub next_block_text_len: usize,
    pub next_block_link_ratio: f64,
    pub blocks_since_heading: usize,
    pub blocks_until_heading: usize,

    // --- Section context (filled in second pass) ---
    pub section_heading_text_len: usize,
    pub section_block_count: usize,
    pub section_link_density: f64,

    // --- Page-level ---
    pub page_total_blocks: usize,
    pub page_total_text_len: usize,
    pub page_total_link_ratio: f64,
    pub page_heading_count: usize,
    pub block_text_len_ratio: f64,
}

impl Features {
    /// Produce the feature vector matching the trained GBM model.
    /// Order must match selected_features.json exactly.
    pub fn to_feature_vec(&self) -> [f64; 30] {
        let tag_type_val = match self.tag_type {
            TagType::Paragraph => 0.0,
            TagType::Heading => 1.0,
            TagType::ListItem => 2.0,
            TagType::Preformatted => 3.0,
            TagType::TableCell => 4.0,
            TagType::Blockquote => 5.0,
            TagType::Other => 6.0,
        };
        [
            self.link_ratio,
            self.page_total_blocks as f64,
            self.parent_tag_type,
            self.page_total_link_ratio,
            tag_type_val,
            self.section_block_count as f64,
            self.page_total_text_len as f64,
            self.dom_depth as f64,
            self.section_link_density,
            self.position,
            self.tag_type_score,
            self.section_heading_text_len as f64,
            self.blocks_since_heading as f64,
            self.word_count as f64,
            self.next_block_link_ratio,
            self.prev_block_link_ratio,
            self.text_to_tag_ratio,
            self.semantic_ancestor,
            self.blocks_until_heading as f64,
            self.page_heading_count as f64,
            self.parent_class_id_score,
            self.next_block_text_len as f64,
            self.link_count as f64,
            self.text_len as f64,
            self.tag_count as f64,
            self.avg_word_length,
            self.in_nav as u8 as f64,
            self.in_footer as u8 as f64,
            self.in_aside as u8 as f64,
            self.in_header as u8 as f64,
        ]
    }
}

pub struct Block<'a> {
    pub element: ElementRef<'a>,
    pub node_id: ego_tree::NodeId,
    pub features: Features,
    pub text: String,
}

/// Segment a cleaned document into blocks with extracted features.
pub fn segment<'a>(document: &'a Html) -> Vec<Block<'a>> {
    let root = match document.select(&SEL_BODY).next() {
        Some(body) => body,
        None => document.root_element(),
    };

    let mut elements: Vec<ElementRef<'a>> = Vec::new();
    walk(root, &mut elements);

    let total = elements.len();
    if total == 0 {
        return Vec::new();
    }

    // First pass: compute per-block features
    let mut blocks: Vec<Block<'a>> = elements
        .into_iter()
        .enumerate()
        .map(|(i, element)| {
            let text: String = element.text().collect();
            let features = compute_features(&element, &text, i, total);
            let node_id = element.id();
            Block { element, node_id, features, text }
        })
        .collect();

    // Compute page-level aggregates
    let page_total_text_len: usize = blocks.iter().map(|b| b.features.text_len).sum();
    let page_total_link_len: usize = blocks.iter().map(|b| b.features.link_len).sum();
    let page_total_link_ratio = if page_total_text_len > 0 {
        page_total_link_len as f64 / page_total_text_len as f64
    } else {
        0.0
    };
    let page_heading_count: usize = blocks.iter().map(|b| b.features.heading_count).sum();

    // Second pass: fill context and page-level features
    for i in 0..blocks.len() {
        // Page-level
        blocks[i].features.page_total_blocks = total;
        blocks[i].features.page_total_text_len = page_total_text_len;
        blocks[i].features.page_total_link_ratio = page_total_link_ratio;
        blocks[i].features.page_heading_count = page_heading_count;
        blocks[i].features.block_text_len_ratio = if page_total_text_len > 0 {
            blocks[i].features.text_len as f64 / page_total_text_len as f64
        } else {
            0.0
        };

        // Previous block context
        if i > 0 {
            blocks[i].features.prev_block_text_len = blocks[i - 1].features.text_len;
            blocks[i].features.prev_block_link_ratio = blocks[i - 1].features.link_ratio;
        }

        // Next block context
        if i + 1 < blocks.len() {
            blocks[i].features.next_block_text_len = blocks[i + 1].features.text_len;
            blocks[i].features.next_block_link_ratio = blocks[i + 1].features.link_ratio;
        }
    }

    // Blocks since/until heading (forward + backward pass)
    let mut since_heading: usize = usize::MAX;
    for i in 0..blocks.len() {
        if blocks[i].features.tag_type == TagType::Heading {
            since_heading = 0;
        } else if since_heading < usize::MAX {
            since_heading += 1;
        }
        blocks[i].features.blocks_since_heading = if since_heading == usize::MAX { 999 } else { since_heading };
    }

    let mut until_heading: usize = usize::MAX;
    for i in (0..blocks.len()).rev() {
        if blocks[i].features.tag_type == TagType::Heading {
            until_heading = 0;
        } else if until_heading < usize::MAX {
            until_heading += 1;
        }
        blocks[i].features.blocks_until_heading = if until_heading == usize::MAX { 999 } else { until_heading };
    }

    // Section-level features: identify sections (heading → next heading),
    // compute aggregates, assign to each block in the section.
    {
        // Find heading indices to define section boundaries
        let heading_indices: Vec<usize> = blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.features.tag_type == TagType::Heading)
            .map(|(i, _)| i)
            .collect();

        // For each section: (start_idx, end_idx_exclusive, heading_text_len, block_count, link_density)
        let mut sections: Vec<(usize, usize, usize, usize, f64)> = Vec::new();

        // Blocks before first heading (no section heading)
        if heading_indices.is_empty() {
            // No headings at all — all blocks get defaults (already 0)
        } else {
            // Blocks before the first heading get no section features (defaults)
            // Each heading starts a section that runs until the next heading (exclusive)
            for (idx, &h) in heading_indices.iter().enumerate() {
                let end = if idx + 1 < heading_indices.len() {
                    heading_indices[idx + 1]
                } else {
                    blocks.len()
                };
                let heading_text_len = blocks[h].features.text_len;
                let section_blocks = &blocks[h..end];
                let block_count = section_blocks.len();
                let total_link_ratio: f64 = section_blocks
                    .iter()
                    .map(|b| b.features.link_ratio)
                    .sum();
                let link_density = if block_count > 0 {
                    total_link_ratio / block_count as f64
                } else {
                    0.0
                };
                sections.push((h, end, heading_text_len, block_count, link_density));
            }

            // Assign to blocks
            for (start, end, heading_text_len, block_count, link_density) in &sections {
                for i in *start..*end {
                    blocks[i].features.section_heading_text_len = *heading_text_len;
                    blocks[i].features.section_block_count = *block_count;
                    blocks[i].features.section_link_density = *link_density;
                }
            }
        }
    }

    blocks
}

fn walk<'a>(element: ElementRef<'a>, blocks: &mut Vec<ElementRef<'a>>) {
    let tag = element.value().name();

    if BLOCK_TAGS.contains(&tag) {
        let text: String = element.text().collect();
        if text.trim().len() >= 5 {
            blocks.push(element);
        }
        return;
    }

    if CONTAINER_TAGS.contains(&tag) || is_inline_or_unknown(tag) {
        if has_block_descendants(&element) {
            for child in element.children() {
                if let Some(child_el) = ElementRef::wrap(child) {
                    walk(child_el, blocks);
                }
            }
        } else {
            let text: String = element.text().collect();
            if text.trim().len() >= 5 {
                blocks.push(element);
            }
        }
        return;
    }

    for child in element.children() {
        if let Some(child_el) = ElementRef::wrap(child) {
            walk(child_el, blocks);
        }
    }
}

fn has_block_descendants(element: &ElementRef<'_>) -> bool {
    for descendant in element.descendants() {
        if let Some(el) = descendant.value().as_element() {
            if BLOCK_TAGS.contains(&el.name.local.as_ref()) {
                return true;
            }
        }
    }
    false
}

fn is_inline_or_unknown(tag: &str) -> bool {
    !BLOCK_TAGS.contains(&tag) && !CONTAINER_TAGS.contains(&tag)
}

fn compute_features(element: &ElementRef<'_>, text: &str, index: usize, total: usize) -> Features {
    let text_len = text.len();
    let text_trimmed = text.trim();

    // Word stats
    let words: Vec<&str> = text_trimmed.split_whitespace().collect();
    let word_count = words.len();
    let avg_word_length = if word_count > 0 {
        words.iter().map(|w| w.len()).sum::<usize>() as f64 / word_count as f64
    } else {
        0.0
    };

    // Sentence count (rough: split on .!?)
    let sentence_count = text_trimmed
        .chars()
        .filter(|c| matches!(c, '.' | '!' | '?'))
        .count()
        .max(if word_count > 0 { 1 } else { 0 });

    let comma_count = text_trimmed.chars().filter(|c| *c == ',').count();

    // Stop word ratio
    let stop_count = words
        .iter()
        .filter(|w| STOP_WORDS.contains(&w.to_lowercase().as_str()))
        .count();
    let stop_word_ratio = if word_count > 0 {
        stop_count as f64 / word_count as f64
    } else {
        0.0
    };

    // Capitalization ratio
    let alpha_chars: usize = text_trimmed.chars().filter(|c| c.is_alphabetic()).count();
    let upper_chars: usize = text_trimmed.chars().filter(|c| c.is_uppercase()).count();
    let capitalization_ratio = if alpha_chars > 0 {
        upper_chars as f64 / alpha_chars as f64
    } else {
        0.0
    };

    // Punctuation density
    let punct_chars: usize = text_trimmed.chars().filter(|c| c.is_ascii_punctuation()).count();
    let punctuation_density = if text_len > 0 {
        punct_chars as f64 / text_len as f64
    } else {
        0.0
    };

    // Copyright / date patterns
    let text_lower = text_trimmed.to_lowercase();
    let has_copyright = text_lower.contains('©') || text_lower.contains("copyright")
        || text_lower.contains("all rights reserved");
    let has_date_pattern = has_date_like(&text_lower);

    // Links
    let link_len: usize = element
        .select(&SEL_A)
        .map(|a| a.text().collect::<String>().len())
        .sum();
    let link_count = element.select(&SEL_A).count();
    let link_ratio = if text_len > 0 {
        link_len as f64 / text_len as f64
    } else {
        1.0
    };

    // Structural counts
    let tag_count = element
        .descendants()
        .filter(|n| n.value().is_element())
        .count()
        .max(1);
    let paragraph_count = element.select(&SEL_P).count();
    let heading_count = element.select(&SEL_HEADINGS).count();
    let list_item_count = element.select(&SEL_LI).count();
    let image_count = element.select(&SEL_IMG).count();
    let text_to_tag_ratio = text_len as f64 / tag_count as f64;

    // Class/ID scoring
    let class = element.value().attr("class").unwrap_or("").to_lowercase();
    let id = element.value().attr("id").unwrap_or("").to_lowercase();
    let class_id_score = compute_class_id_score(&class, &id);

    // Parent class/ID scoring
    let parent_class_id_score = element
        .parent()
        .and_then(ElementRef::wrap)
        .map(|p| {
            let pc = p.value().attr("class").unwrap_or("").to_lowercase();
            let pi = p.value().attr("id").unwrap_or("").to_lowercase();
            compute_class_id_score(&pc, &pi)
        })
        .unwrap_or(0.0);

    let has_boilerplate_class =
        BOILERPLATE.iter().any(|p| class.contains(p) || id.contains(p));

    // Tag type
    let tag = element.value().name();
    let tag_type = match tag {
        "p" => TagType::Paragraph,
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => TagType::Heading,
        "li" | "dt" | "dd" => TagType::ListItem,
        "pre" => TagType::Preformatted,
        "td" | "th" => TagType::TableCell,
        "blockquote" => TagType::Blockquote,
        _ => TagType::Other,
    };
    let tag_type_score = tag_type.score();

    // Position
    let dom_depth = element.ancestors().count();
    let position = if total > 1 {
        index as f64 / (total - 1) as f64
    } else {
        0.5
    };

    // Parent tag type (immediate parent, categorical)
    let parent_tag_type = element
        .parent()
        .and_then(ElementRef::wrap)
        .map(|p| match p.value().name() {
            "nav" => 1.0,
            "article" => 2.0,
            "main" => 3.0,
            "aside" => 4.0,
            "footer" => 5.0,
            "header" => 6.0,
            "ul" | "ol" => 7.0,
            "div" => 8.0,
            "section" => 9.0,
            "table" | "tbody" => 10.0,
            _ => 0.0,
        })
        .unwrap_or(0.0);

    // Nearest semantic ancestor (walk up, break on first match)
    let mut in_nav = false;
    let mut in_footer = false;
    let mut in_aside = false;
    let mut in_header = false;
    let semantic_ancestor = {
        let mut found = 0.0;
        for anc in element.ancestors() {
            if let Some(el) = anc.value().as_element() {
                match el.name.local.as_ref() {
                    "nav" => { in_nav = true; if found == 0.0 { found = 3.0; } }
                    "footer" => { in_footer = true; if found == 0.0 { found = 5.0; } }
                    "aside" => { in_aside = true; if found == 0.0 { found = 4.0; } }
                    "header" => { in_header = true; if found == 0.0 { found = 6.0; } }
                    "article" => { if found == 0.0 { found = 1.0; } }
                    "main" => { if found == 0.0 { found = 2.0; } }
                    _ => {}
                }
            }
        }
        found
    };

    Features {
        text_len,
        word_count,
        sentence_count,
        comma_count,
        avg_word_length,
        stop_word_ratio,
        capitalization_ratio,
        punctuation_density,
        has_copyright,
        has_date_pattern,

        link_len,
        link_count,
        link_ratio,
        tag_count,
        paragraph_count,
        heading_count,
        list_item_count,
        image_count,
        text_to_tag_ratio,

        class_id_score,
        parent_class_id_score,
        tag_type,
        tag_type_score,

        dom_depth,
        position,
        distance_from_end: 1.0 - position,
        is_first_10pct: position < 0.1,
        is_last_10pct: position > 0.9,
        has_boilerplate_class,
        parent_tag_type,
        semantic_ancestor,
        in_nav,
        in_footer,
        in_aside,
        in_header,

        // Section context — filled in second pass
        section_heading_text_len: 0,
        section_block_count: 0,
        section_link_density: 0.0,

        // Context — filled in second pass
        prev_block_text_len: 0,
        prev_block_link_ratio: 0.0,
        next_block_text_len: 0,
        next_block_link_ratio: 0.0,
        blocks_since_heading: 999,
        blocks_until_heading: 999,

        // Page-level — filled in second pass
        page_total_blocks: 0,
        page_total_text_len: 0,
        page_total_link_ratio: 0.0,
        page_heading_count: 0,
        block_text_len_ratio: 0.0,
    }
}

fn compute_class_id_score(class: &str, id: &str) -> f64 {
    let mut score = 0.0;
    for kw in CONTENT_KEYWORDS {
        if class.contains(kw) || id.contains(kw) {
            score += 25.0;
        }
    }
    for kw in BOILERPLATE {
        if class.contains(kw) || id.contains(kw) {
            score -= 25.0;
        }
    }
    score
}

fn has_date_like(text: &str) -> bool {
    // Quick heuristic: month names or YYYY patterns near numbers
    const MONTHS: &[&str] = &[
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ];
    if MONTHS.iter().any(|m| text.contains(m)) {
        return true;
    }
    // YYYY-MM-DD or MM/DD/YYYY patterns
    let bytes = text.as_bytes();
    for i in 0..bytes.len().saturating_sub(9) {
        if bytes[i].is_ascii_digit()
            && bytes[i + 1].is_ascii_digit()
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
            && (bytes[i + 4] == b'-' || bytes[i + 4] == b'/')
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_basic() {
        let html = r#"<html><body>
            <p>First paragraph with enough text.</p>
            <p>Second paragraph with enough text.</p>
        </body></html>"#;
        let doc = Html::parse_document(html);
        let blocks = segment(&doc);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].features.tag_type, TagType::Paragraph);
    }

    #[test]
    fn test_segment_skips_short() {
        let html = r#"<html><body><p>Hi</p><p>Long enough paragraph here.</p></body></html>"#;
        let doc = Html::parse_document(html);
        let blocks = segment(&doc);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn test_segment_leaf_div() {
        let html = r#"<html><body><div>This div has no block children but has text.</div></body></html>"#;
        let doc = Html::parse_document(html);
        let blocks = segment(&doc);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].features.tag_type, TagType::Other);
    }

    #[test]
    fn test_boilerplate_class_detected() {
        let html = r#"<html><body><p class="sidebar-widget">Some sidebar content here.</p></body></html>"#;
        let doc = Html::parse_document(html);
        let blocks = segment(&doc);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].features.has_boilerplate_class);
    }

    #[test]
    fn test_extended_features() {
        let html = r#"<html><body>
            <h2>Introduction</h2>
            <p>The quick brown fox jumps over the lazy dog. This is a test sentence with some words.</p>
            <p>Copyright © 2024 Example Corp. All rights reserved.</p>
        </body></html>"#;
        let doc = Html::parse_document(html);
        let blocks = segment(&doc);
        assert_eq!(blocks.len(), 3);

        // Heading
        assert_eq!(blocks[0].features.tag_type, TagType::Heading);
        assert!(blocks[0].features.tag_type_score > 0.0);

        // Content paragraph
        assert!(blocks[1].features.word_count > 5);
        assert!(blocks[1].features.stop_word_ratio > 0.0);
        assert!(!blocks[1].features.has_copyright);

        // Copyright paragraph
        assert!(blocks[2].features.has_copyright);

        // Context: middle block should have prev/next
        assert!(blocks[1].features.prev_block_text_len > 0);
        assert!(blocks[1].features.next_block_text_len > 0);

        // Page-level
        assert_eq!(blocks[0].features.page_total_blocks, 3);
    }
}
