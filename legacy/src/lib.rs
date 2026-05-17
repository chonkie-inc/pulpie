//! Pulpie: fast CPU-only web content extraction pipeline.
//!
//! Pipeline: sanitize → prune → segment → classify → markdown → cleanup

pub mod clean;
pub mod classify;
pub mod error;
pub mod gbm;
pub mod markdown;
pub mod postprocess;
pub mod segment;

use scraper::Html;

pub use error::{HummingbirdError, Result};

/// Extract clean markdown from raw HTML.
pub fn extract(html: &str) -> Result<String> {
    // Stage 1: DOM cleaning
    let sanitized = clean::sanitize(html);
    let mut document = Html::parse_document(&sanitized);
    clean::prune_boilerplate(&mut document);

    // Stage 2: Block segmentation + feature extraction
    let blocks = segment::segment(&document);
    if blocks.is_empty() {
        return Err(HummingbirdError::NoContent);
    }

    // Stage 3: Classification
    let content_blocks = classify::filter_content(blocks);
    if content_blocks.is_empty() {
        return Err(HummingbirdError::NoContent);
    }

    // Stage 4: HTML → Markdown (per-block, then join)
    // md_from now processes the element itself (not just children),
    // so headings get #, blockquotes get >, lists get -, etc.
    let mut parts: Vec<String> = Vec::new();
    for block in &content_blocks {
        let md = markdown::md_from(block.element);
        let trimmed = md.trim().to_string();
        if !trimmed.is_empty() {
            parts.push(trimmed);
        }
    }

    let combined = parts.join("\n\n");

    // Stage 5: MD cleanup
    let result = postprocess::postprocess(&combined);

    if result.is_empty() {
        return Err(HummingbirdError::NoContent);
    }

    Ok(result)
}

/// Extract content as raw HTML (for piping through external converters like html2text).
pub fn extract_html(html: &str) -> Result<String> {
    let sanitized = clean::sanitize(html);
    let mut document = Html::parse_document(&sanitized);
    clean::prune_boilerplate(&mut document);

    let blocks = segment::segment(&document);
    if blocks.is_empty() {
        return Err(HummingbirdError::NoContent);
    }

    let content_blocks = classify::filter_content(blocks);
    if content_blocks.is_empty() {
        return Err(HummingbirdError::NoContent);
    }

    let mut parts: Vec<String> = Vec::new();
    for block in &content_blocks {
        parts.push(block.element.html());
    }

    let result = parts.join("\n");
    if result.is_empty() {
        return Err(HummingbirdError::NoContent);
    }

    Ok(result)
}
