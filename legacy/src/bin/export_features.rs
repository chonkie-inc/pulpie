//! Export block features as JSON for training data generation.
//!
//! Reads HTML from stdin, segments into blocks, extracts features,
//! and outputs JSON array with block text + features.

use pulpie::{clean, segment};
use scraper::Html;
use serde::Serialize;
use std::io::Read;

#[derive(Serialize)]
struct BlockExport {
    index: usize,
    text: String,
    features: segment::Features,
}

fn main() {
    let mut html = String::new();
    std::io::stdin().read_to_string(&mut html).expect("Failed to read stdin");

    let sanitized = clean::sanitize(&html);
    let mut document = Html::parse_document(&sanitized);
    clean::prune_boilerplate(&mut document);

    let blocks = segment::segment(&document);

    let exports: Vec<BlockExport> = blocks
        .into_iter()
        .enumerate()
        .map(|(i, block)| BlockExport {
            index: i,
            text: block.text.clone(),
            features: block.features,
        })
        .collect();

    let json = serde_json::to_string(&exports).expect("Failed to serialize");
    println!("{}", json);
}
