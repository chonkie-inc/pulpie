//! Stage 3: Block classifier using LightGBM model.
//!
//! Loads a trained GBM model at compile time and classifies blocks
//! as content (keep) or boilerplate (discard).

use std::sync::LazyLock;

use crate::gbm::GbmModel;
use crate::segment::Block;

static MODEL: LazyLock<GbmModel> = LazyLock::new(|| {
    GbmModel::load(include_str!("../data/model_dom_v3_combined.txt"))
});

const THRESHOLD: f64 = 0.5;

/// Classify a block as content (true) or boilerplate (false).
pub fn classify(block: &Block) -> bool {
    let features = block.features.to_feature_vec();
    MODEL.predict(&features, THRESHOLD)
}

/// Alias for classify — returns true if block should be kept.
pub fn should_keep(block: &Block) -> bool {
    classify(block)
}

/// Filter blocks, keeping only those classified as content.
pub fn filter_content(blocks: Vec<Block<'_>>) -> Vec<Block<'_>> {
    blocks.into_iter().filter(|b| classify(b)).collect()
}

/// Filter blocks using only the first `n_trees` trees (faster, slightly less accurate).
pub fn filter_content_fast(blocks: Vec<Block<'_>>, n_trees: usize) -> Vec<Block<'_>> {
    blocks.into_iter().filter(|b| {
        let features = b.features.to_feature_vec();
        MODEL.predict_n(&features, THRESHOLD, n_trees)
    }).collect()
}
