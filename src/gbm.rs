//! Pure Rust LightGBM model inference.
//!
//! Parses LightGBM's text model format and runs tree traversal for prediction.
//! Binary classification only (sigmoid output).

/// A single decision tree.
struct Tree {
    num_leaves: usize,
    split_feature: Vec<usize>,
    threshold: Vec<f64>,
    left_child: Vec<i32>,
    right_child: Vec<i32>,
    leaf_value: Vec<f64>,
    shrinkage: f64,
}

impl Tree {
    /// Predict raw leaf value for a feature vector.
    /// Leaf values in LightGBM's text format already incorporate shrinkage.
    fn predict(&self, features: &[f64]) -> f64 {
        let mut node: i32 = 0;
        loop {
            let idx = node as usize;
            let feat_idx = self.split_feature[idx];
            let feat_val = features[feat_idx];

            let child = if feat_val <= self.threshold[idx] {
                self.left_child[idx]
            } else {
                self.right_child[idx]
            };

            if child < 0 {
                let leaf_idx = (-child - 1) as usize;
                return self.leaf_value[leaf_idx];
            }
            node = child;
        }
    }
}

/// LightGBM binary classifier model.
pub struct GbmModel {
    trees: Vec<Tree>,
    num_features: usize,
    max_leaf_abs: f64,
}

impl GbmModel {
    /// Parse a LightGBM text model file.
    pub fn load(model_text: &str) -> Self {
        let lines: Vec<&str> = model_text.lines().collect();
        let mut trees = Vec::new();
        let mut num_features = 0;
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            if line.starts_with("max_feature_idx=") {
                num_features = parse_val::<usize>(line) + 1;
            }

            if line.starts_with("Tree=") {
                let tree = parse_tree(&lines, &mut i);
                trees.push(tree);
                continue;
            }

            i += 1;
        }

        let max_leaf_abs = trees
            .iter()
            .flat_map(|t| t.leaf_value.iter())
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);

        GbmModel { trees, num_features, max_leaf_abs }
    }

    /// Predict probability of class 1 (content).
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        debug_assert_eq!(features.len(), self.num_features);
        let raw: f64 = self.trees.iter().map(|t| t.predict(features)).sum();
        sigmoid(raw)
    }

    /// Predict class (true = content, false = boilerplate).
    pub fn predict(&self, features: &[f64], threshold: f64) -> bool {
        self.predict_proba(features) > threshold
    }

    /// Predict using only the first `n` trees (for speed/accuracy tradeoff).
    pub fn predict_n(&self, features: &[f64], threshold: f64, n_trees: usize) -> bool {
        debug_assert_eq!(features.len(), self.num_features);
        let n = n_trees.min(self.trees.len());
        let raw: f64 = self.trees[..n].iter().map(|t| t.predict(features)).sum();
        sigmoid(raw) > threshold
    }

    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

fn parse_tree(lines: &[&str], pos: &mut usize) -> Tree {
    let mut num_leaves = 0;
    let mut split_feature = Vec::new();
    let mut threshold = Vec::new();
    let mut left_child = Vec::new();
    let mut right_child = Vec::new();
    let mut leaf_value = Vec::new();
    let mut shrinkage = 0.05_f64;

    *pos += 1; // skip "Tree=N" line

    while *pos < lines.len() {
        let line = lines[*pos].trim();

        if line.is_empty() || line.starts_with("Tree=") {
            break;
        }

        if line.starts_with("num_leaves=") {
            num_leaves = parse_val(line);
        } else if line.starts_with("split_feature=") {
            split_feature = parse_vec(line);
        } else if line.starts_with("threshold=") {
            threshold = parse_vec(line);
        } else if line.starts_with("left_child=") {
            left_child = parse_vec(line);
        } else if line.starts_with("right_child=") {
            right_child = parse_vec(line);
        } else if line.starts_with("leaf_value=") {
            leaf_value = parse_vec(line);
        } else if line.starts_with("shrinkage=") {
            shrinkage = parse_val(line);
        }

        *pos += 1;
    }

    Tree {
        num_leaves,
        split_feature,
        threshold,
        left_child,
        right_child,
        leaf_value,
        shrinkage,
    }
}

fn parse_val<T: std::str::FromStr>(line: &str) -> T
where
    T::Err: std::fmt::Debug,
{
    line.split_once('=').unwrap().1.trim().parse().unwrap()
}

fn parse_vec<T: std::str::FromStr>(line: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    line.split_once('=')
        .unwrap()
        .1
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_load_model() {
        let model_text = include_str!("../data/model_dom.txt");
        let model = GbmModel::load(model_text);
        assert_eq!(model.num_trees(), 1635);
        assert_eq!(model.num_features(), 35);

        // Check tree 0 predicts correctly for zeros
        let zeros = vec![0.0; 35];
        let t0_raw = model.trees[0].predict(&zeros);
        assert!((t0_raw - 0.3197829639).abs() < 1e-6, "tree 0: got {t0_raw}");

        let t1_raw = model.trees[1].predict(&zeros);
        assert!((t1_raw - (-0.0111815308)).abs() < 1e-6, "tree 1: got {t1_raw}");
    }

    #[test]
    fn test_predict_matches_python() {
        let model_text = include_str!("../data/model_dom.txt");
        let model = GbmModel::load(model_text);

        // Values verified against Python: lightgbm.Booster.predict()
        let zeros = vec![0.0; 35];
        let prob = model.predict_proba(&zeros);
        assert!((prob - 0.0212071791).abs() < 1e-4, "zeros: got {prob}");

        // Content-like features
        let content = vec![
            100.0, 20.0, 3.0, 2.0, 5.0, 0.3, 0.05, 0.0, 0.0, 0.0, 0.0,
            5.0, 1.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 3.0, 5.0, 0.5,
            0.0, 0.0, 50.0, 0.0, 50.0, 0.0, 1.0, 5.0, 20.0, 5000.0, 0.1, 3.0,
        ];
        let prob = model.predict_proba(&content);
        assert!((prob - 0.4430015452).abs() < 1e-4, "content: got {prob}");

        // Boilerplate-like features
        let boiler = vec![
            20.0, 5.0, 0.0, 0.0, 4.0, 0.1, 0.5, 0.0, 15.0, 3.0, 0.75,
            2.0, 0.0, 0.0, 0.0, 0.0, 10.0, -50.0, 0.0, 6.0, -1.0, 3.0, 0.05,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 999.0, 999.0, 10.0, 2000.0, 0.5, 0.0,
        ];
        let prob = model.predict_proba(&boiler);
        assert!((prob - 0.0006101581).abs() < 1e-4, "boiler: got {prob}");
    }
}
