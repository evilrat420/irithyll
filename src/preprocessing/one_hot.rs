//! Streaming one-hot encoder for categorical features.

use crate::pipeline::StreamingPreprocessor;

/// Streaming one-hot encoder that discovers categories online from the data stream.
///
/// Specified categorical feature indices are expanded into binary one-hot vectors.
/// Non-categorical features pass through unchanged. Categories are maintained in
/// sorted order for deterministic column assignment.
///
/// A per-feature safety cap ([`max_categories`](Self::max_categories), default 64)
/// prevents unbounded expansion from high-cardinality features.
///
/// # Example
///
/// ```
/// use irithyll::preprocessing::OneHotEncoder;
/// use irithyll::pipeline::StreamingPreprocessor;
///
/// let mut enc = OneHotEncoder::new(vec![1]); // feature 1 is categorical
///
/// let out1 = enc.update_and_transform(&[5.0, 0.0, 3.0]);
/// let out2 = enc.update_and_transform(&[5.0, 1.0, 3.0]);
/// let out3 = enc.update_and_transform(&[5.0, 2.0, 3.0]);
///
/// // After seeing {0, 1, 2}, feature 1 expands to 3 binary columns.
/// assert_eq!(enc.output_dim(), Some(5)); // 1 passthrough + 3 one-hot + 1 passthrough
/// ```
#[derive(Clone, Debug)]
pub struct OneHotEncoder {
    /// Which input feature positions are categorical.
    categorical_indices: Vec<usize>,
    /// Per-categorical-index list of discovered category values (sorted).
    ///
    /// `categories[j]` corresponds to `categorical_indices[j]`.
    categories: Vec<Vec<f64>>,
    /// Maximum categories per feature (safety cap).
    max_categories: usize,
    /// Input feature dimensionality, recorded on first sample for `output_dim`.
    n_input_features: Option<usize>,
}

impl OneHotEncoder {
    /// Create a one-hot encoder with the given categorical feature indices.
    ///
    /// Uses the default `max_categories` of 64.
    ///
    /// ```
    /// use irithyll::preprocessing::OneHotEncoder;
    /// let enc = OneHotEncoder::new(vec![0, 2]);
    /// assert_eq!(enc.max_categories(), 64);
    /// ```
    pub fn new(categorical_indices: Vec<usize>) -> Self {
        let n_cat = categorical_indices.len();
        Self {
            categorical_indices,
            categories: vec![Vec::new(); n_cat],
            max_categories: 64,
            n_input_features: None,
        }
    }

    /// Create a one-hot encoder with a custom per-feature category cap.
    ///
    /// # Panics
    ///
    /// Panics if `max_categories` is 0.
    ///
    /// ```
    /// use irithyll::preprocessing::OneHotEncoder;
    /// use irithyll::pipeline::StreamingPreprocessor;
    /// let enc = OneHotEncoder::with_max_categories(vec![1], 10);
    /// assert_eq!(enc.output_dim(), None);
    /// ```
    pub fn with_max_categories(categorical_indices: Vec<usize>, max_categories: usize) -> Self {
        assert!(max_categories > 0, "max_categories must be > 0");
        let n_cat = categorical_indices.len();
        Self {
            categorical_indices,
            categories: vec![Vec::new(); n_cat],
            max_categories,
            n_input_features: None,
        }
    }

    /// The input feature indices that are treated as categorical.
    pub fn categorical_indices(&self) -> &[usize] {
        &self.categorical_indices
    }

    /// The discovered category values for each categorical index (sorted).
    ///
    /// `categories()[j]` corresponds to `categorical_indices()[j]`.
    pub fn categories(&self) -> &[Vec<f64>] {
        &self.categories
    }

    /// The maximum number of categories allowed per feature.
    pub fn max_categories(&self) -> usize {
        self.max_categories
    }

    /// Number of discovered categories for the given categorical index slot.
    ///
    /// `cat_idx` is a positional index into [`categorical_indices`](Self::categorical_indices),
    /// **not** the original feature index.
    ///
    /// # Panics
    ///
    /// Panics if `cat_idx >= categorical_indices.len()`.
    pub fn n_discovered_categories(&self, cat_idx: usize) -> usize {
        self.categories[cat_idx].len()
    }

    /// Return the categorical-feature ordinal index for `feature_idx`, or `None`.
    fn cat_ordinal(&self, feature_idx: usize) -> Option<usize> {
        self.categorical_indices
            .iter()
            .position(|&ci| ci == feature_idx)
    }

    /// Register a new category value for the `j`-th categorical feature,
    /// maintaining sorted order. Does nothing if the value already exists
    /// or if `max_categories` has been reached.
    fn register_category(&mut self, j: usize, value: f64) {
        let cats = &mut self.categories[j];
        if cats.len() >= self.max_categories {
            return;
        }
        match cats.binary_search_by(|probe| probe.partial_cmp(&value).unwrap()) {
            Ok(_) => {} // already present
            Err(pos) => {
                cats.insert(pos, value);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingPreprocessor impl
// ---------------------------------------------------------------------------

impl StreamingPreprocessor for OneHotEncoder {
    /// Update category tables from this sample and return one-hot-encoded features.
    ///
    /// For each categorical index, if the value is not yet known and the cap
    /// has not been reached, it is added to the sorted category list. The
    /// sample is then transformed using the (possibly expanded) category tables.
    fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64> {
        self.n_input_features = Some(features.len());

        // Register new categories.
        let n_cat = self.categorical_indices.len();
        for j in 0..n_cat {
            let ci = self.categorical_indices[j];
            if ci < features.len() {
                let value = features[ci];
                self.register_category(j, value);
            }
        }

        self.transform(features)
    }

    /// Transform features using current category tables without updating them.
    ///
    /// Unknown categories produce all-zero columns.
    fn transform(&self, features: &[f64]) -> Vec<f64> {
        let mut out = Vec::new();

        for (i, &fval) in features.iter().enumerate() {
            if let Some(j) = self.cat_ordinal(i) {
                // Categorical: expand to one-hot.
                let cats = &self.categories[j];
                match cats.binary_search_by(|probe| probe.partial_cmp(&fval).unwrap()) {
                    Ok(pos) => {
                        for k in 0..cats.len() {
                            out.push(if k == pos { 1.0 } else { 0.0 });
                        }
                    }
                    Err(_) => {
                        // Unknown category: all zeros.
                        out.extend(std::iter::repeat(0.0).take(cats.len()));
                    }
                }
            } else {
                // Numeric: passthrough.
                out.push(fval);
            }
        }

        out
    }

    /// Number of output features, or `None` if no data has been seen yet.
    ///
    /// Computed as `passthrough_features + sum(categories[i].len())`.
    /// Returns `None` if no data has been seen yet (all categories empty).
    fn output_dim(&self) -> Option<usize> {
        let n_input = self.n_input_features?;
        if self.categories.iter().all(|c| c.is_empty()) {
            return None;
        }
        let n_categorical = self.categorical_indices.len();
        let n_one_hot: usize = self.categories.iter().map(|c| c.len()).sum();
        Some(n_input - n_categorical + n_one_hot)
    }

    /// Reset to initial untrained state -- clears all discovered categories.
    /// Preserves `categorical_indices` and `max_categories`.
    fn reset(&mut self) {
        for cats in &mut self.categories {
            cats.clear();
        }
        self.n_input_features = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_non_categorical() {
        let mut enc = OneHotEncoder::new(vec![]);
        let input = [1.0, 2.5, 3.0];
        let out = enc.update_and_transform(&input);
        assert_eq!(out, vec![1.0, 2.5, 3.0]);
    }

    #[test]
    fn discovers_categories_online() {
        let mut enc = OneHotEncoder::new(vec![0]);

        // First sample: only category 0 known.
        let out1 = enc.update_and_transform(&[0.0, 7.0]);
        assert_eq!(enc.n_discovered_categories(0), 1);
        assert_eq!(out1, vec![1.0, 7.0]);

        // Second sample: category 1 discovered, output dim grows.
        let out2 = enc.update_and_transform(&[1.0, 8.0]);
        assert_eq!(enc.n_discovered_categories(0), 2);
        assert_eq!(out2, vec![0.0, 1.0, 8.0]);

        // Third sample: category 2 discovered.
        let out3 = enc.update_and_transform(&[2.0, 9.0]);
        assert_eq!(enc.n_discovered_categories(0), 3);
        assert_eq!(out3, vec![0.0, 0.0, 1.0, 9.0]);
    }

    #[test]
    fn max_categories_cap() {
        let mut enc = OneHotEncoder::with_max_categories(vec![0], 2);

        enc.update_and_transform(&[0.0]);
        enc.update_and_transform(&[1.0]);
        // Third unique value should NOT be added.
        enc.update_and_transform(&[2.0]);

        assert_eq!(enc.n_discovered_categories(0), 2);
        // Category 2 is unknown, so all zeros in the 2 one-hot columns.
        let out = enc.transform(&[2.0]);
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn one_hot_encoding_correct() {
        let mut enc = OneHotEncoder::new(vec![0]);

        // Discover categories 0, 1, 2.
        enc.update_and_transform(&[0.0]);
        enc.update_and_transform(&[1.0]);
        enc.update_and_transform(&[2.0]);

        // Categories sorted: [0, 1, 2].
        // cat=0 -> [1.0, 0.0, 0.0]
        assert_eq!(enc.transform(&[0.0]), vec![1.0, 0.0, 0.0]);
        // cat=1 -> [0.0, 1.0, 0.0]
        assert_eq!(enc.transform(&[1.0]), vec![0.0, 1.0, 0.0]);
        // cat=2 -> [0.0, 0.0, 1.0]
        assert_eq!(enc.transform(&[2.0]), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn unknown_category_in_transform() {
        let mut enc = OneHotEncoder::new(vec![0]);

        // Learn categories 0 and 1.
        enc.update_and_transform(&[0.0, 1.0]);
        enc.update_and_transform(&[1.0, 2.0]);

        // Transform with unseen category 5 -> all zeros in one-hot columns.
        let out = enc.transform(&[5.0, 3.0]);
        assert_eq!(out, vec![0.0, 0.0, 3.0]);
    }

    #[test]
    fn mixed_categorical_and_continuous() {
        // Features: [cat_a, numeric, cat_b]
        let mut enc = OneHotEncoder::new(vec![0, 2]);

        enc.update_and_transform(&[0.0, 5.0, 10.0]);
        enc.update_and_transform(&[1.0, 6.0, 20.0]);
        enc.update_and_transform(&[0.0, 7.0, 30.0]);

        // cat_a has {0, 1} -> 2 columns
        // cat_b has {10, 20, 30} -> 3 columns
        // output_dim = 2 + 1 + 3 = 6
        assert_eq!(enc.output_dim(), Some(6));

        // Transform [1.0, 9.0, 20.0]:
        //   cat_a=1 -> [0.0, 1.0]
        //   numeric -> [9.0]
        //   cat_b=20 -> [0.0, 1.0, 0.0]
        let out = enc.transform(&[1.0, 9.0, 20.0]);
        assert_eq!(out, vec![0.0, 1.0, 9.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn reset_clears_categories() {
        let mut enc = OneHotEncoder::new(vec![0]);
        enc.update_and_transform(&[0.0, 1.0]);
        enc.update_and_transform(&[1.0, 2.0]);
        assert!(enc.output_dim().is_some());

        enc.reset();
        assert_eq!(enc.n_discovered_categories(0), 0);
        assert_eq!(enc.output_dim(), None);
    }

    #[test]
    fn deterministic_ordering() {
        // Feed categories in reverse order; output columns should still be
        // sorted by value.
        let mut enc = OneHotEncoder::new(vec![0]);
        enc.update_and_transform(&[2.0]);
        enc.update_and_transform(&[0.0]);
        enc.update_and_transform(&[1.0]);

        // Categories sorted: [0, 1, 2].
        // Encode cat=0 -> [1.0, 0.0, 0.0]
        assert_eq!(enc.transform(&[0.0]), vec![1.0, 0.0, 0.0]);
        // Encode cat=1 -> [0.0, 1.0, 0.0]
        assert_eq!(enc.transform(&[1.0]), vec![0.0, 1.0, 0.0]);
        // Encode cat=2 -> [0.0, 0.0, 1.0]
        assert_eq!(enc.transform(&[2.0]), vec![0.0, 0.0, 1.0]);
    }
}
