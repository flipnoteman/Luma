//! The `ml` suite: dense layers, MLP training steps, row-wise normalisations, attention,
//! embeddings, convolutions, classic loops (logistic regression, k-means, PCA) and batched small
//! matrix products. Groups live in `ml/`.

mod attention;
mod batched;
mod classic;
mod common;
mod conv;
mod dense;
mod embed;
mod norm;
mod train;

use crate::harness::Harness;

pub fn run(h: &mut Harness) {
    dense::run(h);
    train::run(h);
    norm::run(h);
    attention::run(h);
    embed::run(h);
    conv::run(h);
    classic::run(h);
    batched::run(h);
}
