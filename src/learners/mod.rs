//! Streaming learner implementations for polymorphic model composition.
//!
//! Each learner in this module implements the [`crate::learner::StreamingLearner`] trait,
//! enabling runtime-polymorphic stacking via `Box<dyn StreamingLearner>`.
//!
//! # Learners
//!
//! | Module | Type | Algorithm |
//! |--------|------|-----------|
//! | [`linear`] | [`StreamingLinearModel`] | SGD with L1/L2/ElasticNet regularization |
//! | [`naive_bayes`] | [`GaussianNB`] | Incremental Gaussian Naive Bayes classifier |
//! | [`mondrian`] | [`MondrianForest`] | Online random forest with lifetime-based splits |
//! | [`rls`] | [`RecursiveLeastSquares`] | Exact streaming OLS via Sherman-Morrison |
//! | [`rls`] | [`StreamingPolynomialRegression`] | RLS with polynomial feature expansion |
//! | [`rls`] | [`LocallyWeightedRegression`] | Nadaraya-Watson with circular buffer |

pub mod linear;
pub mod mondrian;
pub mod naive_bayes;
pub mod rls;

pub use linear::StreamingLinearModel;
pub use mondrian::MondrianForest;
pub use naive_bayes::GaussianNB;
pub use rls::{LocallyWeightedRegression, RecursiveLeastSquares, StreamingPolynomialRegression};
