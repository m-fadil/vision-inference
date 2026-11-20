//! Face and scene inference primitives.
//!
//! The crate owns image preprocessing, model execution, and embedding
//! generation. Persistence, thresholds for business decisions, and API
//! concerns belong to the caller.

pub mod face;
pub mod scene;
