//! Face and scene inference primitives.
//!
//! The crate owns image preprocessing, model execution, and embedding
//! generation. Persistence, thresholds for business decisions, and API
//! concerns belong to the caller.
#![doc = include_str!("../README.md")]

mod math;
mod runtime;

pub use ort;

pub mod face;
pub mod scene;
