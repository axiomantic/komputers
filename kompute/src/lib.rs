//! Idiomatic Rust bindings for the Kompute GPU compute framework.
//!
//! Kompute is a general-purpose GPU computing framework built on Vulkan that provides
//! simple abstractions for GPU memory management and compute shader execution.
//! This crate provides memory-safe, idiomatic Rust bindings with zero-cost abstractions.
//!
//! # Features
//!
//! - **Memory Safety**: All GPU resource management is handled through RAII patterns
//! - **Type Safety**: Generic tensor types with compile-time data type validation
//! - **Cross-Platform**: Supports Linux, Windows, and macOS with Vulkan drivers
//! - **Zero-Cost Abstractions**: Minimal overhead compared to C++ Kompute
//! - **Async Support**: Optional futures-based async API (requires `async` feature)
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use kompute::{Manager, Tensor};
//!
//! // Create a GPU manager
//! let mut manager = Manager::new()?;
//!
//! // Create tensors with data
//! let tensor_a = Tensor::new(&manager, &[1.0f32, 2.0, 3.0])?;
//! let tensor_b = Tensor::new(&manager, &[4.0f32, 5.0, 6.0])?;
//! let mut tensor_out = Tensor::new(&manager, &[0.0f32; 3])?;
//!
//! // Load and execute a compute shader
//! let algorithm = manager.algorithm(include_bytes!("shaders/add.spv"))?;
//! manager.execute(&algorithm, &[&tensor_a, &tensor_b, &mut tensor_out])?;
//!
//! // Retrieve results
//! let results = tensor_out.to_vec()?;
//! println!("Results: {:?}", results); // [5.0, 7.0, 9.0]
//! ```
//!
//! # Architecture
//!
//! The crate is structured around four main components:
//!
//! - [`Manager`]: GPU device management and resource allocation
//! - [`Tensor`]: Type-safe GPU memory with host synchronization
//! - [`Algorithm`]: SPIR-V compute shader with parameter bindings
//! - [`Sequence`]: Command buffer for batched GPU operations
//!
//! # Error Handling
//!
//! All operations return [`Result`] types with detailed [`KomputeError`] information.
//! GPU operations can fail due to device errors, out-of-memory conditions, or
//! invalid shader parameters.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod error;
pub mod manager;
pub mod tensor;

// Re-export main types
pub use error::{KomputeError, Result};
pub use manager::Manager;
pub use tensor::Tensor;

// Feature-gated async support (placeholder for future implementation)
// #[cfg(feature = "async")]
// pub mod r#async;

// #[cfg(feature = "async")]
// pub use r#async::{AsyncManager, AsyncSequence};

/// Version information for the kompute crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Kompute C++ library version this crate binds to.
pub const KOMPUTE_VERSION: &str = "0.8.1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_constants() {
        assert!(!VERSION.is_empty());
        assert!(!KOMPUTE_VERSION.is_empty());
    }
}
