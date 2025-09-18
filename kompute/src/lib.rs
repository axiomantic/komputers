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
//! ```rust,no_run
//! use kompute::{Manager, Tensor, Result};
//!
//! fn main() -> Result<()> {
//!     // Create a GPU manager
//!     let mut manager = Manager::new()?;
//!
//!     // Create tensors with data
//!     let tensor_a = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0])?;
//!     let tensor_b = Tensor::new(&mut manager, &[4.0f32, 5.0, 6.0])?;
//!     let mut tensor_out: Tensor<f32> = Tensor::uninit(&mut manager, 3)?;
//!
//!     // Tensor operations would involve compute shaders
//!     // (Algorithm creation and execution not yet implemented)
//!
//!     // Access tensor properties
//!     println!("Tensor A size: {}", tensor_a.size());
//!     println!("Tensor B size: {}", tensor_b.size());
//!
//!     Ok(())
//! }
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

pub mod algorithm;
pub mod error;
pub mod handle_tracker;
pub mod manager;
pub mod multi_queue;
pub mod sequence;
pub mod tensor;
pub mod vulkan_setup;

// Feature-gated async support
#[cfg(feature = "async")]
pub mod async_executor;

#[cfg(feature = "async")]
pub mod async_gpu;

// Re-export main types
pub use algorithm::Algorithm;
pub use error::{KomputeError, Result};
pub use handle_tracker::{HandleStatistics, HandleTracker, TrackedHandle};
pub use manager::Manager;
pub use multi_queue::{DeviceCapabilities, MultiQueueManager, MultiQueueWorkflow};
pub use sequence::Sequence;
pub use tensor::{Tensor, TensorData};
pub use vulkan_setup::{configure_vulkan_environment, verify_vulkan_availability};

// Feature-gated async support
#[cfg(feature = "async")]
pub use async_executor::{AsyncGpuExecutor, AsyncSequence, AsyncSequenceExecution};

#[cfg(feature = "async")]
pub use async_gpu::{
    AsyncComputeKernel, AsyncGpuContext, AsyncSequence as AsyncGpuSequence, AsyncTensor,
    AsyncWorkflowBuilder,
};

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
