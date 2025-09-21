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
//! ## Basic Tensor Operations
//!
//! ```rust,no_run
//! use kompute::{Manager, Tensor, Result};
//!
//! fn basic_example() -> Result<()> {
//!     // Create a GPU manager
//!     let mut manager = Manager::new()?;
//!
//!     // Create tensors with data
//!     let tensor_a = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0])?;
//!     let tensor_b = Tensor::new(&mut manager, &[4.0f32, 5.0, 6.0])?;
//!     let mut tensor_out: Tensor<f32> = Tensor::uninit(&mut manager, 3)?;
//!
//!     // Access tensor properties
//!     println!("Tensor A size: {}", tensor_a.size());
//!     println!("Tensor B size: {}", tensor_b.size());
//!
//!     Ok(())
//! }
//! ```
//!
//! ## High-Level GPU Operations
//!
//! ```rust,no_run
//! use kompute::{GpuOperations, Tensor, WorkgroupConfig, Result};
//!
//! fn gpu_operations_example() -> Result<()> {
//!     // Create high-level GPU operations manager
//!     let mut ops = GpuOperations::new()?;
//!
//!     // Create tensors for computation
//!     let source = Tensor::new(ops.manager(), &[1.0f32, 2.0, 3.0, 4.0])?;
//!     let mut dest: Tensor<f32> = Tensor::uninit(ops.manager(), 4)?;
//!
//!     // Perform GPU copy operation
//!     ops.copy(&source, &mut dest)?;
//!
//!     // Configure optimal workgroup for 1000 elements
//!     let workgroup = WorkgroupConfig::optimal_1d(1000);
//!     println!("Optimal workgroup: {}x{}x{}", workgroup.x, workgroup.y, workgroup.z);
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

// =============================================================================
// BUILD-TIME VERSION COMPATIBILITY CHECKS
// =============================================================================

// Ensure exactly one Kompute version feature is enabled
#[cfg(not(any(
    feature = "kompute-master",
    feature = "kompute-v090",
    feature = "kompute-v081"
)))]
compile_error!("
❌ NO KOMPUTE VERSION SELECTED
✅ SOLUTION: Use kompute-master (the only currently supported version as of September 2025)
📖 Example: cargo build --features kompute-master
⚠️  Note: v0.8.1 and v0.9.0 are currently not supported due to file structure differences and macOS/MoltenVK compatibility issues
💡 This status will likely change when the next official Kompute release is published
");

#[cfg(all(feature = "kompute-master", feature = "kompute-v090"))]
compile_error!("
❌ MULTIPLE KOMPUTE VERSIONS ENABLED
✅ SOLUTION: Use only kompute-master (the only currently supported version as of September 2025)
⚠️  Note: v0.9.0 is currently not supported due to file structure differences and macOS/MoltenVK compatibility issues
💡 This status will likely change when the next official Kompute release is published
");

#[cfg(all(feature = "kompute-master", feature = "kompute-v081"))]
compile_error!("
❌ MULTIPLE KOMPUTE VERSIONS ENABLED
✅ SOLUTION: Use only kompute-master (the only currently supported version as of September 2025)
⚠️  Note: v0.8.1 is currently not supported due to file structure differences and macOS/MoltenVK compatibility issues
💡 This status will likely change when the next official Kompute release is published
");

#[cfg(all(feature = "kompute-v090", feature = "kompute-v081"))]
compile_error!("
❌ MULTIPLE KOMPUTE VERSIONS ENABLED
✅ SOLUTION: Use only kompute-master (the only currently supported version as of September 2025)
⚠️  Note: v0.8.1 and v0.9.0 are currently not supported due to file structure differences and macOS/MoltenVK compatibility issues
💡 This status will likely change when the next official Kompute release is published
");

// Platform-specific compatibility warnings
#[cfg(all(target_os = "macos", feature = "kompute-v090"))]
compile_error!("
❌ INCOMPATIBLE: Kompute v0.9.0 + macOS causes Manager segfaults!
✅ SOLUTION: Use 'kompute-master' (the only currently supported version as of September 2025)
📖 See: https://github.com/KomputeProject/kompute/issues/378
🔧 Command: cargo build --features kompute-master
⚠️  Note: Current implementation is designed for master branch with file structure differences from v0.9.0
💡 This status will likely change when the next official Kompute release is published
");

#[cfg(all(target_os = "macos", feature = "kompute-v081"))]
compile_error!("
❌ INCOMPATIBLE: Kompute v0.8.1 + macOS causes compatibility issues!
✅ SOLUTION: Use 'kompute-master' (the only currently supported version as of September 2025)
🔧 Command: cargo build --features kompute-master
⚠️  Note: Current implementation is designed for master branch; older versions have compatibility issues
💡 This status will likely change when the next official Kompute release is published
");

pub mod algorithm;
pub mod algorithm_binding;
pub mod convenience;
#[cfg(feature = "crash-debug")]
pub mod crash_debug;
#[cfg(feature = "debug")]
pub mod debug_tools;
pub mod error;
pub mod error_recovery;
#[cfg(feature = "glsl")]
pub mod glsl_compiler;
pub mod handle_tracker;
pub mod manager;
pub mod memory_optimizer;
pub mod multi_queue;
pub mod operations;
pub mod performance_profiler;
pub mod platform;
pub mod queue;
pub mod sequence;
pub mod spirv_shaders;
pub mod spirv_utils;
pub mod tensor;
#[cfg(test)]
pub mod test_utils;
pub mod vulkan_setup;

// Feature-gated async support
#[cfg(feature = "async")]
pub mod async_executor;

#[cfg(feature = "async")]
pub mod async_gpu;

// Re-export main types
pub use algorithm::Algorithm;
pub use algorithm_binding::{
    AccessPattern, AlgorithmBinding, BindingTemplate, BindingValidationReport,
    CommonBindingTemplates, ParameterSpec, TensorBinding, TensorDataType, ValidationRule,
};
pub use convenience::{
    BenchmarkConfig, BenchmarkResults, BinaryOp, DevicePreference, GpuOps, MathOps,
    OperationResult, ReductionOp, TensorExt,
};
#[cfg(feature = "crash-debug")]
pub use crash_debug::{
    debug_execute, install_crash_handlers, test_crash_debug, CrashAnalysis, CrashInfo, GpuContext,
    MemoryInfo, ProcessInfo,
};
#[cfg(feature = "debug")]
pub use debug_tools::{
    DebugContext, DebugEvent, DebugEventType, DebugReport, DebugSeverity, DebugUtils,
    MemorySnapshot, PerformanceCounters, ValidationError, ValidationErrorType,
};
pub use error::{KomputeError, Result};
pub use error_recovery::{
    DiagnosticReport, ErrorCategory, ErrorRecoveryManager, ErrorSeverity, RecoveryResult,
};
#[cfg(feature = "glsl")]
pub use glsl_compiler::{
    BindingInfo, CompilationResult, CompilerOptions, DataType as GlslDataType, DescriptorType,
    GlslCompiler, GlslTemplates, PushConstantInfo, PushConstantMember, ShaderReflection,
    SpecConstantInfo,
};
pub use handle_tracker::{HandleStatistics, HandleTracker, TrackedHandle};
pub use manager::Manager;
pub use memory_optimizer::{
    AllocationStrategy, MemoryOptimizer, MemoryPool, PerformanceMetric, PerformanceMetrics,
};
pub use multi_queue::{DeviceCapabilities, MultiQueueManager, MultiQueueWorkflow};
pub use operations::{DeviceInfo, GpuOperations, MemoryPatterns, WorkgroupConfig};
pub use performance_profiler::{
    GpuMetric, GpuUtilizationSummary, MemoryAllocationPattern, MemoryAnalysis,
    MemorySnapshot as ProfilerMemorySnapshot, OperationSummary, OptimizationRecommendation,
    PerformanceProfiler, PerformanceReport, RecommendationCategory, RecommendationPriority,
};
pub use platform::{
    EnvironmentValidator, GpuSupport, MetalGpuDetection, PlatformInfo, RuntimeUtils,
    ValidationReport,
};
pub use queue::Queue;
pub use sequence::Sequence;
pub use spirv_utils::{CommonKernels, ShaderCompilationGuide, SpirvUtils};
pub use tensor::{DataType, Tensor, TensorData};
pub use vulkan_setup::{
    configure_vulkan_environment, configure_vulkan_environment_intelligent,
    diagnose_vulkan_environment, verify_vulkan_availability,
};

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
        assert!(!VERSION.trim().is_empty());
        assert!(!KOMPUTE_VERSION.trim().is_empty());
    }
}
