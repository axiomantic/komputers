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

#![deny(missing_docs)]
#![warn(rust_2018_idioms)]

// =============================================================================
// BUILD-TIME VERSION COMPATIBILITY CHECKS
// =============================================================================

// Ensure Kompute master is enabled (the only supported version for v0.1.0 release)
#[cfg(not(feature = "kompute-master"))]
compile_error!(
    "
❌ NO KOMPUTE VERSION SELECTED
✅ SOLUTION: v0.1.0 release only supports kompute-master
📖 Example: cargo build --features kompute-master
⚠️  Note: This release uses Kompute master branch due to compatibility requirements
💡 Future releases may support official Kompute versions when compatibility is resolved
"
);

pub mod algorithm;
pub mod algorithm_binding;
pub mod convenience;
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
pub mod vulkan_config;
pub mod vulkan_setup;

// Type-safe handle system for compile-time validation
pub mod handle;
pub mod typed_manager;
pub mod types;

// Runtime validation system for cross-manager handle safety
pub mod runtime_validation;

// Workgroup validation system for GPU compute operations
pub mod workgroup_validation;

// Feature-gated async support (temporarily disabled - needs update for consuming builder)
// #[cfg(feature = "async")]
// pub mod async_executor;

// #[cfg(feature = "async")]
// pub mod async_gpu;

#[cfg(feature = "async")]
pub mod async_sequence;

// Performance profiling and metrics system
pub mod performance_metrics;

// Re-export main types
pub use algorithm::{Algorithm, AlgorithmPerformanceHints};
pub use algorithm_binding::{
    AccessPattern, AlgorithmBinding, BindingTemplate, BindingValidationReport,
    CommonBindingTemplates, ParameterSpec, TensorBinding, TensorDataType, ValidationRule,
};
pub use convenience::{
    BenchmarkConfig, BenchmarkResults, BinaryOp, DevicePreference, GpuOps, MathOps,
    OperationResult, ReductionOp, TensorExt,
};
// Crash debug feature removed for v0.1.0 release due to dependency issues
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
pub use vulkan_config::VulkanConfig;
pub use vulkan_setup::{
    configure_vulkan_environment, configure_vulkan_environment_intelligent,
    diagnose_vulkan_environment, verify_vulkan_availability,
};

// Type-safe handle system exports
pub use handle::{
    AlgorithmType, AsyncAlgorithmHandle, AsyncMarker, AsyncQueueHandle, AsyncSequenceHandle,
    AsyncTensorHandle, CompatibleHandles, Handle, IsAsync, IsSync, QueueType, SequenceType,
    SyncAlgorithmHandle, SyncMarker, SyncQueueHandle, SyncSequenceHandle, SyncTensorHandle,
    TensorType, UntypedHandle,
};
pub use runtime_validation::{
    get_validation_metrics, is_validation_enabled, is_validation_overhead_acceptable,
    register_handle_with_manager, register_handles_batch, set_validation_enabled,
    validate_handle_with_manager, validate_handles_same_manager, ManagerInstance,
};
pub use typed_manager::{
    create_async_manager, create_async_manager_with_config, create_sync_manager,
    create_sync_manager_with_config, AsyncManager, AsyncTypedAlgorithm, AsyncTypedSequence,
    AsyncTypedTensor, SyncManager, TypedAlgorithm, TypedSequence, TypedTensor,
};
pub use types::{
    AsyncManagerType, DefaultAsyncManager, DefaultManagerId, DefaultSyncManager, HasManagerId,
    ManagerId, RuntimeManagerId, SyncManagerType, TypeSafetyError,
};
pub use workgroup_validation::{
    common_1d, common_2d, common_3d, DynamicWorkgroupConfig, GpuLimits, VendorInfo,
    WorkgroupConfig as TypedWorkgroupConfig, WorkgroupPerformanceHints, WorkgroupValidation,
};

// Feature-gated async support (temporarily disabled - needs update for consuming builder)
// #[cfg(feature = "async")]
// pub use async_executor::{AsyncGpuExecutor, AsyncSequence, AsyncSequenceExecution};

// #[cfg(feature = "async")]
// pub use async_gpu::{
//     AsyncComputeKernel, AsyncGpuContext, AsyncSequence as AsyncGpuSequence, AsyncTensor,
//     AsyncWorkflowBuilder,
// };

#[cfg(feature = "async")]
pub use async_sequence::{
    async_utils, AsyncPerformanceMetrics, AsyncSequence as TypedAsyncSequence,
    AsyncSequenceExecution as TypedAsyncSequenceExecution, AsyncSequenceResult,
};

// Performance profiling and metrics (with aliases to avoid conflicts)
pub use performance_metrics::{
    global_profiler, GlobalStatistics, OperationCategory, OperationProfile, OptimizationCategory,
    OptimizationRecommendation as MetricsOptimizationRecommendation, PerformanceImprovement,
    PerformanceProfiler as MetricsPerformanceProfiler, ProfilerConfig,
    RecommendationPriority as MetricsRecommendationPriority, SessionReport, SessionSummary,
    ValidationMetrics,
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
