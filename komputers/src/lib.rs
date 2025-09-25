//! # Komputers: Type-Safe GPU Computing for Rust
//!
//! A Rust wrapper for Vulkan GPU compute operations, built on the [Kompute](https://github.com/KomputeProject/kompute)
//! C++ framework. Komputers provides memory-safe GPU computing with compile-time and runtime safety checks.
//!
//! ## Features
//!
//! - **Type Safety**: Compile-time validation of GPU resource usage and tensor operations
//! - **Memory Safety**: RAII resource management with automatic cleanup and lifetime tracking
//! - **Cross-Manager Protection**: Runtime validation prevents resource corruption across GPU contexts
//! - **Consuming Builders**: Rust-idiomatic APIs that prevent accidental reuse of GPU sequences
//! - **Async Support**: Optional async/await support for non-blocking GPU operations
//!
//! ## Quick Start
//!
//! ### Basic GPU Operations
//!
//! ```rust,no_run
//! use komputers::{Manager, Tensor, Result};
//!
//! fn main() -> Result<()> {
//!     let mut manager = Manager::new()?;
//!     let tensor = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0, 4.0])?;
//!
//!     // Consuming builder pattern ensures sequences are used exactly once
//!     let result = manager
//!         .sequence()?
//!         .record_tensor_sync_device(tensor.handle())?
//!         .record_tensor_sync_local(tensor.handle())?
//!         .execute()?;
//!
//!     println!("Executed {} operations in {:?}",
//!              result.operation_count, result.execution_time);
//!     Ok(())
//! }
//! ```
//!
//! ### Cross-Manager Safety
//!
//! ```rust,compile_fail
//! use komputers::{Manager, Tensor};
//!
//! fn cross_manager_example() -> komputers::Result<()> {
//!     let mut manager1 = Manager::new()?;
//!     let mut manager2 = Manager::new()?;
//!
//!     let tensor = Tensor::new(&mut manager1, &[1.0f32])?;
//!
//!     // This will be caught at runtime with a clear error message
//!     manager2.sequence()?
//!         .record_tensor_sync_device(tensor.handle())?
//!         .execute()?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Async Operations
//!
//! ```rust,no_run
//! # #[cfg(feature = "async")]
//! use komputers::*;
//!
//! # #[cfg(feature = "async")]
//! async fn async_example() -> Result<()> {
//!     let mut manager = Manager::new()?;
//!     let tensor = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0])?;
//!
//!     let result = manager
//!         .sequence()?
//!         .record_tensor_sync_device(tensor.handle())?
//!         .execute_async()?
//!         .await?;
//!
//!     println!("Async execution completed in {:?}", result.execution_time);
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! Komputers combines compile-time and runtime validation:
//!
//! - **[`Manager`]**: GPU device management with resource tracking
//! - **[`Tensor<T>`]**: Type-safe GPU memory with automatic cleanup
//! - **[`Sequence<State>`]**: Consuming builder for operation chaining
//! - **[`Algorithm`]**: SPIR-V compute shaders with validation
//! - **Handle System**: Phantom types prevent resource misuse
//! - **Runtime Validation**: Cross-manager usage detection
//!
//! ## Implementation Details
//!
//! ### Phantom Type System
//! ```rust,ignore
//! pub struct Handle<ObjectType, SyncMode, ManagerType> {
//!     value: u64,
//!     _phantom: PhantomData<(ObjectType, SyncMode, ManagerType)>,
//! }
//! ```
//!
//! ### Consuming Builder Pattern
//! ```rust,ignore
//! impl Sequence<Recording> {
//!     pub fn record_tensor_sync_device(mut self, handle: u64) -> Result<Self>;
//!     pub fn execute(self) -> Result<SequenceResult>;  // Consumes self
//! }
//! ```
//!
//! ### Runtime Validation
//! ```rust,ignore
//! static HANDLE_REGISTRY: LazyLock<Mutex<HandleRegistry>> =
//!     LazyLock::new(|| Mutex::new(HandleRegistry::new()));
//! ```
//!
//! ## Komputers vs wgpu
//!
//! **When to choose Komputers:**
//! - High-level compute operations with tensors and algorithms
//! - Need built-in safety validation for cross-manager resource usage
//! - Prefer consuming builders that prevent accidental reuse
//! - Working with machine learning, scientific computing, or image processing
//! - Want automatic memory management and RAII cleanup
//!
//! **When to choose wgpu:**
//! - Need graphics rendering (wgpu supports both compute and graphics)
//! - Want lower-level control over GPU resources and binding
//! - Prefer WebGPU API compatibility and broader platform support
//! - Need more mature ecosystem and documentation
//! - Working with game development or complex graphics pipelines
//!
//! **Feature Comparison:**
//!
//! | Feature | Komputers | wgpu |
//! |---------|-----------|------|
//! | API Level | High-level compute wrapper | Low-level WebGPU implementation |
//! | Safety Model | Compile-time + runtime validation | Standard Rust memory safety |
//! | Graphics Support | Compute only | Full graphics + compute |
//! | Backend | Vulkan via Kompute | Vulkan, Metal, DX12, WebGPU |
//! | Tensor Operations | Built-in tensor types | Manual buffer management |
//! | Cross-Manager Safety | Automatic validation | Manual resource tracking |
//! | Learning Curve | Moderate (tensor-focused) | Steep (low-level GPU concepts) |
//! | Use Cases | ML, scientific computing, GPGPU | Games, graphics, general compute |
//!
//! ## Error Handling
//!
//! Operations return [`Result`] types with detailed [`KomputeError`] information:
//!
//! ```rust,no_run
//! # use komputers::*;
//! # fn example() -> Result<()> {
//! match manager.sequence()?.execute() {
//!     Ok(result) => {
//!         println!("Success: {} operations", result.operation_count);
//!     },
//!     Err(KomputeError::TypeSafety(safety_error)) => {
//!         eprintln!("Safety validation failed: {}", safety_error);
//!     },
//!     Err(other) => {
//!         eprintln!("GPU error: {}", other);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

#![deny(missing_docs)]
#![warn(rust_2018_idioms)]

// =============================================================================
// BUILD-TIME VERSION COMPATIBILITY CHECKS
// =============================================================================

// Ensure Kompute master is enabled (the only supported version for v0.1.0 release)
#[cfg(not(feature = "kompute-master"))]
compile_error!(
    "
NO KOMPUTE VERSION SELECTED
SOLUTION: v0.1.0 release only supports kompute-master
Example: cargo build --features kompute-master
Note: This release uses Kompute master branch due to compatibility requirements
Future releases may support official Kompute versions when compatibility is resolved
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
pub mod memory_layout;
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

// Resource lifecycle and borrowing safety system
pub mod lifetime_safety;

// Compute shader dispatch safety system
pub mod dispatch_safety;

// Feature-gated async support - async_executor and async_gpu removed due to deprecated APIs

#[cfg(feature = "async")]
pub mod async_sequence;

// Performance profiling and metrics system
pub mod performance_metrics;

// Re-export main types
pub use algorithm::{
    Algorithm, AlgorithmPerformanceHints, ShaderDataType, ShaderInterface, TypeSafeAlgorithm,
};
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
pub use memory_layout::{alignment, AlignedData, AlignedTensor, GpuAligned, MemoryLayoutInfo};
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
pub use tensor::{DataType, DimensionSafety, DimensionTensor, Tensor, TensorData};
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

// Lifetime safety exports
pub use lifetime_safety::{
    LifetimeBound, LifetimeBoundAlgorithm, LifetimeBoundResource, LifetimeBoundResourcePool,
    LifetimeBoundSequence, LifetimeBoundTensor, LifetimeValidator, ScopedResourceHelper,
};

// Dispatch safety exports
pub use dispatch_safety::{
    BatchDispatchManager, BatchDispatchResult, DispatchAnalysis, DispatchOptimizer, Param1, Param2,
    Param3, Param4, ParamCons, ParamNil, ParameterList, SafeDispatchBuilder, SafeDispatchResult,
    ShaderParameter, TypeIndexedAlgorithm,
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
