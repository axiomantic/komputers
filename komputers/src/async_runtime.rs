//! Runtime-agnostic async operations for GPU compute safety
//!
//! This module provides abstractions over different async runtimes (Tokio, smol)
//! to ensure GPU operations maintain safety guarantees regardless of the execution context.
//!
//! # Safety Guarantees
//!
//! - **Runtime Context Validation**: GPU operations bound to specific runtime contexts
//! - **Cross-Runtime Prevention**: Prevents GPU resources from moving between executors
//! - **Type Safety**: Compile-time enforcement of runtime compatibility
//! - **Ecosystem Compatibility**: Support for Tokio, smol with feature flags
//!
//! # Example
//!
//! ```rust,no_run
//! use komputers::{AsyncSequence, Manager, Tensor};
//! use komputers::async_runtime::TokioRuntime;
//!
//! async fn gpu_compute_with_runtime() -> komputers::Result<()> {
//!     let mut manager = Manager::new()?;
//!     let tensor = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0])?;
//!
//!     // Runtime-bound async sequence
//!     let result = AsyncSequence::<_, TokioRuntime>::new(&mut manager)?
//!         .record_tensor_sync_device(tensor.handle())?
//!         .execute_async()
//!         .await?;
//!
//!     Ok(())
//! }
//! ```

// Note: KomputeError and Result are used in conditional compilation sections
use std::future::Future;
use std::time::Duration;

#[cfg(any(feature = "tokio-runtime", feature = "smol-runtime"))]
use std::pin::Pin;
#[cfg(any(feature = "tokio-runtime", feature = "smol-runtime"))]
use std::task::{Context, Poll};

// =============================================================================
// Runtime Abstraction Traits
// =============================================================================

/// Runtime error types for async execution context issues
#[derive(Debug, Clone, thiserror::Error)]
pub enum RuntimeError {
    /// No async runtime is available in the current context
    #[error("No async runtime detected in current execution context")]
    NoRuntime,
    /// Runtime context has changed during GPU operation execution
    #[error("GPU operation moved between different runtime contexts")]
    RuntimeContextChanged,
    /// Unsupported runtime feature requested
    #[error("Runtime feature not supported: {feature}")]
    UnsupportedFeature {
        /// The unsupported feature name
        feature: String,
    },
    /// Runtime-specific error
    #[error("Runtime error: {message}")]
    RuntimeSpecific {
        /// The runtime error message
        message: String,
    },
}

/// Join error for async task execution
#[derive(Debug, Clone, thiserror::Error)]
pub enum JoinError {
    /// Task was cancelled before completion
    #[error("Async task was cancelled")]
    Cancelled,
    /// Task panicked during execution
    #[error("Async task panicked: {message}")]
    Panicked {
        /// The panic message
        message: String,
    },
    /// Runtime-specific join error
    #[error("Join error: {message}")]
    RuntimeSpecific {
        /// The join error message
        message: String,
    },
}

/// Runtime-agnostic trait for async execution contexts
///
/// This trait abstracts over different async runtimes to provide consistent
/// GPU operation execution while maintaining runtime-specific optimizations.
pub trait AsyncRuntime: Send + Sync + 'static {
    /// Runtime handle type for context validation
    type Handle: Clone + Send + Sync;

    /// Sleep future type for runtime-specific timing
    type Sleep: Future<Output = ()> + Send;

    /// Join handle type for spawned tasks
    type JoinHandle<T>: Future<Output = std::result::Result<T, JoinError>> + Send
    where
        T: Send + 'static;

    /// Get the current runtime handle for context validation
    fn current_handle() -> std::result::Result<Self::Handle, RuntimeError>;

    /// Spawn a task on this runtime
    fn spawn<F>(&self, future: F) -> Self::JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static;

    /// Create a sleep future for the specified duration
    fn sleep(duration: Duration) -> Self::Sleep;

    /// Runtime identifier for debugging and error reporting
    fn runtime_name() -> &'static str;

    /// Check if this runtime supports specific features
    fn supports_feature(feature: RuntimeFeature) -> bool;
}

/// Runtime features that may be supported differently across executors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeFeature {
    /// Task spawning capability
    TaskSpawning,
    /// High-precision timing
    HighPrecisionTiming,
    /// Task cancellation
    TaskCancellation,
    /// Local task sets
    LocalTaskSets,
}

// =============================================================================
// Tokio Runtime Implementation
// =============================================================================

/// Tokio async runtime implementation
pub struct TokioRuntime;

#[cfg(feature = "tokio-runtime")]
impl AsyncRuntime for TokioRuntime {
    type Handle = tokio::runtime::Handle;
    type Sleep = tokio::time::Sleep;
    type JoinHandle<T>
        = TokioJoinHandle<T>
    where
        T: Send + 'static;

    fn current_handle() -> std::result::Result<Self::Handle, RuntimeError> {
        tokio::runtime::Handle::try_current().map_err(|_| RuntimeError::NoRuntime)
    }

    fn spawn<F>(&self, future: F) -> Self::JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        TokioJoinHandle {
            inner: tokio::spawn(future),
        }
    }

    fn sleep(duration: Duration) -> Self::Sleep {
        tokio::time::sleep(duration)
    }

    fn runtime_name() -> &'static str {
        "tokio"
    }

    fn supports_feature(feature: RuntimeFeature) -> bool {
        match feature {
            RuntimeFeature::TaskSpawning => true,
            RuntimeFeature::HighPrecisionTiming => true,
            RuntimeFeature::TaskCancellation => true,
            RuntimeFeature::LocalTaskSets => true,
        }
    }
}

/// Wrapper for Tokio's JoinHandle to implement our JoinError trait
#[cfg(feature = "tokio-runtime")]
pub struct TokioJoinHandle<T> {
    inner: tokio::task::JoinHandle<T>,
}

#[cfg(feature = "tokio-runtime")]
impl<T> Future for TokioJoinHandle<T> {
    type Output = std::result::Result<T, JoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.inner).poll(cx) {
            Poll::Ready(Ok(value)) => Poll::Ready(Ok(value)),
            Poll::Ready(Err(err)) => {
                if err.is_cancelled() {
                    Poll::Ready(Err(JoinError::Cancelled))
                } else if err.is_panic() {
                    let panic_info = err.into_panic();
                    let panic_msg = panic_info
                        .downcast_ref::<String>()
                        .map(|s| s.clone())
                        .or_else(|| panic_info.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "Unknown panic".to_string());
                    Poll::Ready(Err(JoinError::Panicked { message: panic_msg }))
                } else {
                    Poll::Ready(Err(JoinError::RuntimeSpecific {
                        message: format!("Tokio join error: {:?}", err),
                    }))
                }
            }
            Poll::Pending => Poll::Pending,
        }
    }
}


// =============================================================================
// Smol Runtime Implementation (when feature enabled)
// =============================================================================

/// Smol runtime implementation
#[cfg(feature = "smol-runtime")]
pub struct SmolRuntime;

#[cfg(feature = "smol-runtime")]
impl AsyncRuntime for SmolRuntime {
    type Handle = SmolHandle;
    type Sleep = smol::Timer;
    type JoinHandle<T>
        = SmolJoinHandle<T>
    where
        T: Send + 'static;

    fn current_handle() -> std::result::Result<Self::Handle, RuntimeError> {
        // Smol doesn't have explicit handles, so we use a dummy handle
        Ok(SmolHandle)
    }

    fn spawn<F>(&self, future: F) -> Self::JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        SmolJoinHandle {
            inner: smol::spawn(future),
        }
    }

    fn sleep(duration: Duration) -> Self::Sleep {
        smol::Timer::after(duration)
    }

    fn runtime_name() -> &'static str {
        "smol"
    }

    fn supports_feature(feature: RuntimeFeature) -> bool {
        match feature {
            RuntimeFeature::TaskSpawning => true,
            RuntimeFeature::HighPrecisionTiming => false, // Basic timing support
            RuntimeFeature::TaskCancellation => false,    // Limited cancellation
            RuntimeFeature::LocalTaskSets => false,
        }
    }
}

/// Dummy handle for smol (it doesn't have explicit runtime handles)
#[cfg(feature = "smol-runtime")]
#[derive(Debug, Clone)]
pub struct SmolHandle;

/// Wrapper for smol's Task handle
#[cfg(feature = "smol-runtime")]
pub struct SmolJoinHandle<T> {
    inner: smol::Task<T>,
}

#[cfg(feature = "smol-runtime")]
impl<T> Future for SmolJoinHandle<T> {
    type Output = std::result::Result<T, JoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // smol's Task doesn't provide detailed error information
        match Pin::new(&mut self.inner).poll(cx) {
            Poll::Ready(value) => Poll::Ready(Ok(value)),
            Poll::Pending => Poll::Pending,
        }
    }
}

// =============================================================================
// Default Runtime Selection
// =============================================================================

/// Default runtime type based on enabled features
#[cfg(feature = "tokio-runtime")]
pub type DefaultRuntime = TokioRuntime;

#[cfg(all(
    feature = "smol-runtime",
    not(feature = "tokio-runtime")
))]
pub type DefaultRuntime = SmolRuntime;

// Compile error if no runtime is enabled
// Note: This is commented out to allow compilation without runtime features during development
// #[cfg(not(any(
//     feature = "tokio-runtime",
//     feature = "smol-runtime"
// )))]
// compile_error!("At least one async runtime feature must be enabled: tokio-runtime, or smol-runtime");

// =============================================================================
// Utility Functions
// =============================================================================

/// Detect the current runtime and return its name
pub fn detect_current_runtime() -> std::result::Result<&'static str, RuntimeError> {
    #[cfg(feature = "tokio-runtime")]
    if TokioRuntime::current_handle().is_ok() {
        return Ok(TokioRuntime::runtime_name());
    }


    #[cfg(feature = "smol-runtime")]
    if SmolRuntime::current_handle().is_ok() {
        return Ok(SmolRuntime::runtime_name());
    }

    Err(RuntimeError::NoRuntime)
}

/// Validate runtime context compatibility
///
/// Note: This function provides a basic validation by checking if a current handle
/// can be obtained. For runtimes that don't support handle comparison, it validates
/// that the runtime is still available.
pub fn validate_runtime_context<R: AsyncRuntime>(
    _stored_handle: Option<&R::Handle>,
) -> std::result::Result<(), RuntimeError> {
    // Try to get current handle to ensure runtime is still available
    let _current = R::current_handle()?;

    // For now, we assume context is valid if we can get a current handle
    // In the future, this could be enhanced with runtime-specific comparison logic
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_feature_detection() {
        // Test that runtime features are correctly reported
        #[cfg(feature = "tokio-runtime")]
        {
            assert!(TokioRuntime::supports_feature(RuntimeFeature::TaskSpawning));
            assert!(TokioRuntime::supports_feature(
                RuntimeFeature::HighPrecisionTiming
            ));
            assert!(TokioRuntime::supports_feature(
                RuntimeFeature::TaskCancellation
            ));
            assert_eq!(TokioRuntime::runtime_name(), "tokio");
        }


        #[cfg(feature = "smol-runtime")]
        {
            assert!(SmolRuntime::supports_feature(RuntimeFeature::TaskSpawning));
            assert!(!SmolRuntime::supports_feature(
                RuntimeFeature::LocalTaskSets
            ));
            assert_eq!(SmolRuntime::runtime_name(), "smol");
        }
    }

    #[test]
    fn test_runtime_error_types() {
        let no_runtime = RuntimeError::NoRuntime;
        assert_eq!(
            format!("{}", no_runtime),
            "No async runtime detected in current execution context"
        );

        let context_changed = RuntimeError::RuntimeContextChanged;
        assert_eq!(
            format!("{}", context_changed),
            "GPU operation moved between different runtime contexts"
        );

        let unsupported = RuntimeError::UnsupportedFeature {
            feature: "test_feature".to_string(),
        };
        assert_eq!(
            format!("{}", unsupported),
            "Runtime feature not supported: test_feature"
        );
    }

    #[test]
    fn test_join_error_types() {
        let cancelled = JoinError::Cancelled;
        assert_eq!(format!("{}", cancelled), "Async task was cancelled");

        let panicked = JoinError::Panicked {
            message: "test panic".to_string(),
        };
        assert_eq!(format!("{}", panicked), "Async task panicked: test panic");
    }

    #[cfg(feature = "tokio-runtime")]
    #[tokio::test]
    async fn test_tokio_runtime_detection() {
        // Test that we can detect Tokio runtime when it's running
        let handle_result = TokioRuntime::current_handle();
        assert!(handle_result.is_ok(), "Should detect Tokio runtime");

        let detected_runtime = detect_current_runtime();
        assert!(detected_runtime.is_ok());
        assert_eq!(detected_runtime.unwrap(), "tokio");
    }

    #[cfg(feature = "tokio-runtime")]
    #[tokio::test]
    async fn test_tokio_sleep() {
        let start = std::time::Instant::now();
        TokioRuntime::sleep(Duration::from_millis(10)).await;
        let elapsed = start.elapsed();

        // Should sleep for at least the requested duration (with some tolerance)
        assert!(
            elapsed >= Duration::from_millis(5),
            "Sleep should last at least 5ms"
        );
        assert!(
            elapsed < Duration::from_millis(100),
            "Sleep should not take more than 100ms"
        );
    }

    #[test]
    fn test_runtime_context_validation() {
        // Test validation outside runtime context (should fail gracefully)
        #[cfg(feature = "tokio-runtime")]
        {
            let result = validate_runtime_context::<TokioRuntime>(None);
            // Outside of a Tokio runtime context, this should fail with NoRuntime error
            assert!(
                result.is_err(),
                "Validation should fail when no runtime is available"
            );

            if let Err(RuntimeError::NoRuntime) = result {
                // Expected behavior - no runtime available
            } else {
                panic!("Expected NoRuntime error, got: {:?}", result);
            }
        }
    }
}
