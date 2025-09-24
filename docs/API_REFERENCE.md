# Komputers API Reference

Complete API documentation for the revolutionary type-safe GPU computing library. This reference covers all public APIs with examples, safety guarantees, and usage patterns.

## 📚 Table of Contents

1. [Core Components](#-core-components)
2. [Type-Safe Handle System](#️-type-safe-handle-system)
3. [Manager Types](#️-manager-types)
4. [Runtime Validation](#-runtime-validation)
5. [Async Operations](#-async-operations)
6. [Performance Profiling](#-performance-profiling)
7. [Workgroup Validation](#️-workgroup-validation)
8. [Error Types](#-error-types)
9. [Convenience APIs](#️-convenience-apis)
10. [Migration Utilities](#-migration-utilities)

---

## 🏗️ Core Components

### Manager

The central GPU resource manager with automatic Vulkan setup and type-safe resource tracking.

```rust
pub struct Manager {
    // Private implementation
}
```

#### Safety Guarantees

- **Memory Safety**: All GPU resources automatically cleaned up via RAII
- **Handle Tracking**: All created handles are tracked for validation
- **Vulkan Management**: Automatic device initialization and cleanup
- **Thread Safety**: Safe for concurrent access with proper synchronization

#### Construction

````rust
impl Manager {
    /// Create a new GPU manager with automatic Vulkan setup
    ///
    /// # Returns
    /// * `Result<Manager, KomputeError>` - New manager or initialization error
    ///
    /// # Example
    /// ```rust
    /// use komputers::Manager;
    ///
    /// let mut manager = Manager::new()?;
    /// println!("GPU device: {}", manager.device_name()?);
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn new() -> Result<Self, KomputeError>;

    /// Create manager with custom Vulkan configuration
    ///
    /// # Arguments
    /// * `config` - Custom Vulkan device configuration
    ///
    /// # Example
    /// ```rust
    /// use komputers::{Manager, VulkanConfig};
    ///
    /// let config = VulkanConfig::builder()
    ///     .prefer_dedicated_gpu(true)
    ///     .enable_validation(true)
    ///     .build();
    ///
    /// let mut manager = Manager::with_config(config)?;
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn with_config(config: VulkanConfig) -> Result<Self, KomputeError>;
}
````

#### Resource Creation

````rust
impl Manager {
    /// Create a type-safe sequence for recording GPU operations
    ///
    /// # Returns
    /// * `Result<Sequence<Recording>, KomputeError>` - New sequence in recording state
    ///
    /// # Example
    /// ```rust
    /// use komputers::Manager;
    ///
    /// let mut manager = Manager::new()?;
    /// let sequence = manager.sequence()?;
    ///
    /// // Use consuming builder pattern
    /// let result = sequence
    ///     .record_tensor_sync_device(handle)?
    ///     .execute()?;
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn sequence(&mut self) -> Result<Sequence<Recording>, KomputeError>;

    /// Get GPU device information
    ///
    /// # Returns
    /// * `Result<String, KomputeError>` - GPU device name
    ///
    /// # Example
    /// ```rust
    /// let mut manager = Manager::new()?;
    /// println!("Using GPU: {}", manager.device_name()?);
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn device_name(&self) -> Result<String, KomputeError>;

    /// Get manager's unique runtime ID for validation
    ///
    /// # Returns
    /// * `u64` - Unique manager instance identifier
    ///
    /// # Example
    /// ```rust
    /// let manager1 = Manager::new()?;
    /// let manager2 = Manager::new()?;
    /// assert_ne!(manager1.manager_id(), manager2.manager_id());
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn manager_id(&self) -> u64;
}
````

### Tensor<T>

Type-safe GPU tensor with automatic memory management and data type validation.

```rust
pub struct Tensor<T: TensorData> {
    // Private implementation
}
```

#### Safety Guarantees

- **Type Safety**: Generic over data types with compile-time validation
- **Memory Safety**: Automatic GPU memory allocation and cleanup
- **Handle Validation**: All operations validated against creating manager
- **Data Integrity**: Host-device synchronization managed automatically

#### Construction

````rust
impl<T: TensorData> Tensor<T> {
    /// Create tensor from host data with automatic GPU upload
    ///
    /// # Arguments
    /// * `manager` - GPU manager for resource allocation
    /// * `data` - Host data to upload to GPU
    ///
    /// # Returns
    /// * `Result<Tensor<T>, KomputeError>` - New GPU tensor or error
    ///
    /// # Example
    /// ```rust
    /// use komputers::{Manager, Tensor};
    ///
    /// let mut manager = Manager::new()?;
    ///
    /// // Create f32 tensor
    /// let tensor_f32 = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0])?;
    ///
    /// // Create i32 tensor
    /// let tensor_i32 = Tensor::new(&mut manager, &[1i32, 2, 3])?;
    ///
    /// assert_eq!(tensor_f32.size(), 3);
    /// assert_eq!(tensor_i32.size(), 3);
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn new(manager: &mut Manager, data: &[T]) -> Result<Self, KomputeError>;

    /// Create uninitialized tensor with specified size
    ///
    /// # Arguments
    /// * `manager` - GPU manager for resource allocation
    /// * `size` - Number of elements to allocate
    ///
    /// # Returns
    /// * `Result<Tensor<T>, KomputeError>` - Uninitialized GPU tensor
    ///
    /// # Example
    /// ```rust
    /// use komputers::{Manager, Tensor};
    ///
    /// let mut manager = Manager::new()?;
    /// let mut output: Tensor<f32> = Tensor::uninit(&mut manager, 1000)?;
    ///
    /// // Use as output tensor for GPU computation
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn uninit(manager: &mut Manager, size: usize) -> Result<Self, KomputeError>;
}
````

#### Data Access

````rust
impl<T: TensorData> Tensor<T> {
    /// Get tensor data with automatic host synchronization
    ///
    /// # Returns
    /// * `Result<Vec<T>, KomputeError>` - Host copy of tensor data
    ///
    /// # Example
    /// ```rust
    /// use komputers::{Manager, Tensor};
    ///
    /// let mut manager = Manager::new()?;
    /// let tensor = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0])?;
    ///
    /// // Sync from GPU and get data
    /// let data = tensor.data()?;
    /// assert_eq!(data, vec![1.0, 2.0, 3.0]);
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn data(&self) -> Result<Vec<T>, KomputeError>;

    /// Get tensor size (number of elements)
    ///
    /// # Returns
    /// * `usize` - Number of elements in tensor
    ///
    /// # Example
    /// ```rust
    /// let tensor = Tensor::new(&mut manager, &[1.0f32; 100])?;
    /// assert_eq!(tensor.size(), 100);
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn size(&self) -> usize;

    /// Get type-safe handle for operations
    ///
    /// # Returns
    /// * `u64` - Handle value for sequence operations
    ///
    /// # Example
    /// ```rust
    /// let tensor = Tensor::new(&mut manager, &[1.0f32, 2.0])?;
    /// let handle = tensor.handle();
    ///
    /// // Use in type-safe sequence operations
    /// manager.sequence()?
    ///     .record_tensor_sync_device(handle)?
    ///     .execute()?;
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn handle(&self) -> u64;
}
````

### Sequence<State>

Type-state pattern implementation for safe GPU operation recording and execution.

```rust
pub struct Sequence<State = Recording> {
    // Private implementation
}

// Type states
pub struct Recording;
pub struct Executed;
```

#### Safety Guarantees

- **Type-State Safety**: Consuming builder pattern prevents reuse and ensures execution
- **Operation Validation**: All recorded operations validated at record time
- **Resource Tracking**: Cross-manager handle usage prevented
- **Execution Safety**: Empty sequences cannot be executed

#### Recording Operations (Consuming Builder Pattern)

````rust
impl Sequence<Recording> {
    /// Record tensor device synchronization (consuming)
    ///
    /// # Arguments
    /// * `tensor_handle` - Handle of tensor to synchronize to device
    ///
    /// # Returns
    /// * `Result<Self, KomputeError>` - Self for chaining or error
    ///
    /// # Safety
    /// This method validates that the tensor handle belongs to the same
    /// manager that created this sequence, preventing cross-manager usage.
    ///
    /// # Example
    /// ```rust
    /// use komputers::{Manager, Tensor};
    ///
    /// let mut manager = Manager::new()?;
    /// let tensor = Tensor::new(&mut manager, &[1.0f32, 2.0])?;
    ///
    /// let result = manager.sequence()?
    ///     .record_tensor_sync_device(tensor.handle())?  // Upload to GPU
    ///     .execute()?;
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn record_tensor_sync_device(mut self, tensor_handle: u64) -> Result<Self, KomputeError>;

    /// Record tensor local synchronization (consuming)
    ///
    /// # Arguments
    /// * `tensor_handle` - Handle of tensor to synchronize to host
    ///
    /// # Returns
    /// * `Result<Self, KomputeError>` - Self for chaining or error
    ///
    /// # Example
    /// ```rust
    /// let result = manager.sequence()?
    ///     .record_tensor_sync_device(tensor.handle())?  // Upload to GPU
    ///     .record_tensor_sync_local(tensor.handle())?   // Download from GPU
    ///     .execute()?;
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn record_tensor_sync_local(mut self, tensor_handle: u64) -> Result<Self, KomputeError>;

    /// Record algorithm dispatch operation (consuming)
    ///
    /// # Arguments
    /// * `algorithm_handle` - Handle of algorithm to dispatch
    ///
    /// # Returns
    /// * `Result<Self, KomputeError>` - Self for chaining or error
    ///
    /// # Example
    /// ```rust
    /// use komputers::{Manager, Algorithm};
    ///
    /// let mut manager = Manager::new()?;
    /// let algorithm = Algorithm::from_spirv(&mut manager, spirv_data, tensors)?;
    ///
    /// let result = manager.sequence()?
    ///     .record_algorithm_dispatch(algorithm.handle())?
    ///     .execute()?;
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn record_algorithm_dispatch(mut self, algorithm_handle: u64) -> Result<Self, KomputeError>;

    /// Execute the recorded sequence (consuming)
    ///
    /// # Returns
    /// * `Result<SequenceResult, KomputeError>` - Execution result or error
    ///
    /// # Safety
    /// This method consumes the sequence, preventing reuse. Empty sequences
    /// (no operations recorded) will return an error.
    ///
    /// # Example
    /// ```rust
    /// let result = manager.sequence()?
    ///     .record_tensor_sync_device(tensor.handle())?
    ///     .execute()?;  // Consumes sequence - cannot reuse
    ///
    /// println!("Executed {} operations in {:?}",
    ///          result.operation_count, result.execution_time);
    /// # Ok::<(), komputers::KomputeError>(())
    /// ```
    pub fn execute(self) -> Result<SequenceResult, KomputeError>;
}
````

#### Execution Results

```rust
/// Result of executing a GPU sequence
#[derive(Debug, Clone)]
pub struct SequenceResult {
    /// Number of operations executed
    pub operation_count: usize,
    /// Total execution time on GPU
    pub execution_time: std::time::Duration,
}
```

---

## 🛡️ Type-Safe Handle System

The handle system provides compile-time safety through phantom types that encode object type, synchronization mode, and manager type information.

### Handle<ObjectType, SyncMode, ManagerType>

```rust
/// Type-safe handle with phantom type markers
pub struct Handle<ObjectType, SyncMode, ManagerType> {
    pub(crate) value: u64,
    _phantom: PhantomData<(ObjectType, SyncMode, ManagerType)>,
}
```

#### Type Markers

```rust
// Object type markers
pub struct TensorType;
pub struct AlgorithmType;
pub struct SequenceType;
pub struct QueueType;

// Synchronization mode markers
pub struct AsyncMarker;
pub struct SyncMarker;

// Manager type markers - see Manager Types section
```

#### Type Aliases for Common Patterns

```rust
/// Type-safe tensor handles
pub type SyncTensorHandle<ManagerType> = Handle<TensorType, SyncMarker, ManagerType>;
pub type AsyncTensorHandle<ManagerType> = Handle<TensorType, AsyncMarker, ManagerType>;

/// Type-safe algorithm handles
pub type SyncAlgorithmHandle<ManagerType> = Handle<AlgorithmType, SyncMarker, ManagerType>;
pub type AsyncAlgorithmHandle<ManagerType> = Handle<AlgorithmType, AsyncMarker, ManagerType>;

/// Type-safe sequence handles
pub type SyncSequenceHandle<ManagerType> = Handle<SequenceType, SyncMarker, ManagerType>;
pub type AsyncSequenceHandle<ManagerType> = Handle<SequenceType, AsyncMarker, ManagerType>;

/// Type-safe queue handles
pub type SyncQueueHandle<ManagerType> = Handle<QueueType, SyncMarker, ManagerType>;
pub type AsyncQueueHandle<ManagerType> = Handle<QueueType, AsyncMarker, ManagerType>;
```

#### Handle Operations

````rust
impl<ObjectType, SyncMode, ManagerType> Handle<ObjectType, SyncMode, ManagerType> {
    /// Create new typed handle from raw value
    ///
    /// # Arguments
    /// * `value` - Raw handle value from C++ layer
    ///
    /// # Returns
    /// * `Handle<ObjectType, SyncMode, ManagerType>` - Typed handle
    ///
    /// # Safety
    /// This method should only be used internally. The type parameters
    /// must correctly reflect the actual object type and manager.
    ///
    /// # Example
    /// ```rust
    /// // Internal usage only
    /// let handle: SyncTensorHandle<DefaultManagerId> = Handle::new(raw_value);
    /// ```
    pub fn new(value: u64) -> Self;

    /// Get raw handle value for C++ interop
    ///
    /// # Returns
    /// * `u64` - Raw handle value
    ///
    /// # Example
    /// ```rust
    /// let handle: SyncTensorHandle<_> = tensor.handle();
    /// let raw_value = handle.raw_value();
    /// ```
    pub fn raw_value(&self) -> u64;

    /// Convert to untyped handle
    ///
    /// # Returns
    /// * `UntypedHandle` - Handle without type information
    ///
    /// # Example
    /// ```rust
    /// let typed_handle: SyncTensorHandle<_> = tensor.handle();
    /// let untyped = typed_handle.to_untyped();
    /// ```
    pub fn to_untyped(self) -> UntypedHandle;
}
````

#### Compatibility Checking

```rust
/// Trait for checking handle compatibility at compile time
pub trait CompatibleHandles<Other> {
    /// Compile-time compatibility check
    const COMPATIBLE: bool;
}

// Implementations ensure only compatible handles can be used together
impl<ObjectType, ManagerType> CompatibleHandles<Handle<ObjectType, SyncMarker, ManagerType>>
    for Handle<ObjectType, SyncMarker, ManagerType>
{
    const COMPATIBLE: bool = true;
}

// Cross-mode compatibility is forbidden
impl<ObjectType, ManagerType> CompatibleHandles<Handle<ObjectType, AsyncMarker, ManagerType>>
    for Handle<ObjectType, SyncMarker, ManagerType>
{
    const COMPATIBLE: bool = false;
}
```

#### Usage Example

```rust
use komputers::*;

fn type_safe_handles_example() -> Result<()> {
    let mut manager = Manager::new()?;
    let tensor = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0])?;

    // ✅ Type-safe handle usage
    let handle: u64 = tensor.handle();  // Gets raw value

    manager.sequence()?
        .record_tensor_sync_device(handle)?  // Type-safe operation
        .execute()?;

    // ❌ This cannot compile - different manager types
    // let other_manager = Manager::new()?;
    // other_manager.sequence()?
    //     .record_tensor_sync_device(handle)?;  // COMPILE ERROR!

    Ok(())
}
```

---

## 🏷️ Manager Types

The manager type system provides compile-time differentiation between different manager instances and synchronization modes.

### Manager ID System

```rust
/// Compile-time manager identifier
pub struct ManagerId<const ID: u64>;

/// Runtime manager identifier for validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeManagerId {
    id: u64,
}

/// Default manager ID for simple usage
pub type DefaultManagerId = ManagerId<0>;
```

#### Manager Type Markers

```rust
/// Synchronous manager type marker
pub struct SyncManagerType<Id>(PhantomData<Id>);

/// Asynchronous manager type marker
pub struct AsyncManagerType<Id>(PhantomData<Id>);

/// Type aliases for common patterns
pub type DefaultSyncManager = SyncManagerType<DefaultManagerId>;
pub type DefaultAsyncManager = AsyncManagerType<DefaultManagerId>;
```

#### HasManagerId Trait

```rust
/// Trait for types that have a manager ID
pub trait HasManagerId {
    /// Get the runtime manager ID
    fn manager_id(&self) -> u64;
}

impl HasManagerId for Manager {
    fn manager_id(&self) -> u64 {
        self.validation_instance.id()
    }
}
```

#### Type Safety Errors

```rust
/// Errors related to type safety violations
#[derive(Debug, Error)]
pub enum TypeSafetyError {
    /// Async/sync mode mismatch
    #[error("Cannot use {async_type} with {sync_type}")]
    AsyncSyncMismatch {
        async_type: &'static str,
        sync_type: &'static str,
    },

    /// Object type mismatch
    #[error("Cannot use {actual_type} in {expected_type} operation")]
    ObjectTypeMismatch {
        actual_type: &'static str,
        expected_type: &'static str,
    },

    /// Manager type mismatch
    #[error("Cannot use handle from manager type {source_manager} with manager type {target_manager}")]
    ManagerTypeMismatch {
        source_manager: &'static str,
        target_manager: &'static str,
    },

    /// Manager instance mismatch (runtime validation)
    #[error("Cannot use handle from manager instance {source_id} with manager instance {target_id}")]
    ManagerInstanceMismatch {
        source_id: u64,
        target_id: u64,
    },

    /// Invalid handle
    #[error("Handle {handle_id} is invalid or from destroyed manager")]
    InvalidHandle {
        handle_id: u64,
    },
}
```

#### Usage Example

```rust
use komputers::*;

fn manager_type_safety() -> Result<()> {
    // Different manager types have different compile-time IDs
    let manager1: Manager = Manager::new()?;
    let manager2: Manager = Manager::new()?;

    // Same type at compile time, but different instances at runtime
    assert_ne!(manager1.manager_id(), manager2.manager_id());

    let tensor1 = Tensor::new(&mut manager1, &[1.0f32])?;

    // ❌ Runtime validation will catch this
    // manager2.sequence()?
    //     .record_tensor_sync_device(tensor1.handle())?  // Runtime error!
    //     .execute()?;

    Ok(())
}
```

---

## 🔍 Runtime Validation

The runtime validation system provides efficient validation for scenarios that cannot be caught at compile time, such as cross-manager instance usage.

### Configuration

````rust
/// Enable or disable runtime validation globally
///
/// # Arguments
/// * `enabled` - Whether to enable validation
///
/// # Performance Impact
/// When enabled, adds <5% overhead to operations. When disabled,
/// validation functions return immediately with no overhead.
///
/// # Example
/// ```rust
/// use komputers::runtime_validation::set_validation_enabled;
///
/// // Enable for development/testing
/// set_validation_enabled(true);
///
/// // Disable for production if needed
/// set_validation_enabled(false);
/// ```
pub fn set_validation_enabled(enabled: bool);

/// Check if runtime validation is currently enabled
///
/// # Returns
/// * `bool` - True if validation is enabled
///
/// # Example
/// ```rust
/// use komputers::runtime_validation::is_validation_enabled;
///
/// if is_validation_enabled() {
///     println!("Runtime validation is active");
/// }
/// ```
pub fn is_validation_enabled() -> bool;
````

### Validation Functions

````rust
/// Validate that a handle belongs to a specific manager
///
/// # Arguments
/// * `handle` - Handle to validate
/// * `manager` - Manager that should own the handle
///
/// # Returns
/// * `Result<(), KomputeError>` - Ok if valid, error if invalid
///
/// # Performance
/// Typical validation time: 10-50 nanoseconds
///
/// # Example
/// ```rust
/// use komputers::runtime_validation::validate_handle_with_manager;
///
/// let manager = Manager::new()?;
/// let tensor = Tensor::new(&mut manager, &[1.0f32])?;
///
/// // This will succeed
/// validate_handle_with_manager(tensor.handle(), &manager)?;
///
/// let other_manager = Manager::new()?;
/// // This will fail with descriptive error
/// let result = validate_handle_with_manager(tensor.handle(), &other_manager);
/// assert!(result.is_err());
/// # Ok::<(), komputers::KomputeError>(())
/// ```
pub fn validate_handle_with_manager(handle: u64, manager: &Manager) -> Result<(), KomputeError>;

/// Validate that multiple handles belong to the same manager
///
/// # Arguments
/// * `handles` - Slice of handles to validate
///
/// # Returns
/// * `Result<(), KomputeError>` - Ok if all from same manager, error otherwise
///
/// # Example
/// ```rust
/// use komputers::runtime_validation::validate_handles_same_manager;
///
/// let mut manager = Manager::new()?;
/// let tensor1 = Tensor::new(&mut manager, &[1.0f32])?;
/// let tensor2 = Tensor::new(&mut manager, &[2.0f32])?;
///
/// // This will succeed - same manager
/// validate_handles_same_manager(&[tensor1.handle(), tensor2.handle()])?;
/// # Ok::<(), komputers::KomputeError>(())
/// ```
pub fn validate_handles_same_manager(handles: &[u64]) -> Result<(), KomputeError>;
````

### Performance Monitoring

````rust
/// Get validation performance metrics
///
/// # Returns
/// * `Option<String>` - Formatted metrics string or None if disabled
///
/// # Example
/// ```rust
/// use komputers::runtime_validation::get_validation_metrics;
///
/// if let Some(metrics) = get_validation_metrics() {
///     println!("Validation metrics:\n{}", metrics);
///     // Output:
///     // Total validations: 1,234
///     // Average validation time: 25ns
///     // Cache hit rate: 94%
///     // Total overhead: 2.3%
/// }
/// ```
pub fn get_validation_metrics() -> Option<String>;

/// Check if validation overhead is within acceptable limits
///
/// # Arguments
/// * `total_operation_time` - Total time for operations being validated
///
/// # Returns
/// * `bool` - True if overhead is <5% of total time
///
/// # Example
/// ```rust
/// use komputers::runtime_validation::is_validation_overhead_acceptable;
/// use std::time::Duration;
///
/// let operation_time = Duration::from_millis(100);
/// if !is_validation_overhead_acceptable(operation_time) {
///     println!("⚠️ Validation overhead is high - consider disabling");
/// }
/// ```
pub fn is_validation_overhead_acceptable(total_operation_time: Duration) -> bool;
````

### Manager Instance Tracking

````rust
/// Register a handle with a manager for validation
///
/// # Arguments
/// * `handle` - Handle to register
/// * `manager` - Manager that owns the handle
///
/// # Example
/// ```rust
/// // Internal usage only - automatically called by constructors
/// use komputers::runtime_validation::register_handle_with_manager;
///
/// register_handle_with_manager(handle_value, &manager);
/// ```
pub fn register_handle_with_manager(handle: u64, manager: &Manager);

/// Register multiple handles in batch for efficiency
///
/// # Arguments
/// * `handles` - Handles to register
/// * `manager` - Manager that owns the handles
///
/// # Example
/// ```rust
/// // Internal usage - batch registration for performance
/// use komputers::runtime_validation::register_handles_batch;
///
/// register_handles_batch(&[handle1, handle2, handle3], &manager);
/// ```
pub fn register_handles_batch(handles: &[u64], manager: &Manager);
````

### Usage Example

```rust
use komputers::*;
use komputers::runtime_validation::*;

fn runtime_validation_example() -> Result<()> {
    // Enable validation for development
    set_validation_enabled(true);

    let mut manager1 = Manager::new()?;
    let mut manager2 = Manager::new()?;

    let tensor1 = Tensor::new(&mut manager1, &[1.0f32, 2.0])?;
    let tensor2 = Tensor::new(&mut manager2, &[3.0f32, 4.0])?;

    // ✅ Same manager validation succeeds
    validate_handle_with_manager(tensor1.handle(), &manager1)?;

    // ❌ Cross-manager validation fails with helpful error
    match validate_handle_with_manager(tensor1.handle(), &manager2) {
        Err(KomputeError::TypeSafety(TypeSafetyError::ManagerInstanceMismatch {
            source_id, target_id
        })) => {
            println!("Caught cross-manager usage: {} vs {}", source_id, target_id);
        },
        _ => panic!("Expected manager mismatch error"),
    }

    // Check performance overhead
    if let Some(metrics) = get_validation_metrics() {
        println!("Validation performance:\n{}", metrics);
    }

    Ok(())
}
```

---

## ⚡ Async Operations

Type-safe asynchronous GPU operations with tokio integration and cancellation safety.

### AsyncSequence<State>

```rust
/// Asynchronous sequence with type-state safety
pub struct AsyncSequence<State = Recording> {
    // Private implementation
}
```

#### Safety Guarantees

- **Cancellation Safety**: Operations can be safely cancelled at await points
- **Type Safety**: Same phantom type system as synchronous operations
- **Resource Management**: Automatic cleanup even if futures are dropped
- **Performance Monitoring**: Built-in async performance metrics

#### Construction and Recording

````rust
impl AsyncSequence<Recording> {
    /// Record tensor sync operation asynchronously (consuming)
    ///
    /// # Arguments
    /// * `tensor_handle` - Handle of tensor to synchronize
    ///
    /// # Returns
    /// * `Result<Self, KomputeError>` - Self for chaining or error
    ///
    /// # Example
    /// ```rust
    /// use komputers::AsyncSequence;
    ///
    /// async fn async_operations() -> Result<()> {
    ///     let mut manager = AsyncManager::new()?;
    ///     let tensor = manager.create_tensor(&[1.0f32, 2.0]).await?;
    ///
    ///     let result = AsyncSequence::new(&mut manager)?
    ///         .record_tensor_sync_device(tensor.handle())?
    ///         .execute_async()
    ///         .await?;
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn record_tensor_sync_device(mut self, tensor_handle: u64) -> Result<Self, KomputeError>;

    /// Record local sync operation asynchronously (consuming)
    pub fn record_tensor_sync_local(mut self, tensor_handle: u64) -> Result<Self, KomputeError>;

    /// Record algorithm dispatch asynchronously (consuming)
    pub fn record_algorithm_dispatch(mut self, algorithm_handle: u64) -> Result<Self, KomputeError>;

    /// Execute the sequence asynchronously (consuming)
    ///
    /// # Returns
    /// * `AsyncSequenceExecution` - Future that resolves to execution result
    ///
    /// # Cancellation Safety
    /// This future is cancellation-safe. If cancelled, GPU resources are
    /// properly cleaned up and no operations are left in undefined states.
    ///
    /// # Example
    /// ```rust
    /// use tokio::time::{timeout, Duration};
    ///
    /// async fn with_timeout() -> Result<()> {
    ///     let sequence = AsyncSequence::new(&mut manager)?
    ///         .record_tensor_sync_device(handle)?;
    ///
    ///     // Execute with timeout
    ///     let result = timeout(
    ///         Duration::from_secs(5),
    ///         sequence.execute_async()
    ///     ).await??;
    ///
    ///     println!("Completed in {:?}", result.execution_time);
    ///     Ok(())
    /// }
    /// ```
    pub fn execute_async(self) -> AsyncSequenceExecution;
}
````

#### AsyncSequenceExecution Future

```rust
/// Future representing async sequence execution
pub struct AsyncSequenceExecution {
    // Private implementation
}

impl Future for AsyncSequenceExecution {
    type Output = Result<AsyncSequenceResult, KomputeError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}
```

#### Async Results

```rust
/// Result of asynchronous sequence execution
#[derive(Debug, Clone)]
pub struct AsyncSequenceResult {
    /// Number of operations executed
    pub operation_count: usize,
    /// Total execution time (including async overhead)
    pub execution_time: Duration,
    /// Async-specific performance metrics
    pub metrics: AsyncPerformanceMetrics,
}

/// Async-specific performance metrics
#[derive(Debug, Clone)]
pub struct AsyncPerformanceMetrics {
    /// Number of polls before completion
    pub poll_count: u32,
    /// Average time per poll
    pub avg_poll_time: Duration,
    /// Maximum poll time observed
    pub max_poll_time: Duration,
    /// Total async overhead
    pub async_overhead: Duration,
}
```

### Async Utilities

````rust
/// Utility functions for async GPU operations
pub mod async_utils {
    use super::*;
    use std::future::Future;
    use tokio::time::Duration;

    /// Execute multiple sequences concurrently
    ///
    /// # Arguments
    /// * `sequences` - Iterator of async sequences to execute
    /// * `timeout_duration` - Maximum time to wait for all sequences
    ///
    /// # Returns
    /// * `Result<Vec<Result<AsyncSequenceResult>>, KomputeError>` - Results or timeout error
    ///
    /// # Example
    /// ```rust
    /// use komputers::async_utils::execute_concurrent;
    /// use tokio::time::Duration;
    ///
    /// async fn concurrent_example() -> Result<()> {
    ///     let sequences = vec![
    ///         sequence1.execute_async(),
    ///         sequence2.execute_async(),
    ///         sequence3.execute_async(),
    ///     ];
    ///
    ///     let results = execute_concurrent(
    ///         sequences.into_iter(),
    ///         Duration::from_secs(10)
    ///     ).await?;
    ///
    ///     for result in results {
    ///         match result {
    ///             Ok(res) => println!("Success: {:?}", res.execution_time),
    ///             Err(e) => eprintln!("Error: {}", e),
    ///         }
    ///     }
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn execute_concurrent<I>(
        sequences: I,
        timeout_duration: Duration,
    ) -> Result<Vec<Result<AsyncSequenceResult>>, KomputeError>
    where
        I: Iterator<Item = AsyncSequenceExecution>;

    /// Create optimized async sequence with performance hints
    ///
    /// # Arguments
    /// * `manager` - Async manager instance
    ///
    /// # Returns
    /// * `Result<AsyncSequence<Recording>, KomputeError>` - Optimized sequence
    ///
    /// # Example
    /// ```rust
    /// let optimized = async_utils::create_optimized_async_sequence(&mut manager)?;
    /// ```
    pub fn create_optimized_async_sequence(
        manager: &mut AsyncManager
    ) -> Result<AsyncSequence<Recording>, KomputeError>;
}
````

### Complete Async Example

```rust
use komputers::*;
use tokio::time::{timeout, Duration};

async fn comprehensive_async_example() -> Result<()> {
    // Create async-capable manager
    let mut manager = AsyncManager::new()?;

    // Create tensors asynchronously
    let tensor_a = manager.create_tensor(&[1.0f32, 2.0, 3.0]).await?;
    let tensor_b = manager.create_tensor(&[4.0f32, 5.0, 6.0]).await?;
    let mut tensor_out = manager.create_tensor_uninit::<f32>(3).await?;

    // Execute multiple operations concurrently
    let operation1 = AsyncSequence::new(&mut manager)?
        .record_tensor_sync_device(tensor_a.handle())?
        .execute_async();

    let operation2 = AsyncSequence::new(&mut manager)?
        .record_tensor_sync_device(tensor_b.handle())?
        .execute_async();

    // Wait for both with timeout
    let (result1, result2) = timeout(
        Duration::from_secs(5),
        tokio::try_join!(operation1, operation2)
    ).await??;

    println!("Operation 1: {} ops in {:?}", result1.operation_count, result1.execution_time);
    println!("Operation 2: {} ops in {:?}", result2.operation_count, result2.execution_time);

    // Advanced: Use concurrent execution utility
    let sequences = vec![
        AsyncSequence::new(&mut manager)?.record_tensor_sync_local(tensor_out.handle())?.execute_async(),
        // ... more sequences
    ];

    let results = async_utils::execute_concurrent(
        sequences.into_iter(),
        Duration::from_secs(10)
    ).await?;

    // Process results
    for (i, result) in results.into_iter().enumerate() {
        match result {
            Ok(res) => {
                println!("Sequence {}: Success - {:?}", i, res.metrics);
            },
            Err(e) => {
                eprintln!("Sequence {}: Failed - {}", i, e);
            }
        }
    }

    Ok(())
}

// Run the async example
#[tokio::main]
async fn main() -> Result<()> {
    comprehensive_async_example().await
}
```

---

_[This is Part 1 of the API Reference. The complete documentation continues with Performance Profiling, Workgroup Validation, Error Types, and more components...]_

**Next sections to be documented:**

- 📊 Performance Profiling (performance_metrics.rs)
- 🏗️ Workgroup Validation (workgroup_validation.rs)
- ❌ Error Types (Complete error hierarchy)
- 🛠️ Convenience APIs (High-level operations)
- 🔄 Migration Utilities (Upgrading helpers)

Continue reading the full API reference to master type-safe GPU computing with Komputers!
