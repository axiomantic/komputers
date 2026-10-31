//! Type-safe manager implementations with compile-time validation
//!
//! This module provides type-safe wrappers around the core Manager that enforce
//! compile-time safety for async/sync operations and cross-manager validation.
//!
//! # Type Safety Guarantees
//!
//! - **Manager Type Safety**: Different manager types cannot be mixed
//! - **Async/Sync Safety**: Async managers only create async objects
//! - **Handle Compatibility**: Handles are only compatible within the same manager type
//! - **Zero Runtime Cost**: All type safety is compile-time only
//!
//! # Examples
//!
//! ```rust,no_run
//! use komputers::{create_sync_manager, create_async_manager, Result};
//!
//! async fn example() -> Result<()> {
//!     // Create type-safe managers
//!     let sync_manager = create_sync_manager();
//!     let async_manager = create_async_manager();
//!
//!     // Type-safe object creation
//!     let sync_tensor = sync_manager.create_tensor(&[1.0f32, 2.0, 3.0])?;
//!     let async_tensor = async_manager.create_tensor(&[4.0f32, 5.0, 6.0]).await?;
//!
//!     // Type-safe operations
//!     let sync_sequence = sync_manager.create_sequence()?;
//!     sync_sequence.record_tensor_sync_device(sync_tensor.handle())?; // ✅ Compiles
//!
//!     let async_sequence = async_manager.create_sequence().await?;
//!     async_sequence.record_tensor_sync_device(async_tensor.handle()).await?; // ✅ Compiles
//!
//!     // These would be compile errors:
//!     // sync_sequence.record_tensor_sync_device(async_tensor.handle())?; // ❌ Async/sync mismatch
//!     // async_sequence.record_tensor_sync_device(sync_tensor.handle()).await?; // ❌ Sync/async mismatch
//!
//!     Ok(())
//! }
//! ```

use crate::algorithm::Algorithm;
use crate::error::Result;
use crate::handle::{
    AsyncAlgorithmHandle, AsyncSequenceHandle, AsyncTensorHandle, SyncAlgorithmHandle,
    SyncSequenceHandle, SyncTensorHandle,
};
use crate::manager::Manager;
use crate::sequence::{Sequence, SequenceResult};
use crate::tensor::{Tensor, TensorData};
use crate::types::{AsyncManagerType, RuntimeManagerId, SyncManagerType};
use crate::VulkanConfig;
use std::marker::PhantomData;

// =============================================================================
// Type-Safe Manager Implementations
// =============================================================================

/// Type-safe synchronous manager
///
/// This manager ensures that all created objects are synchronous and
/// compatible with each other through compile-time type checking.
///
/// # Type Parameters
///
/// * `Id` - Unique identifier for this manager type (prevents cross-manager usage)
pub struct SyncManager<Id = RuntimeManagerId> {
    /// The underlying manager implementation
    inner: Manager,
    /// Unique identifier for this manager instance
    id: Id,
    /// Phantom data for compile-time type checking
    _phantom: PhantomData<SyncManagerType<Id>>,
}

impl<Id> SyncManager<Id>
where
    Id: Clone,
{
    /// Create a new sync manager with explicit configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Vulkan configuration
    /// * `id` - Unique identifier for this manager type
    ///
    /// # Returns
    ///
    /// A new type-safe sync manager
    pub fn with_config_and_id(config: VulkanConfig, id: Id) -> Result<Self> {
        let inner = Manager::with_config(config)?;
        Ok(Self {
            inner,
            id,
            _phantom: PhantomData,
        })
    }

    /// Create a new sync manager with default configuration
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this manager type
    pub fn with_id(id: Id) -> Result<Self> {
        let inner = Manager::new()?;
        Ok(Self {
            inner,
            id,
            _phantom: PhantomData,
        })
    }

    /// Get the manager's unique identifier
    pub fn id(&self) -> &Id {
        &self.id
    }

    /// Access the underlying manager for legacy operations
    pub fn inner(&self) -> &Manager {
        &self.inner
    }

    /// Access the underlying manager mutably for legacy operations
    pub fn inner_mut(&mut self) -> &mut Manager {
        &mut self.inner
    }
}

impl SyncManager<RuntimeManagerId> {
    /// Create a new sync manager with auto-generated ID
    pub fn new() -> Result<Self> {
        Self::with_id(RuntimeManagerId::new())
    }

    /// Create a new sync manager with explicit configuration and auto-generated ID
    pub fn with_config(config: VulkanConfig) -> Result<Self> {
        Self::with_config_and_id(config, RuntimeManagerId::new())
    }
}

/// Type-safe asynchronous manager
///
/// This manager ensures that all created objects are asynchronous and
/// compatible with each other through compile-time type checking.
///
/// # Type Parameters
///
/// * `Id` - Unique identifier for this manager type (prevents cross-manager usage)
pub struct AsyncManager<Id = RuntimeManagerId> {
    /// The underlying manager implementation
    inner: Manager,
    /// Unique identifier for this manager instance
    id: Id,
    /// Phantom data for compile-time type checking
    _phantom: PhantomData<AsyncManagerType<Id>>,
}

impl<Id> AsyncManager<Id>
where
    Id: Clone,
{
    /// Create a new async manager with explicit configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Vulkan configuration
    /// * `id` - Unique identifier for this manager type
    pub fn with_config_and_id(config: VulkanConfig, id: Id) -> Result<Self> {
        let inner = Manager::with_config(config)?;
        Ok(Self {
            inner,
            id,
            _phantom: PhantomData,
        })
    }

    /// Create a new async manager with default configuration
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this manager type
    pub fn with_id(id: Id) -> Result<Self> {
        let inner = Manager::new()?;
        Ok(Self {
            inner,
            id,
            _phantom: PhantomData,
        })
    }

    /// Get the manager's unique identifier
    pub fn id(&self) -> &Id {
        &self.id
    }

    /// Access the underlying manager for legacy operations
    pub fn inner(&self) -> &Manager {
        &self.inner
    }

    /// Access the underlying manager mutably for legacy operations
    pub fn inner_mut(&mut self) -> &mut Manager {
        &mut self.inner
    }
}

impl AsyncManager<RuntimeManagerId> {
    /// Create a new async manager with auto-generated ID
    pub fn new() -> Result<Self> {
        Self::with_id(RuntimeManagerId::new())
    }

    /// Create a new async manager with explicit configuration and auto-generated ID
    pub fn with_config(config: VulkanConfig) -> Result<Self> {
        Self::with_config_and_id(config, RuntimeManagerId::new())
    }
}

// =============================================================================
// Type-Safe Object Creation for Sync Manager
// =============================================================================

impl<Id> SyncManager<Id>
where
    Id: Clone + 'static,
{
    /// Create a type-safe tensor
    ///
    /// # Arguments
    ///
    /// * `data` - Initial tensor data
    ///
    /// # Returns
    ///
    /// A type-safe sync tensor handle
    pub fn create_tensor<T>(&mut self, data: &[T]) -> Result<TypedTensor<T, Id>>
    where
        T: TensorData,
    {
        let tensor = Tensor::new(&mut self.inner, data)?;
        let handle = SyncTensorHandle::new(tensor.handle());
        Ok(TypedTensor {
            inner: tensor,
            handle,
            _phantom: PhantomData,
        })
    }

    /// Create a type-safe tensor with specified size
    ///
    /// # Arguments
    ///
    /// * `size` - Number of elements in the tensor
    ///
    /// # Returns
    ///
    /// A type-safe sync tensor handle
    pub fn create_tensor_uninit<T>(&mut self, size: usize) -> Result<TypedTensor<T, Id>>
    where
        T: TensorData,
    {
        let tensor = Tensor::uninit(&mut self.inner, size)?;
        let handle = SyncTensorHandle::new(tensor.handle());
        Ok(TypedTensor {
            inner: tensor,
            handle,
            _phantom: PhantomData,
        })
    }

    /// Create a type-safe algorithm
    ///
    /// # Arguments
    ///
    /// * `spirv` - SPIR-V bytecode for the compute shader
    ///
    /// # Returns
    ///
    /// A type-safe sync algorithm handle
    pub fn create_algorithm(&mut self, spirv: &[u32]) -> Result<TypedAlgorithm<Id>> {
        let algorithm = Algorithm::new(&mut self.inner, spirv)?;
        let handle = SyncAlgorithmHandle::new(algorithm.handle());
        Ok(TypedAlgorithm {
            inner: algorithm,
            handle,
            _phantom: PhantomData,
        })
    }

    /// Create a type-safe sequence
    ///
    /// # Returns
    ///
    /// A type-safe sync sequence handle
    pub fn create_sequence(&mut self) -> Result<TypedSequence<Id>> {
        let sequence = Sequence::new(&mut self.inner)?;
        let handle = SyncSequenceHandle::new(sequence.handle());
        Ok(TypedSequence {
            inner: sequence,
            handle,
            _phantom: PhantomData,
        })
    }
}

// =============================================================================
// Type-Safe Object Creation for Async Manager (Future Implementation)
// =============================================================================

impl<Id> AsyncManager<Id>
where
    Id: Clone + 'static,
{
    /// Create a type-safe async tensor
    ///
    /// # Arguments
    ///
    /// * `data` - Initial tensor data
    ///
    /// # Returns
    ///
    /// A future that resolves to a type-safe async tensor handle
    pub async fn create_tensor<T>(&mut self, data: &[T]) -> Result<AsyncTypedTensor<T, Id>>
    where
        T: TensorData,
    {
        // Yield to allow other async operations
        #[cfg(feature = "async")]
        tokio::task::yield_now().await;

        let tensor = Tensor::new(&mut self.inner, data)?;
        let handle = AsyncTensorHandle::new(tensor.handle());
        Ok(AsyncTypedTensor {
            inner: tensor,
            handle,
            _phantom: PhantomData,
        })
    }

    /// Create a type-safe async algorithm
    ///
    /// # Arguments
    ///
    /// * `spirv` - SPIR-V bytecode for the compute shader
    ///
    /// # Returns
    ///
    /// A future that resolves to a type-safe async algorithm handle
    pub async fn create_algorithm(&mut self, spirv: &[u32]) -> Result<AsyncTypedAlgorithm<Id>> {
        #[cfg(feature = "async")]
        tokio::task::yield_now().await;

        let algorithm = Algorithm::new(&mut self.inner, spirv)?;
        let handle = AsyncAlgorithmHandle::new(algorithm.handle());
        Ok(AsyncTypedAlgorithm {
            inner: algorithm,
            handle,
            _phantom: PhantomData,
        })
    }

    /// Create a type-safe async sequence
    ///
    /// # Returns
    ///
    /// A future that resolves to a type-safe async sequence handle
    pub async fn create_sequence(&mut self) -> Result<AsyncTypedSequence<Id>> {
        #[cfg(feature = "async")]
        tokio::task::yield_now().await;

        let sequence = Sequence::new(&mut self.inner)?;
        let handle = AsyncSequenceHandle::new(sequence.handle());
        Ok(AsyncTypedSequence {
            inner: sequence,
            handle,
            _phantom: PhantomData,
        })
    }
}

// =============================================================================
// Type-Safe Object Wrappers
// =============================================================================

/// Type-safe tensor wrapper for sync operations
pub struct TypedTensor<T, Id> {
    inner: Tensor<T>,
    handle: SyncTensorHandle<SyncManagerType<Id>>,
    _phantom: PhantomData<T>,
}

impl<T, Id> TypedTensor<T, Id>
where
    Id: Copy,
{
    /// Get the type-safe handle
    pub fn handle(&self) -> SyncTensorHandle<SyncManagerType<Id>> {
        self.handle
    }

    /// Access the underlying tensor
    pub fn inner(&self) -> &Tensor<T> {
        &self.inner
    }

    /// Access the underlying tensor mutably
    pub fn inner_mut(&mut self) -> &mut Tensor<T> {
        &mut self.inner
    }
}

/// Type-safe tensor wrapper for async operations
pub struct AsyncTypedTensor<T, Id> {
    inner: Tensor<T>,
    handle: AsyncTensorHandle<AsyncManagerType<Id>>,
    _phantom: PhantomData<T>,
}

impl<T, Id> AsyncTypedTensor<T, Id>
where
    Id: Copy,
{
    /// Get the type-safe handle
    pub fn handle(&self) -> AsyncTensorHandle<AsyncManagerType<Id>> {
        self.handle
    }

    /// Access the underlying tensor
    pub fn inner(&self) -> &Tensor<T> {
        &self.inner
    }

    /// Access the underlying tensor mutably
    pub fn inner_mut(&mut self) -> &mut Tensor<T> {
        &mut self.inner
    }
}

/// Type-safe algorithm wrapper for sync operations
pub struct TypedAlgorithm<Id> {
    inner: Algorithm,
    handle: SyncAlgorithmHandle<SyncManagerType<Id>>,
    _phantom: PhantomData<Id>,
}

impl<Id> TypedAlgorithm<Id>
where
    Id: Copy,
{
    /// Get the type-safe handle
    pub fn handle(&self) -> SyncAlgorithmHandle<SyncManagerType<Id>> {
        self.handle
    }

    /// Access the underlying algorithm
    pub fn inner(&self) -> &Algorithm {
        &self.inner
    }

    /// Access the underlying algorithm mutably
    pub fn inner_mut(&mut self) -> &mut Algorithm {
        &mut self.inner
    }
}

/// Type-safe algorithm wrapper for async operations
pub struct AsyncTypedAlgorithm<Id> {
    inner: Algorithm,
    handle: AsyncAlgorithmHandle<AsyncManagerType<Id>>,
    _phantom: PhantomData<Id>,
}

impl<Id> AsyncTypedAlgorithm<Id>
where
    Id: Copy,
{
    /// Get the type-safe handle
    pub fn handle(&self) -> AsyncAlgorithmHandle<AsyncManagerType<Id>> {
        self.handle
    }

    /// Access the underlying algorithm
    pub fn inner(&self) -> &Algorithm {
        &self.inner
    }

    /// Access the underlying algorithm mutably
    pub fn inner_mut(&mut self) -> &mut Algorithm {
        &mut self.inner
    }
}

/// Type-safe sequence wrapper for sync operations
pub struct TypedSequence<Id> {
    inner: Sequence,
    handle: SyncSequenceHandle<SyncManagerType<Id>>,
    _phantom: PhantomData<Id>,
}

impl<Id> TypedSequence<Id>
where
    Id: Copy,
{
    /// Get the type-safe handle
    pub fn handle(&self) -> SyncSequenceHandle<SyncManagerType<Id>> {
        self.handle
    }

    /// Access the underlying sequence
    pub fn inner(&self) -> &Sequence {
        &self.inner
    }

    /// Access the underlying sequence mutably
    pub fn inner_mut(&mut self) -> &mut Sequence {
        &mut self.inner
    }

    /// Record a tensor sync operation with compile-time safety (consuming)
    pub fn record_tensor_sync_device(
        mut self,
        tensor_handle: SyncTensorHandle<SyncManagerType<Id>>,
    ) -> Result<Self> {
        self.inner = self
            .inner
            .record_tensor_sync_device(tensor_handle.raw_value())?;
        Ok(self)
    }

    /// Record an algorithm dispatch operation with compile-time safety (consuming)
    pub fn record_algorithm_dispatch(
        mut self,
        algorithm_handle: SyncAlgorithmHandle<SyncManagerType<Id>>,
    ) -> Result<Self> {
        self.inner = self
            .inner
            .record_algorithm_dispatch(algorithm_handle.raw_value())?;
        Ok(self)
    }

    /// Execute the typed sequence (consuming)
    pub fn execute(self) -> Result<SequenceResult> {
        self.inner.execute()
    }
}

/// Type-safe sequence wrapper for async operations
pub struct AsyncTypedSequence<Id> {
    inner: Sequence,
    handle: AsyncSequenceHandle<AsyncManagerType<Id>>,
    _phantom: PhantomData<Id>,
}

impl<Id> AsyncTypedSequence<Id>
where
    Id: Copy,
{
    /// Get the type-safe handle
    pub fn handle(&self) -> AsyncSequenceHandle<AsyncManagerType<Id>> {
        self.handle
    }

    /// Access the underlying sequence
    pub fn inner(&self) -> &Sequence {
        &self.inner
    }

    /// Access the underlying sequence mutably
    pub fn inner_mut(&mut self) -> &mut Sequence {
        &mut self.inner
    }

    /// Record a tensor sync operation with compile-time safety (consuming)
    pub async fn record_tensor_sync_device(
        mut self,
        tensor_handle: AsyncTensorHandle<AsyncManagerType<Id>>,
    ) -> Result<Self> {
        #[cfg(feature = "async")]
        tokio::task::yield_now().await;
        self.inner = self
            .inner
            .record_tensor_sync_device(tensor_handle.raw_value())?;
        Ok(self)
    }

    /// Record an algorithm dispatch operation with compile-time safety (consuming)
    pub async fn record_algorithm_dispatch(
        mut self,
        algorithm_handle: AsyncAlgorithmHandle<AsyncManagerType<Id>>,
    ) -> Result<Self> {
        #[cfg(feature = "async")]
        tokio::task::yield_now().await;
        self.inner = self
            .inner
            .record_algorithm_dispatch(algorithm_handle.raw_value())?;
        Ok(self)
    }

    /// Execute the async typed sequence (consuming)
    pub async fn execute(self) -> Result<SequenceResult> {
        #[cfg(feature = "async")]
        tokio::task::yield_now().await;
        self.inner.execute()
    }
}

// =============================================================================
// Convenience Functions for Creating Type-Safe Managers
// =============================================================================

/// Create a new type-safe sync manager with auto-generated unique ID
///
/// This is the recommended way to create sync managers for new code.
///
/// # Returns
///
/// A new sync manager with compile-time type safety
pub fn create_sync_manager() -> Result<SyncManager<RuntimeManagerId>> {
    SyncManager::new()
}

/// Create a new type-safe sync manager with explicit configuration
///
/// # Arguments
///
/// * `config` - Vulkan configuration
///
/// # Returns
///
/// A new sync manager with compile-time type safety
pub fn create_sync_manager_with_config(
    config: VulkanConfig,
) -> Result<SyncManager<RuntimeManagerId>> {
    SyncManager::with_config(config)
}

/// Create a new type-safe async manager with auto-generated unique ID
///
/// This is the recommended way to create async managers for new code.
///
/// # Returns
///
/// A new async manager with compile-time type safety
pub fn create_async_manager() -> Result<AsyncManager<RuntimeManagerId>> {
    AsyncManager::new()
}

/// Create a new type-safe async manager with explicit configuration
///
/// # Arguments
///
/// * `config` - Vulkan configuration
///
/// # Returns
///
/// A new async manager with compile-time type safety
pub fn create_async_manager_with_config(
    config: VulkanConfig,
) -> Result<AsyncManager<RuntimeManagerId>> {
    AsyncManager::with_config(config)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_manager_creation() {
        // Skip if no Vulkan available
        if Manager::device_count().unwrap_or(0) == 0 {
            return;
        }

        let manager = create_sync_manager();
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert!(manager.id().id() > 0);
    }

    #[test]
    fn test_async_manager_creation() {
        // Skip if no Vulkan available
        if Manager::device_count().unwrap_or(0) == 0 {
            return;
        }

        let manager = create_async_manager();
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert!(manager.id().id() > 0);
    }

    #[test]
    fn test_different_manager_ids() {
        // Skip if no Vulkan available
        if Manager::device_count().unwrap_or(0) == 0 {
            return;
        }

        let manager1 = create_sync_manager().unwrap();
        let manager2 = create_sync_manager().unwrap();

        assert_ne!(manager1.id().id(), manager2.id().id());
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_type_safe_tensor_creation() {
        // Skip if no Vulkan available
        if Manager::device_count().unwrap_or(0) == 0 {
            return;
        }

        let mut sync_manager = create_sync_manager().unwrap();
        let mut async_manager = create_async_manager().unwrap();

        let data = vec![1.0f32, 2.0, 3.0];

        let sync_tensor = sync_manager.create_tensor(&data);
        assert!(sync_tensor.is_ok());

        let async_tensor = async_manager.create_tensor(&data).await;
        assert!(async_tensor.is_ok());

        let sync_tensor = sync_tensor.unwrap();
        let async_tensor = async_tensor.unwrap();

        // Check that handles have correct types and values
        assert!(sync_tensor.handle().raw_value() > 0);
        assert!(async_tensor.handle().raw_value() > 0);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_type_safe_sequence_operations() {
        // Skip if no Vulkan available
        if Manager::device_count().unwrap_or(0) == 0 {
            return;
        }

        let mut sync_manager = create_sync_manager().unwrap();
        let data = vec![1.0f32, 2.0, 3.0];

        let sync_tensor = sync_manager.create_tensor(&data).unwrap();
        let sync_sequence = sync_manager.create_sequence().unwrap();

        // This should work - same manager type and sync mode
        let result = sync_sequence.record_tensor_sync_device(sync_tensor.handle());
        assert!(result.is_ok());
    }
}
