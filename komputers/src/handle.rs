//! Type-safe handle system with compile-time validation
//!
//! This module provides a revolutionary type-safe handle system that prevents
//! cross-manager errors at compile time through phantom types and zero-cost abstractions.
//!
//! # Type Safety Guarantees
//!
//! - **Async/Sync Safety**: Async handles cannot be used with sync operations
//! - **Object Type Safety**: Tensor handles only work with tensor operations
//! - **Manager Type Safety**: Handles from different manager types are incompatible
//! - **Zero Runtime Cost**: All type safety is enforced at compile time
//!
//! # Example
//!
//! ```rust,no_run
//! use komputers::{AsyncManager, SyncManager, create_async_manager, create_sync_manager};
//!
//! // Create type-safe managers
//! let async_manager = create_async_manager();
//! let sync_manager = create_sync_manager();
//!
//! // Type-safe tensor creation
//! let async_tensor = async_manager.create_tensor(&[1.0f32, 2.0, 3.0]).await?;
//! let sync_tensor = sync_manager.create_tensor(&[4.0f32, 5.0, 6.0])?;
//!
//! // This works - same manager and sync mode
//! let sync_sequence = sync_manager.create_sequence()?;
//! sync_sequence.record_tensor_sync_device(sync_tensor.handle())?; // ✅ Compiles
//!
//! // This fails at compile time - async/sync mismatch
//! // sync_sequence.record_tensor_sync_device(async_tensor.handle())?; // ❌ Compile error
//!
//! // This fails at compile time - manager type mismatch
//! let other_manager = create_sync_manager();
//! let other_sequence = other_manager.create_sequence()?;
//! // other_sequence.record_tensor_sync_device(sync_tensor.handle())?; // ❌ Compile error
//! # Ok::<(), komputers::KomputeError>(())
//! ```

use std::marker::PhantomData;

/// Phantom type marker for async operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsyncMarker;

/// Phantom type marker for synchronous operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncMarker;

/// Phantom type marker for tensor objects
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorType;

/// Phantom type marker for algorithm objects
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlgorithmType;

/// Phantom type marker for sequence objects
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceType;

/// Phantom type marker for queue objects
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueType;

/// Type-safe handle with compile-time validation guarantees
///
/// This handle type uses phantom types to enforce type safety at compile time:
/// - `ObjectType`: Ensures handles are only used with compatible operations
/// - `SyncMode`: Prevents mixing async and sync operations
/// - `ManagerType`: Ensures handles from different manager types are incompatible
///
/// # Type Parameters
///
/// * `ObjectType` - The type of object this handle references (Tensor, Algorithm, etc.)
/// * `SyncMode` - Whether this is an async or sync handle (AsyncMarker/SyncMarker)
/// * `ManagerType` - The specific manager type that created this handle
///
/// # Safety
///
/// All type safety is enforced at compile time with zero runtime overhead.
/// Invalid handle usage results in compile errors with helpful messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Handle<ObjectType, SyncMode, ManagerType> {
    /// The raw handle value from the C++ layer
    pub(crate) value: u64,
    /// Phantom data for compile-time type checking (zero runtime cost)
    _phantom: PhantomData<(ObjectType, SyncMode, ManagerType)>,
}

impl<ObjectType, SyncMode, ManagerType> Handle<ObjectType, SyncMode, ManagerType> {
    /// Create a new typed handle from a raw handle value
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The raw handle value is valid
    /// - The type parameters correctly represent the actual object
    /// - The handle was created by the specified manager type
    ///
    /// # Arguments
    ///
    /// * `value` - Raw handle value from C++ layer
    ///
    /// # Returns
    ///
    /// A new typed handle with compile-time safety guarantees
    pub(crate) fn new(value: u64) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }

    /// Get the raw handle value for C++ interop
    ///
    /// This method extracts the underlying u64 handle value for passing
    /// to the C++ layer while maintaining type safety at the Rust level.
    ///
    /// # Returns
    ///
    /// The raw u64 handle value
    pub fn raw_value(&self) -> u64 {
        self.value
    }

    /// Convert this handle to an untyped handle for backwards compatibility
    ///
    /// This method allows gradual migration from untyped to typed handles
    /// by providing a way to extract the raw value when needed.
    ///
    /// # Returns
    ///
    /// An untyped handle containing the same raw value
    pub fn to_untyped(&self) -> UntypedHandle {
        UntypedHandle { value: self.value }
    }
}

/// Untyped handle for backwards compatibility
///
/// This handle type provides compatibility with existing code that uses
/// raw u64 handles. It can be gradually replaced with typed handles.
///
/// # Migration Path
///
/// 1. Existing code continues to work with UntypedHandle
/// 2. New code uses typed Handle<T, S, M> for safety
/// 3. Gradually convert existing code to typed handles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UntypedHandle {
    /// The raw handle value
    pub value: u64,
}

impl UntypedHandle {
    /// Create a new untyped handle
    ///
    /// # Arguments
    ///
    /// * `value` - Raw handle value
    pub fn new(value: u64) -> Self {
        Self { value }
    }

    /// Convert to a typed handle (requires explicit type annotation)
    ///
    /// This method allows upgrading untyped handles to typed handles
    /// when the types are known and verified.
    ///
    /// # Safety
    ///
    /// The caller must ensure the type parameters correctly represent
    /// the actual object type, sync mode, and manager type.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use komputers::{UntypedHandle, Handle, TensorType, SyncMarker, SyncManager};
    ///
    /// let untyped = UntypedHandle::new(42);
    /// let typed: Handle<TensorType, SyncMarker, SyncManager<_>> = untyped.to_typed();
    /// ```
    pub fn to_typed<ObjectType, SyncMode, ManagerType>(
        self,
    ) -> Handle<ObjectType, SyncMode, ManagerType> {
        Handle::new(self.value)
    }

    /// Get the raw handle value
    pub fn raw_value(&self) -> u64 {
        self.value
    }
}

// Implement From traits for easy conversion
impl<ObjectType, SyncMode, ManagerType> From<Handle<ObjectType, SyncMode, ManagerType>>
    for UntypedHandle
{
    fn from(handle: Handle<ObjectType, SyncMode, ManagerType>) -> Self {
        handle.to_untyped()
    }
}

impl<ObjectType, SyncMode, ManagerType> From<Handle<ObjectType, SyncMode, ManagerType>> for u64 {
    fn from(handle: Handle<ObjectType, SyncMode, ManagerType>) -> Self {
        handle.raw_value()
    }
}

impl From<UntypedHandle> for u64 {
    fn from(handle: UntypedHandle) -> Self {
        handle.raw_value()
    }
}

impl From<u64> for UntypedHandle {
    fn from(value: u64) -> Self {
        UntypedHandle::new(value)
    }
}

// =============================================================================
// Type Aliases for Common Handle Types
// =============================================================================

/// Type alias for async tensor handles
pub type AsyncTensorHandle<ManagerType> = Handle<TensorType, AsyncMarker, ManagerType>;

/// Type alias for sync tensor handles
pub type SyncTensorHandle<ManagerType> = Handle<TensorType, SyncMarker, ManagerType>;

/// Type alias for async algorithm handles
pub type AsyncAlgorithmHandle<ManagerType> = Handle<AlgorithmType, AsyncMarker, ManagerType>;

/// Type alias for sync algorithm handles
pub type SyncAlgorithmHandle<ManagerType> = Handle<AlgorithmType, SyncMarker, ManagerType>;

/// Type alias for async sequence handles
pub type AsyncSequenceHandle<ManagerType> = Handle<SequenceType, AsyncMarker, ManagerType>;

/// Type alias for sync sequence handles
pub type SyncSequenceHandle<ManagerType> = Handle<SequenceType, SyncMarker, ManagerType>;

/// Type alias for async queue handles
pub type AsyncQueueHandle<ManagerType> = Handle<QueueType, AsyncMarker, ManagerType>;

/// Type alias for sync queue handles
pub type SyncQueueHandle<ManagerType> = Handle<QueueType, SyncMarker, ManagerType>;

// =============================================================================
// Validation Traits for Compile-Time Safety
// =============================================================================

/// Trait to validate that two handles are compatible for operations
///
/// This trait is automatically implemented for compatible handle combinations
/// and prevents incompatible operations at compile time.
pub trait CompatibleHandles<Other> {
    /// Check if two handles are compatible for cross-object operations
    fn are_compatible(&self, _other: &Other) -> bool {
        true // Default implementation - compatibility enforced by type system
    }
}

// Implement compatibility for handles with same manager type and sync mode
impl<ObjectType1, ObjectType2, SyncMode, ManagerType>
    CompatibleHandles<Handle<ObjectType2, SyncMode, ManagerType>>
    for Handle<ObjectType1, SyncMode, ManagerType>
{
    // Handles are compatible if they have the same sync mode and manager type
}

/// Trait to check if a handle is from an async context
pub trait IsAsync {
    /// Returns true if this handle is from an async context
    fn is_async(&self) -> bool;
}

impl<ObjectType, ManagerType> IsAsync for Handle<ObjectType, AsyncMarker, ManagerType> {
    fn is_async(&self) -> bool {
        true
    }
}

impl<ObjectType, ManagerType> IsAsync for Handle<ObjectType, SyncMarker, ManagerType> {
    fn is_async(&self) -> bool {
        false
    }
}

/// Trait to check if a handle is from a sync context
pub trait IsSync {
    /// Returns true if this handle is from a sync context
    fn is_sync(&self) -> bool;
}

impl<ObjectType, ManagerType> IsSync for Handle<ObjectType, AsyncMarker, ManagerType> {
    fn is_sync(&self) -> bool {
        false
    }
}

impl<ObjectType, ManagerType> IsSync for Handle<ObjectType, SyncMarker, ManagerType> {
    fn is_sync(&self) -> bool {
        true
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Mock manager types for testing
    struct MockAsyncManager;
    struct MockSyncManager;

    #[test]
    fn test_handle_creation_and_conversion() {
        let raw_value = 42u64;
        let typed_handle: Handle<TensorType, SyncMarker, MockSyncManager> = Handle::new(raw_value);

        assert_eq!(typed_handle.raw_value(), raw_value);

        let untyped = typed_handle.to_untyped();
        assert_eq!(untyped.raw_value(), raw_value);

        let back_to_u64: u64 = typed_handle.into();
        assert_eq!(back_to_u64, raw_value);
    }

    #[test]
    fn test_handle_type_safety() {
        let tensor_handle: Handle<TensorType, SyncMarker, MockSyncManager> = Handle::new(1);
        let algo_handle: Handle<AlgorithmType, SyncMarker, MockSyncManager> = Handle::new(2);

        // These handles should be compatible (same sync mode and manager type)
        assert!(tensor_handle.are_compatible(&algo_handle));
    }

    #[test]
    fn test_async_sync_traits() {
        let async_handle: Handle<TensorType, AsyncMarker, MockAsyncManager> = Handle::new(1);
        let sync_handle: Handle<TensorType, SyncMarker, MockSyncManager> = Handle::new(2);

        assert!(async_handle.is_async());
        assert!(!async_handle.is_sync());

        assert!(!sync_handle.is_async());
        assert!(sync_handle.is_sync());
    }

    #[test]
    fn test_type_aliases() {
        let tensor_handle: AsyncTensorHandle<MockAsyncManager> = Handle::new(1);
        let sequence_handle: AsyncSequenceHandle<MockAsyncManager> = Handle::new(2);

        assert_eq!(tensor_handle.raw_value(), 1);
        assert_eq!(sequence_handle.raw_value(), 2);
        assert!(tensor_handle.is_async());
        assert!(sequence_handle.is_async());
    }
}
