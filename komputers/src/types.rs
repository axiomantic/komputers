//! Type system definitions for compile-time safety
//!
//! This module defines the core types used throughout the type-safe handle system,
//! including manager types, unique identifiers, and validation traits.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique manager IDs
static MANAGER_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for manager instances
///
/// Each manager instance gets a unique ID at compile time or runtime,
/// depending on the context. This enables both compile-time type safety
/// and runtime validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ManagerId<const ID: u64>;

impl<const ID: u64> ManagerId<ID> {
    /// Get the compile-time ID value
    pub const fn id() -> u64 {
        ID
    }
}

/// Runtime manager identifier for dynamic manager creation
///
/// When managers are created at runtime, they get a unique ID from
/// the global counter. This enables runtime validation of manager
/// instance compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeManagerId {
    id: u64,
}

impl RuntimeManagerId {
    /// Create a new runtime manager ID
    pub fn new() -> Self {
        let id = MANAGER_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self { id }
    }

    /// Get the runtime ID value
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl Default for RuntimeManagerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that have a manager ID
pub trait HasManagerId {
    /// Get the manager ID for this type
    fn manager_id(&self) -> u64;
}

impl<const ID: u64> HasManagerId for ManagerId<ID> {
    fn manager_id(&self) -> u64 {
        ID
    }
}

impl HasManagerId for RuntimeManagerId {
    fn manager_id(&self) -> u64 {
        self.id
    }
}

/// Manager type marker for async operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncManagerType<Id> {
    _phantom: PhantomData<Id>,
}

/// Manager type marker for sync operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SyncManagerType<Id> {
    _phantom: PhantomData<Id>,
}

/// Trait for validating manager type compatibility
pub trait ManagerTypeCompatible<Other> {
    /// Check if two manager types are compatible
    fn is_compatible_with(&self) -> bool {
        true
    }
}

// Same manager types are always compatible
impl<Id> ManagerTypeCompatible<AsyncManagerType<Id>> for AsyncManagerType<Id> {}
impl<Id> ManagerTypeCompatible<SyncManagerType<Id>> for SyncManagerType<Id> {}

// Different async manager types are not compatible at compile time
// This is enforced by the type system - no explicit implementation needed

/// Type-level validation for object type compatibility
pub trait ObjectTypeCompatible<Operation> {
    /// Check if this object type is compatible with the given operation
    fn supports_operation(&self) -> bool {
        true
    }
}

// Re-export handle types for convenience
pub use crate::handle::{
    AlgorithmType, AsyncMarker, QueueType, SequenceType, SyncMarker, TensorType,
};

/// Default manager ID for backwards compatibility
pub type DefaultManagerId = ManagerId<0>;

/// Type alias for the default async manager type
pub type DefaultAsyncManager = AsyncManagerType<DefaultManagerId>;

/// Type alias for the default sync manager type
pub type DefaultSyncManager = SyncManagerType<DefaultManagerId>;

// =============================================================================
// Compile-Time Type Validation Utilities
// =============================================================================

/// Compile-time assertion that two types are the same
pub struct AssertSameType<T, U>
where
    T: SameType<U>,
{
    _phantom: PhantomData<(T, U)>,
}

/// Trait for compile-time type equality checking
pub trait SameType<Other> {}

impl<T> SameType<T> for T {}

/// Compile-time assertion that a type has specific properties
pub trait TypeProperties {
    /// Whether this type represents an async operation
    const IS_ASYNC: bool;
    /// Whether this type represents a sync operation
    const IS_SYNC: bool;
    /// The object type this represents
    const OBJECT_TYPE: &'static str;
}

impl TypeProperties for crate::handle::AsyncMarker {
    const IS_ASYNC: bool = true;
    const IS_SYNC: bool = false;
    const OBJECT_TYPE: &'static str = "async";
}

impl TypeProperties for crate::handle::SyncMarker {
    const IS_ASYNC: bool = false;
    const IS_SYNC: bool = true;
    const OBJECT_TYPE: &'static str = "sync";
}

impl TypeProperties for crate::handle::TensorType {
    const IS_ASYNC: bool = false;
    const IS_SYNC: bool = false;
    const OBJECT_TYPE: &'static str = "tensor";
}

impl TypeProperties for crate::handle::AlgorithmType {
    const IS_ASYNC: bool = false;
    const IS_SYNC: bool = false;
    const OBJECT_TYPE: &'static str = "algorithm";
}

impl TypeProperties for crate::handle::SequenceType {
    const IS_ASYNC: bool = false;
    const IS_SYNC: bool = false;
    const OBJECT_TYPE: &'static str = "sequence";
}

impl TypeProperties for crate::handle::QueueType {
    const IS_ASYNC: bool = false;
    const IS_SYNC: bool = false;
    const OBJECT_TYPE: &'static str = "queue";
}

// =============================================================================
// Error Types for Type Safety Violations
// =============================================================================

/// Errors that can occur during type validation
#[derive(Debug, thiserror::Error)]
pub enum TypeSafetyError {
    /// Async/sync mode mismatch
    #[error("Cannot mix async ({async_type}) and sync ({sync_type}) operations")]
    AsyncSyncMismatch {
        /// The async type name
        async_type: &'static str,
        /// The sync type name
        sync_type: &'static str,
    },

    /// Object type mismatch
    #[error("Cannot use {actual_type} handle with {expected_type} operation")]
    ObjectTypeMismatch {
        /// The actual object type provided
        actual_type: &'static str,
        /// The expected object type
        expected_type: &'static str,
    },

    /// Manager type mismatch
    #[error(
        "Cannot use handle from manager type {source_manager} with manager type {target_manager}"
    )]
    ManagerTypeMismatch {
        /// The source manager type
        source_manager: &'static str,
        /// The target manager type
        target_manager: &'static str,
    },

    /// Manager instance mismatch (runtime validation)
    #[error(
        "Cannot use handle from manager instance {source_id} with manager instance {target_id}"
    )]
    ManagerInstanceMismatch {
        /// The source manager instance ID
        source_id: u64,
        /// The target manager instance ID
        target_id: u64,
    },

    /// Invalid handle
    #[error("Handle {handle_id} is invalid or from destroyed manager")]
    InvalidHandle {
        /// The invalid handle ID
        handle_id: u64,
    },
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_id_generation() {
        let id1 = RuntimeManagerId::new();
        let id2 = RuntimeManagerId::new();

        assert_ne!(id1.id(), id2.id());
        assert!(id1.id() > 0);
        assert!(id2.id() > 0);
    }

    #[test]
    fn test_compile_time_manager_id() {
        type Manager1 = ManagerId<1>;
        type Manager2 = ManagerId<2>;

        assert_eq!(Manager1::id(), 1);
        assert_eq!(Manager2::id(), 2);
        assert_ne!(Manager1::id(), Manager2::id());
    }

    #[test]
    fn test_type_properties() {
        assert!(crate::handle::AsyncMarker::IS_ASYNC);
        assert!(!crate::handle::AsyncMarker::IS_SYNC);

        assert!(!crate::handle::SyncMarker::IS_ASYNC);
        assert!(crate::handle::SyncMarker::IS_SYNC);

        assert_eq!(crate::handle::TensorType::OBJECT_TYPE, "tensor");
        assert_eq!(crate::handle::AlgorithmType::OBJECT_TYPE, "algorithm");
    }

    #[test]
    fn test_manager_id_trait() {
        let runtime_id = RuntimeManagerId::new();
        let compile_time_id = ManagerId::<42>;

        assert_eq!(runtime_id.manager_id(), runtime_id.id());
        assert_eq!(compile_time_id.manager_id(), 42);
    }

    #[test]
    fn test_error_display() {
        let error = TypeSafetyError::AsyncSyncMismatch {
            async_type: "AsyncTensor",
            sync_type: "SyncSequence",
        };

        let error_string = error.to_string();
        assert!(error_string.contains("async"));
        assert!(error_string.contains("sync"));
    }
}
