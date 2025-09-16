//! Error types and result handling for Kompute operations.

use thiserror::Error;

/// Specialized Result type for Kompute operations.
pub type Result<T> = std::result::Result<T, KomputeError>;

/// Comprehensive error type for all Kompute operations.
#[derive(Error, Debug)]
pub enum KomputeError {
    /// Vulkan driver or device errors
    #[error("Vulkan error: {0}")]
    VulkanError(String),

    /// GPU hardware errors (device lost, out of memory, etc.)
    #[error("GPU hardware error: {0}")]
    HardwareError(String),

    /// Invalid shader or algorithm errors
    #[error("Shader error: {0}")]
    ShaderError(String),

    /// Tensor operation errors (invalid dimensions, type mismatches, etc.)
    #[error("Tensor error: {0}")]
    TensorError(String),

    /// Memory allocation or management errors
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// General GPU compute operation errors
    #[error("Compute error: {0}")]
    ComputeError(String),

    /// Device initialization or management errors
    #[error("Device error: {0}")]
    DeviceError(String),

    /// Invalid input parameters or configuration
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Resource not found or unavailable
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    /// Operation not supported on current platform or device
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// Generic internal error with context
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl KomputeError {
    /// Create a new Vulkan error with context
    pub fn vulkan<S: Into<String>>(msg: S) -> Self {
        KomputeError::VulkanError(msg.into())
    }

    /// Create a new hardware error with context
    pub fn hardware<S: Into<String>>(msg: S) -> Self {
        KomputeError::HardwareError(msg.into())
    }

    /// Create a new shader error with context
    pub fn shader<S: Into<String>>(msg: S) -> Self {
        KomputeError::ShaderError(msg.into())
    }

    /// Create a new tensor error with context
    pub fn tensor<S: Into<String>>(msg: S) -> Self {
        KomputeError::TensorError(msg.into())
    }

    /// Create a new memory error with context
    pub fn memory<S: Into<String>>(msg: S) -> Self {
        KomputeError::MemoryError(msg.into())
    }

    /// Create a new compute error with context
    pub fn compute<S: Into<String>>(msg: S) -> Self {
        KomputeError::ComputeError(msg.into())
    }

    /// Create a new device error with context
    pub fn device<S: Into<String>>(msg: S) -> Self {
        KomputeError::DeviceError(msg.into())
    }

    /// Create a new invalid parameter error with context
    pub fn invalid_parameter<S: Into<String>>(msg: S) -> Self {
        KomputeError::InvalidParameter(msg.into())
    }

    /// Create a new resource not found error with context
    pub fn resource_not_found<S: Into<String>>(msg: S) -> Self {
        KomputeError::ResourceNotFound(msg.into())
    }

    /// Create a new not supported error with context
    pub fn not_supported<S: Into<String>>(msg: S) -> Self {
        KomputeError::NotSupported(msg.into())
    }

    /// Create a new internal error with context
    pub fn internal<S: Into<String>>(msg: S) -> Self {
        KomputeError::InternalError(msg.into())
    }
}
