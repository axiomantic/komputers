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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation_functions() {
        let vulkan_err = KomputeError::vulkan("Test vulkan error");
        assert!(matches!(vulkan_err, KomputeError::VulkanError(_)));
        assert_eq!(vulkan_err.to_string(), "Vulkan error: Test vulkan error");

        let hardware_err = KomputeError::hardware("Test hardware error");
        assert!(matches!(hardware_err, KomputeError::HardwareError(_)));
        assert_eq!(
            hardware_err.to_string(),
            "GPU hardware error: Test hardware error"
        );

        let shader_err = KomputeError::shader("Test shader error");
        assert!(matches!(shader_err, KomputeError::ShaderError(_)));
        assert_eq!(shader_err.to_string(), "Shader error: Test shader error");

        let tensor_err = KomputeError::tensor("Test tensor error");
        assert!(matches!(tensor_err, KomputeError::TensorError(_)));
        assert_eq!(tensor_err.to_string(), "Tensor error: Test tensor error");

        let memory_err = KomputeError::memory("Test memory error");
        assert!(matches!(memory_err, KomputeError::MemoryError(_)));
        assert_eq!(memory_err.to_string(), "Memory error: Test memory error");

        let compute_err = KomputeError::compute("Test compute error");
        assert!(matches!(compute_err, KomputeError::ComputeError(_)));
        assert_eq!(compute_err.to_string(), "Compute error: Test compute error");

        let device_err = KomputeError::device("Test device error");
        assert!(matches!(device_err, KomputeError::DeviceError(_)));
        assert_eq!(device_err.to_string(), "Device error: Test device error");

        let param_err = KomputeError::invalid_parameter("Test parameter error");
        assert!(matches!(param_err, KomputeError::InvalidParameter(_)));
        assert_eq!(
            param_err.to_string(),
            "Invalid parameter: Test parameter error"
        );

        let resource_err = KomputeError::resource_not_found("Test resource error");
        assert!(matches!(resource_err, KomputeError::ResourceNotFound(_)));
        assert_eq!(
            resource_err.to_string(),
            "Resource not found: Test resource error"
        );

        let not_supported_err = KomputeError::not_supported("Test not supported error");
        assert!(matches!(not_supported_err, KomputeError::NotSupported(_)));
        assert_eq!(
            not_supported_err.to_string(),
            "Operation not supported: Test not supported error"
        );

        let internal_err = KomputeError::internal("Test internal error");
        assert!(matches!(internal_err, KomputeError::InternalError(_)));
        assert_eq!(
            internal_err.to_string(),
            "Internal error: Test internal error"
        );
    }

    #[test]
    fn test_error_string_conversions() {
        // Test that different string types can be converted
        let string_owned = String::from("owned string");
        let err1 = KomputeError::vulkan(string_owned);
        assert_eq!(err1.to_string(), "Vulkan error: owned string");

        let string_ref = "string ref";
        let err2 = KomputeError::vulkan(string_ref);
        assert_eq!(err2.to_string(), "Vulkan error: string ref");

        let cow_string = std::borrow::Cow::from("cow string");
        let err3 = KomputeError::vulkan(cow_string);
        assert_eq!(err3.to_string(), "Vulkan error: cow string");
    }

    #[test]
    fn test_error_debug_formatting() {
        let err = KomputeError::vulkan("test error");
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("VulkanError"));
        assert!(debug_str.contains("test error"));
    }

    #[test]
    fn test_result_type() {
        fn success_function() -> Result<i32> {
            Ok(42)
        }

        fn error_function() -> Result<i32> {
            Err(KomputeError::vulkan("test error"))
        }

        assert_eq!(success_function().unwrap(), 42);
        assert!(error_function().is_err());

        let err = error_function().unwrap_err();
        assert!(matches!(err, KomputeError::VulkanError(_)));
    }

    #[test]
    fn test_error_propagation() {
        fn inner_function() -> Result<()> {
            Err(KomputeError::compute("inner error"))
        }

        fn outer_function() -> Result<()> {
            inner_function()?;
            Ok(())
        }

        let result = outer_function();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, KomputeError::ComputeError(_)));
        assert_eq!(err.to_string(), "Compute error: inner error");
    }

    #[test]
    fn test_error_chaining() {
        use std::error::Error;

        let err = KomputeError::vulkan("test error");

        // Test that error implements std::error::Error trait
        let _: &dyn Error = &err;

        // For this simple error type, source should be None
        assert!(err.source().is_none());
    }
}
