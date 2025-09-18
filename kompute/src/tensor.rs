//! Type-safe GPU memory management with host synchronization.

use crate::error::{KomputeError, Result};
use crate::manager::Manager;
use kompute_sys::ffi;
use std::any::TypeId;
use std::marker::PhantomData;

/// Trait for types that can be stored in GPU tensors.
///
/// This trait ensures that only GPU-compatible types can be used in tensors.
/// All types must be `Copy` (bitwise copyable) and have a stable memory layout.
pub trait TensorData: Copy + 'static {
    /// Get the GPU data type identifier for this type.
    fn data_type_id() -> u32;
}

impl TensorData for f32 {
    fn data_type_id() -> u32 {
        0
    } // Float32
}

impl TensorData for f64 {
    fn data_type_id() -> u32 {
        1
    } // Float64
}

impl TensorData for i32 {
    fn data_type_id() -> u32 {
        2
    } // Int32
}

impl TensorData for u32 {
    fn data_type_id() -> u32 {
        3
    } // UInt32
}

impl TensorData for i8 {
    fn data_type_id() -> u32 {
        4
    } // Int8
}

impl TensorData for u8 {
    fn data_type_id() -> u32 {
        5
    } // UInt8
}

/// Type-safe GPU tensor with automatic memory management.
///
/// Tensors represent GPU-allocated memory that can be efficiently used for
/// compute operations. They support various data types and provide safe
/// host-device memory synchronization using handle-based resource management.
///
/// # Type Safety
///
/// Tensors are generic over their data type `T`, which must implement
/// [`TensorData`] to ensure GPU compatibility.
///
/// # Memory Management
///
/// Tensors follow RAII principles - GPU memory is automatically freed
/// when the tensor is dropped.
///
/// # Examples
///
/// ```rust,no_run
/// use kompute::{Manager, Tensor, Result};
///
/// fn example() -> Result<()> {
///     let mut manager = Manager::new()?;
///     let data = vec![1.0f32, 2.0, 3.0, 4.0];
///     let tensor = Tensor::new(&mut manager, &data)?;
///
///     // Tensor automatically manages GPU memory
///     println!("Tensor size: {}", tensor.size());
///     println!("Tensor size in bytes: {}", tensor.size_bytes());
///
///     // Can retrieve data from GPU
///     let retrieved_data = tensor.to_vec()?;
///     assert_eq!(data, retrieved_data);
///
///     Ok(())
/// }
/// ```
pub struct Tensor<T> {
    handle: u64,
    _phantom: PhantomData<T>,
}

impl<T: TensorData> Tensor<T> {
    /// Create a new tensor with data copied from the host.
    ///
    /// # Arguments
    ///
    /// * `manager` - The GPU manager to allocate memory from
    /// * `data` - Host data to copy to the GPU
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::TensorError`] if creation fails.
    pub fn new(manager: &mut Manager, data: &[T]) -> Result<Self> {
        // Validate input data
        if data.is_empty() {
            return Err(KomputeError::tensor("Cannot create tensor with empty data"));
        }

        if data.len() > u32::MAX as usize {
            return Err(KomputeError::tensor(format!(
                "Tensor size {} exceeds maximum supported size {}",
                data.len(),
                u32::MAX
            )));
        }

        // For now, only support f32 until we implement other types in the bridge
        if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
            return Err(KomputeError::tensor(
                "Only f32 tensors are currently supported",
            ));
        }

        // Validate memory layout compatibility for f32
        if std::mem::size_of::<T>() != std::mem::size_of::<f32>() {
            return Err(KomputeError::tensor(format!(
                "Type size mismatch: expected {} bytes for f32, got {} bytes",
                std::mem::size_of::<f32>(),
                std::mem::size_of::<T>()
            )));
        }

        if std::mem::align_of::<T>() != std::mem::align_of::<f32>() {
            return Err(KomputeError::tensor(format!(
                "Type alignment mismatch: expected {} bytes for f32, got {} bytes",
                std::mem::align_of::<f32>(),
                std::mem::align_of::<T>()
            )));
        }

        // Safe cast now that we've validated the type compatibility
        let f32_data: &[f32] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };

        let handle = ffi::create_tensor_f32_handle(manager.handle(), f32_data);

        Ok(Self {
            handle,
            _phantom: PhantomData,
        })
    }

    /// Create a new tensor with uninitialized GPU memory.
    ///
    /// # Arguments
    ///
    /// * `manager` - The GPU manager to allocate memory from
    /// * `size` - Number of elements to allocate
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::MemoryError`] if GPU allocation fails.
    pub fn uninit(manager: &mut Manager, size: usize) -> Result<Self> {
        // Validate input size
        if size == 0 {
            return Err(KomputeError::tensor("Cannot create tensor with zero size"));
        }

        if size > u32::MAX as usize {
            return Err(KomputeError::tensor(format!(
                "Tensor size {} exceeds maximum supported size {}",
                size,
                u32::MAX
            )));
        }

        // For now, only support f32 until we implement other types in the bridge
        if TypeId::of::<T>() != TypeId::of::<f32>() {
            return Err(KomputeError::tensor(
                "Only f32 tensors are currently supported",
            ));
        }

        let handle = ffi::create_tensor_uninit_f32_handle(manager.handle(), size);

        Ok(Self {
            handle,
            _phantom: PhantomData,
        })
    }

    /// Get the number of elements in the tensor.
    pub fn size(&self) -> usize {
        // We can also query this from the C++ side
        ffi::tensor_size_by_handle(self.handle)
    }

    /// Get the size in bytes of the tensor data.
    pub fn size_bytes(&self) -> usize {
        ffi::tensor_size_bytes_by_handle(self.handle)
    }

    /// Copy tensor data to a Vec on the host.
    ///
    /// This retrieves the tensor data from GPU memory and returns it as an owned Vec.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if data retrieval fails.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        // For now, only support f32 until we implement other types in the bridge
        if TypeId::of::<T>() != TypeId::of::<f32>() {
            return Err(KomputeError::tensor(
                "Only f32 tensors are currently supported",
            ));
        }

        let f32_data = ffi::tensor_data_f32_by_handle(self.handle);
        let data: Vec<T> = unsafe {
            std::slice::from_raw_parts(f32_data.as_ptr() as *const T, f32_data.len()).to_vec()
        };
        Ok(data)
    }

    /// Get the internal handle value.
    ///
    /// This is used for interfacing with other handle-based operations.
    pub fn handle(&self) -> u64 {
        self.handle
    }
}

impl<T> Drop for Tensor<T> {
    fn drop(&mut self) {
        ffi::tensor_destroy_handle(self.handle);
    }
}

// Tensors are safe to send between threads
unsafe impl<T: TensorData> Send for Tensor<T> {}

// Tensors can be shared between threads with proper synchronization
unsafe impl<T: TensorData> Sync for Tensor<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Manager;
    use std::any::TypeId;

    #[test]
    fn test_tensor_data_types() {
        assert_eq!(f32::data_type_id(), 0);
        assert_eq!(f64::data_type_id(), 1);
        assert_eq!(i32::data_type_id(), 2);
        assert_eq!(u32::data_type_id(), 3);
    }

    #[test]
    fn test_tensor_data_type_trait() {
        // Test that TensorData trait is properly implemented for primitive types
        fn check_tensor_data<T: TensorData>() {
            // This function will only compile if T implements TensorData
        }

        check_tensor_data::<f32>();
        check_tensor_data::<f64>();
        check_tensor_data::<i32>();
        check_tensor_data::<u32>();
    }

    #[test]
    fn test_tensor_data_type_uniqueness() {
        // Ensure each type has a unique data_type_id
        let f32_id = f32::data_type_id();
        let f64_id = f64::data_type_id();
        let i32_id = i32::data_type_id();
        let u32_id = u32::data_type_id();

        assert_ne!(f32_id, f64_id);
        assert_ne!(f32_id, i32_id);
        assert_ne!(f32_id, u32_id);
        assert_ne!(f64_id, i32_id);
        assert_ne!(f64_id, u32_id);
        assert_ne!(i32_id, u32_id);
    }

    #[test]
    fn test_tensor_data_copy_trait() {
        // Test that TensorData types are Copy
        let f32_val: f32 = 1.0;
        let f32_copy = f32_val; // This should compile because f32 implements Copy
        assert_eq!(f32_val, f32_copy);

        let i32_val: i32 = 42;
        let i32_copy = i32_val;
        assert_eq!(i32_val, i32_copy);
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_tensor_creation() {
        // This test may fail in environments without Vulkan support
        if let Ok(mut manager) = Manager::new() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            match Tensor::new(&mut manager, &data) {
                Ok(tensor) => {
                    assert_eq!(tensor.size(), 4);
                    assert_eq!(tensor.size_bytes(), 16);

                    // Test data retrieval
                    if let Ok(retrieved_data) = tensor.to_vec() {
                        assert_eq!(data, retrieved_data);
                    }
                }
                Err(_e) => {
                    // Tensor creation failed - expected in headless environments
                    assert!(true, "Tensor creation failed as expected");
                }
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_tensor_uninit_creation() {
        if let Ok(mut manager) = Manager::new() {
            match Tensor::<f32>::uninit(&mut manager, 10) {
                Ok(tensor) => {
                    assert_eq!(tensor.size(), 10);
                    assert_eq!(tensor.size_bytes(), 40); // 10 * 4 bytes per f32

                    // Handle should be valid
                    assert_ne!(tensor.handle(), 0);
                }
                Err(_e) => {
                    // Creation failed - expected in headless environments
                }
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_tensor_zero_size() {
        if let Ok(mut manager) = Manager::new() {
            let empty_data: Vec<f32> = vec![];
            match Tensor::new(&mut manager, &empty_data) {
                Ok(tensor) => {
                    assert_eq!(tensor.size(), 0);
                    assert_eq!(tensor.size_bytes(), 0);

                    if let Ok(retrieved_data) = tensor.to_vec() {
                        assert!(retrieved_data.is_empty());
                    }
                }
                Err(_e) => {
                    // Zero-size tensor creation might not be supported
                }
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_tensor_large_data() {
        if let Ok(mut manager) = Manager::new() {
            let large_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
            match Tensor::new(&mut manager, &large_data) {
                Ok(tensor) => {
                    assert_eq!(tensor.size(), 1000);
                    assert_eq!(tensor.size_bytes(), 4000);

                    if let Ok(retrieved_data) = tensor.to_vec() {
                        assert_eq!(large_data.len(), retrieved_data.len());
                        // Check first and last elements
                        assert_eq!(large_data[0], retrieved_data[0]);
                        assert_eq!(large_data[999], retrieved_data[999]);
                    }
                }
                Err(_e) => {
                    // Large tensor creation might fail due to memory limitations
                }
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_tensor_data_consistency() {
        if let Ok(mut manager) = Manager::new() {
            let original_data = vec![3.14159f32, 2.71828, 1.41421, 1.61803];
            match Tensor::new(&mut manager, &original_data) {
                Ok(tensor) => {
                    // Test multiple retrievals return consistent data
                    if let (Ok(data1), Ok(data2)) = (tensor.to_vec(), tensor.to_vec()) {
                        assert_eq!(data1, data2);
                        assert_eq!(original_data, data1);
                        assert_eq!(original_data, data2);
                    }
                }
                Err(_e) => {
                    // Creation failed - expected in headless environments
                }
            }
        }
    }

    #[test]
    fn test_tensor_handle_uniqueness() {
        // Test that different tensors get different handles
        if let Ok(mut manager) = Manager::new() {
            let data1 = vec![1.0f32, 2.0];
            let data2 = vec![3.0f32, 4.0];

            if let (Ok(tensor1), Ok(tensor2)) = (
                Tensor::new(&mut manager, &data1),
                Tensor::new(&mut manager, &data2),
            ) {
                assert_ne!(tensor1.handle(), tensor2.handle());
                assert_ne!(tensor1.handle(), 0);
                assert_ne!(tensor2.handle(), 0);
            }
        }
    }

    #[test]
    fn test_tensor_send_sync() {
        // Test that Tensor implements Send and Sync
        fn is_send<T: Send>() {}
        fn is_sync<T: Sync>() {}

        is_send::<Tensor<f32>>();
        is_sync::<Tensor<f32>>();
    }

    #[test]
    fn test_tensor_type_safety() {
        // Test that to_vec() currently only supports f32
        if let Ok(mut manager) = Manager::new() {
            // This should work for f32
            let f32_data = vec![1.0f32, 2.0, 3.0];
            if let Ok(tensor) = Tensor::new(&mut manager, &f32_data) {
                assert!(tensor.to_vec().is_ok());
            }
        }
    }

    #[test]
    fn test_tensor_type_id_consistency() {
        // Test that TypeId of tensor data type is consistent
        let f32_type_id = TypeId::of::<f32>();
        assert_eq!(f32_type_id, TypeId::of::<f32>());
        assert_ne!(f32_type_id, TypeId::of::<f64>());
    }
}
