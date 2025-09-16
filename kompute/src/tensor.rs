//! Type-safe GPU memory management with host synchronization.

use crate::error::{KomputeError, Result};
use crate::manager::Manager;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Type-safe GPU tensor with automatic memory management.
///
/// Tensors represent GPU-allocated memory that can be efficiently used for
/// compute operations. They support various data types and provide safe
/// host-device memory synchronization.
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
/// ```rust,ignore
/// use kompute::{Manager, Tensor};
///
/// let manager = Manager::new()?;
/// let data = vec![1.0f32, 2.0, 3.0, 4.0];
/// let tensor = Tensor::new(&manager, &data)?;
///
/// // Tensor automatically manages GPU memory
/// println!("Tensor size: {}", tensor.size());
/// ```
pub struct Tensor<T> {
    handle: NonNull<kompute_sys::kp_Tensor>,
    size: usize,
    _phantom: PhantomData<T>,
}

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
    /// Returns [`KomputeError::MemoryError`] if GPU allocation fails or
    /// [`KomputeError::TensorError`] if data copying fails.
    pub fn new(manager: &Manager, data: &[T]) -> Result<Self> {
        unsafe {
            let size_bytes = std::mem::size_of_val(data);
            let handle = kompute_sys::kp_tensor_new(
                manager.handle(),
                data.as_ptr() as *const std::ffi::c_void,
                size_bytes as u64,
                T::data_type_id(),
            );

            if handle.is_null() {
                return Err(KomputeError::memory("Failed to create tensor"));
            }

            Ok(Self {
                handle: NonNull::new_unchecked(handle),
                size: data.len(),
                _phantom: PhantomData,
            })
        }
    }

    /// Create a new tensor with uninitialized GPU memory.
    ///
    /// # Arguments
    ///
    /// * `manager` - The GPU manager to allocate memory from
    /// * `size` - Number of elements to allocate
    ///
    /// # Safety
    ///
    /// The tensor contains uninitialized memory until data is explicitly
    /// written to it via compute shaders or host synchronization.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::MemoryError`] if GPU allocation fails.
    pub fn uninit(manager: &Manager, size: usize) -> Result<Self> {
        unsafe {
            let size_bytes = size * std::mem::size_of::<T>();
            let handle = kompute_sys::kp_tensor_new(
                manager.handle(),
                std::ptr::null(),
                size_bytes as u64,
                T::data_type_id(),
            );

            if handle.is_null() {
                return Err(KomputeError::memory(
                    "Failed to create uninitialized tensor",
                ));
            }

            Ok(Self {
                handle: NonNull::new_unchecked(handle),
                size,
                _phantom: PhantomData,
            })
        }
    }

    /// Get the number of elements in the tensor.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the size in bytes of the tensor data.
    pub fn size_bytes(&self) -> usize {
        self.size * std::mem::size_of::<T>()
    }

    /// Synchronize tensor data from device to host memory.
    ///
    /// This copies GPU memory back to host-accessible memory, making it
    /// available for CPU operations.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if synchronization fails.
    pub fn sync_device_to_host(&mut self) -> Result<()> {
        unsafe {
            let result = kompute_sys::kp_tensor_map_to_host(self.handle.as_ptr());
            if result != 0 {
                return Err(KomputeError::compute("Failed to sync device to host"));
            }
        }
        Ok(())
    }

    /// Synchronize tensor data from host to device memory.
    ///
    /// This copies host-accessible memory to GPU memory, making updates
    /// visible to compute shaders.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if synchronization fails.
    pub fn sync_host_to_device(&mut self) -> Result<()> {
        unsafe {
            let result = kompute_sys::kp_tensor_map_to_device(self.handle.as_ptr());
            if result != 0 {
                return Err(KomputeError::compute("Failed to sync host to device"));
            }
        }
        Ok(())
    }

    /// Copy tensor data to a Vec on the host.
    ///
    /// This is a convenience method that performs device-to-host synchronization
    /// and returns the data as an owned Vec.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if synchronization fails.
    pub fn to_vec(&mut self) -> Result<Vec<T>> {
        self.sync_device_to_host()?;
        // Note: This would require additional FFI to get the host pointer
        // For now, this is a placeholder for the safe API design
        todo!("Implement safe host data access via additional FFI")
    }

    /// Get the raw handle for FFI operations.
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while this Tensor exists.
    /// Using the handle after the Tensor is dropped results in undefined behavior.
    #[allow(dead_code)]
    pub(crate) unsafe fn handle(&self) -> *mut kompute_sys::kp_Tensor {
        self.handle.as_ptr()
    }
}

// RAII cleanup - automatically destroy the tensor when dropped
impl<T> Drop for Tensor<T> {
    fn drop(&mut self) {
        unsafe {
            kompute_sys::kp_tensor_destroy(self.handle.as_ptr());
        }
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

    #[test]
    fn test_tensor_data_types() {
        assert_eq!(f32::data_type_id(), 0);
        assert_eq!(f64::data_type_id(), 1);
        assert_eq!(i32::data_type_id(), 2);
        assert_eq!(u32::data_type_id(), 3);
    }

    #[test]
    fn test_tensor_creation() {
        // This test may fail in environments without Vulkan support
        if let Ok(manager) = Manager::new() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            match Tensor::new(&manager, &data) {
                Ok(tensor) => {
                    assert_eq!(tensor.size(), 4);
                    assert_eq!(tensor.size_bytes(), 16);
                }
                Err(e) => {
                    println!("Tensor creation failed (expected in CI): {}", e);
                }
            }
        }
    }
}
