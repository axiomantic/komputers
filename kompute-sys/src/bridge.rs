//! CXX bridge for Kompute C++ library

#[cxx::bridge]
pub mod ffi {
    // Opaque types for C++ objects
    unsafe extern "C++" {
        include!("kompute_bridge.hpp");

        // Manager class
        type Manager;

        // Factory method for Manager
        fn create_manager(device_id: u32) -> UniquePtr<Manager>;

        // Config-based factory method for Manager
        fn create_manager_with_config(
            device_id: u32,
            use_moltenvk: bool,
            use_validation: bool,
            use_portability_extension: bool,
            icd_path: &str,
            layer_paths: &[String],
        ) -> UniquePtr<Manager>;

        // Device enumeration
        fn get_device_count() -> i32;

        // Config-based device enumeration
        fn get_device_count_with_config(
            use_moltenvk: bool,
            use_validation: bool,
            use_portability_extension: bool,
            icd_path: &str,
            layer_paths: &[String],
        ) -> i32;

        // Tensor class
        type Tensor;

        // Create tensor from f32 data
        fn create_tensor_f32(manager: Pin<&mut Manager>, data: &[f32]) -> SharedPtr<Tensor>;

        // Create uninitialized tensor
        fn create_tensor_uninit_f32(manager: Pin<&mut Manager>, size: usize) -> SharedPtr<Tensor>;

        // Multi-type tensor creation functions
        fn create_tensor_f64(manager: Pin<&mut Manager>, data: &[f64]) -> SharedPtr<Tensor>;
        fn create_tensor_i32(manager: Pin<&mut Manager>, data: &[i32]) -> SharedPtr<Tensor>;
        fn create_tensor_u32(manager: Pin<&mut Manager>, data: &[u32]) -> SharedPtr<Tensor>;
        fn create_tensor_i8(manager: Pin<&mut Manager>, data: &[i8]) -> SharedPtr<Tensor>;
        fn create_tensor_u8(manager: Pin<&mut Manager>, data: &[u8]) -> SharedPtr<Tensor>;

        // Create uninitialized multi-type tensors
        fn create_tensor_uninit_f64(manager: Pin<&mut Manager>, size: usize) -> SharedPtr<Tensor>;
        fn create_tensor_uninit_i32(manager: Pin<&mut Manager>, size: usize) -> SharedPtr<Tensor>;
        fn create_tensor_uninit_u32(manager: Pin<&mut Manager>, size: usize) -> SharedPtr<Tensor>;
        fn create_tensor_uninit_i8(manager: Pin<&mut Manager>, size: usize) -> SharedPtr<Tensor>;
        fn create_tensor_uninit_u8(manager: Pin<&mut Manager>, size: usize) -> SharedPtr<Tensor>;

        // Tensor operations
        fn tensor_size(tensor: Pin<&mut Tensor>) -> usize;
        fn tensor_size_bytes(tensor: Pin<&mut Tensor>) -> usize;

        // Get tensor data back to Rust
        fn tensor_data_f32(tensor: Pin<&mut Tensor>) -> Vec<f32>;

        // Algorithm class
        type Algorithm;

        // Create algorithm from SPIRV
        fn create_algorithm(manager: Pin<&mut Manager>, spirv: &[u32]) -> SharedPtr<Algorithm>;

        // Sequence class
        type Sequence;

        // Create sequence
        fn create_sequence(manager: Pin<&mut Manager>) -> SharedPtr<Sequence>;

        // Sequence operations
        fn sequence_record_tensor_sync_device(sequence: Pin<&mut Sequence>, tensor: &Tensor);
        fn sequence_record_tensor_sync_local(sequence: Pin<&mut Sequence>, tensor: &Tensor);
        fn sequence_record_algorithm_dispatch(sequence: Pin<&mut Sequence>, algorithm: &Algorithm);
        fn sequence_eval(sequence: Pin<&mut Sequence>);

        // Handle-based operations (new approach to solve SharedPtr limitations)

        // Tensor operations by handle
        fn create_tensor_f32_handle(manager: Pin<&mut Manager>, data: &[f32]) -> u64;
        fn create_tensor_uninit_f32_handle(manager: Pin<&mut Manager>, size: usize) -> u64;

        // Multi-type tensor operations by handle
        fn create_tensor_f64_handle(manager: Pin<&mut Manager>, data: &[f64]) -> u64;
        fn create_tensor_i32_handle(manager: Pin<&mut Manager>, data: &[i32]) -> u64;
        fn create_tensor_u32_handle(manager: Pin<&mut Manager>, data: &[u32]) -> u64;

        fn create_tensor_uninit_f64_handle(manager: Pin<&mut Manager>, size: usize) -> u64;
        fn create_tensor_uninit_i32_handle(manager: Pin<&mut Manager>, size: usize) -> u64;
        fn create_tensor_uninit_u32_handle(manager: Pin<&mut Manager>, size: usize) -> u64;

        // Filled tensor creation by handle
        fn create_tensor_filled_f32_handle(
            manager: Pin<&mut Manager>,
            size: usize,
            fill_value: f32,
        ) -> u64;
        fn create_tensor_filled_f64_handle(
            manager: Pin<&mut Manager>,
            size: usize,
            fill_value: f64,
        ) -> u64;
        fn create_tensor_filled_i32_handle(
            manager: Pin<&mut Manager>,
            size: usize,
            fill_value: i32,
        ) -> u64;
        fn create_tensor_filled_u32_handle(
            manager: Pin<&mut Manager>,
            size: usize,
            fill_value: u32,
        ) -> u64;

        // Multi-type tensor data retrieval
        fn tensor_data_f64_by_handle(handle: u64) -> Vec<f64>;
        fn tensor_data_i32_by_handle(handle: u64) -> Vec<i32>;
        fn tensor_data_u32_by_handle(handle: u64) -> Vec<u32>;

        fn tensor_size_by_handle(handle: u64) -> usize;
        fn tensor_size_bytes_by_handle(handle: u64) -> usize;
        fn tensor_data_f32_by_handle(handle: u64) -> Vec<f32>;
        fn tensor_data_type_by_handle(handle: u64) -> u32; // DataType enum as u32
        fn tensor_destroy_handle(handle: u64);

        // Algorithm operations by handle
        fn create_algorithm_handle(manager: Pin<&mut Manager>, spirv: &[u32]) -> u64;
        fn algorithm_destroy_handle(handle: u64);
        fn algorithm_set_specialization_constants(algo_handle: u64, constants: &[f32]) -> bool;
        fn algorithm_set_push_constants(algo_handle: u64, constants: &[f32]) -> bool;

        // Sequence operations by handle
        fn create_sequence_handle(manager: Pin<&mut Manager>) -> u64;
        fn sequence_record_tensor_sync_device_by_handles(seq_handle: u64, tensor_handle: u64);
        fn sequence_record_tensor_sync_local_by_handles(seq_handle: u64, tensor_handle: u64);
        fn sequence_record_algorithm_dispatch_by_handles(seq_handle: u64, algo_handle: u64);
        fn sequence_eval_by_handle(handle: u64);
        fn sequence_eval_async_by_handle(handle: u64);
        fn sequence_eval_await_by_handle(handle: u64, timeout_ms: u64);
        fn sequence_destroy_handle(handle: u64);

        // Queue management operations
        fn create_queue_handle(manager: Pin<&mut Manager>, queue_family_index: u32) -> u64;
        fn create_sequence_on_queue_handle(manager: Pin<&mut Manager>, queue_handle: u64) -> u64;
        fn queue_submit_sequence(queue_handle: u64, sequence_handle: u64);
        fn queue_wait_idle(queue_handle: u64);
        fn queue_destroy_handle(handle: u64);
    }
}
