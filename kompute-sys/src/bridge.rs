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

        // Device enumeration
        fn get_device_count() -> i32;

        // Tensor class
        type Tensor;

        // Create tensor from f32 data
        fn create_tensor_f32(manager: Pin<&mut Manager>, data: &[f32]) -> SharedPtr<Tensor>;

        // Create uninitialized tensor
        fn create_tensor_uninit_f32(manager: Pin<&mut Manager>, size: usize) -> SharedPtr<Tensor>;

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
        fn tensor_size_by_handle(handle: u64) -> usize;
        fn tensor_size_bytes_by_handle(handle: u64) -> usize;
        fn tensor_data_f32_by_handle(handle: u64) -> Vec<f32>;
        fn tensor_destroy_handle(handle: u64);

        // Algorithm operations by handle
        fn create_algorithm_handle(manager: Pin<&mut Manager>, spirv: &[u32]) -> u64;
        fn algorithm_destroy_handle(handle: u64);

        // Sequence operations by handle
        fn create_sequence_handle(manager: Pin<&mut Manager>) -> u64;
        fn sequence_record_tensor_sync_device_by_handles(seq_handle: u64, tensor_handle: u64);
        fn sequence_record_tensor_sync_local_by_handles(seq_handle: u64, tensor_handle: u64);
        fn sequence_record_algorithm_dispatch_by_handles(seq_handle: u64, algo_handle: u64);
        fn sequence_eval_by_handle(handle: u64);
        fn sequence_eval_async_by_handle(handle: u64);
        fn sequence_eval_await_by_handle(handle: u64, timeout_ms: u64);
        fn sequence_destroy_handle(handle: u64);
    }
}
