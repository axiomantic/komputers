#pragma once

// =============================================================================
// C++ Safety Patterns Implementation
// =============================================================================
// This header follows modern C++ safety patterns to prevent segfaults:
// 1. RAII for automatic resource management
// 2. Smart pointer safety with null checking
// 3. Exception safety guarantees (basic to strong)
// 4. Input validation at all API boundaries
// 5. Thread safety with proper synchronization
// 6. Const-correctness and move semantics
// =============================================================================

#include "rust/cxx.h"
// Include Vulkan headers for direct device enumeration
#include <vulkan/vulkan.h>
// Include only specific Kompute headers we need, not the full Kompute.hpp
#include "kompute/Manager.hpp"
#include "kompute/Tensor.hpp"
#include "kompute/Algorithm.hpp"
#include "kompute/Sequence.hpp"
#include "kompute/operations/OpTensorSyncDevice.hpp"
#include "kompute/operations/OpTensorSyncLocal.hpp"
#include "kompute/operations/OpAlgoDispatch.hpp"

// Standard library includes for safety patterns
#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <optional>
#include <string_view>
#include <limits>
#include <cassert>

// Type aliases for cleaner code and type safety
using Manager = kp::Manager;
using Tensor = kp::Tensor;
using Algorithm = kp::Algorithm;
using Sequence = kp::Sequence;

// =============================================================================
// Safety Constants and Limits
// =============================================================================
namespace safety_limits {
    constexpr uint64_t INVALID_HANDLE = 0;
    constexpr uint64_t MAX_HANDLE_VALUE = std::numeric_limits<uint64_t>::max() - 1;
    constexpr size_t MAX_TENSOR_SIZE = 1024 * 1024 * 1024; // 1GB elements max
    constexpr size_t MAX_SPIRV_SIZE = 16 * 1024 * 1024;    // 16MB SPIR-V max
    constexpr uint32_t SPIRV_MAGIC_NUMBER = 0x07230203;
    constexpr size_t MIN_SPIRV_HEADER_SIZE = 5; // 5 32-bit words minimum
}

// =============================================================================
// Input Validation Utilities
// =============================================================================
namespace validation {
    // Check if a pointer is valid and not null
    template<typename T>
    inline bool is_valid_ptr(const T* ptr) noexcept {
        return ptr != nullptr;
    }

    // Check if a handle is valid (non-zero)
    inline bool is_valid_handle(uint64_t handle) noexcept {
        return handle != safety_limits::INVALID_HANDLE && handle <= safety_limits::MAX_HANDLE_VALUE;
    }

    // Check if a size is within safe limits
    inline bool is_safe_size(size_t size, size_t max_size) noexcept {
        return size > 0 && size <= max_size;
    }

    // Validate SPIR-V data
    inline bool is_valid_spirv(rust::Slice<const uint32_t> spirv) noexcept {
        return spirv.size() >= safety_limits::MIN_SPIRV_HEADER_SIZE &&
               spirv.size() <= safety_limits::MAX_SPIRV_SIZE &&
               spirv[0] == safety_limits::SPIRV_MAGIC_NUMBER;
    }
}

// Handle-based registry system for solving SharedPtr mutable access limitations
// These are now encapsulated in handle_registry.hpp for better safety
extern std::unordered_map<uint64_t, std::shared_ptr<Tensor>> tensor_registry;
extern std::unordered_map<uint64_t, std::shared_ptr<Algorithm>> algorithm_registry;
extern std::unordered_map<uint64_t, std::shared_ptr<Sequence>> sequence_registry;
extern std::atomic<uint64_t> next_handle;
extern std::mutex registry_mutex;

// Manager operations
std::unique_ptr<Manager> create_manager(uint32_t device_id);
int32_t get_device_count();

// Legacy SharedPtr operations (kept for compatibility)
std::shared_ptr<Tensor> create_tensor_f32(Manager& manager, rust::Slice<const float> data);
std::shared_ptr<Tensor> create_tensor_uninit_f32(Manager& manager, size_t size);

// Multi-type tensor creation (SharedPtr)
std::shared_ptr<Tensor> create_tensor_f64(Manager& manager, rust::Slice<const double> data);
std::shared_ptr<Tensor> create_tensor_i32(Manager& manager, rust::Slice<const int32_t> data);
std::shared_ptr<Tensor> create_tensor_u32(Manager& manager, rust::Slice<const uint32_t> data);
std::shared_ptr<Tensor> create_tensor_i8(Manager& manager, rust::Slice<const int8_t> data);
std::shared_ptr<Tensor> create_tensor_u8(Manager& manager, rust::Slice<const uint8_t> data);

std::shared_ptr<Tensor> create_tensor_uninit_f64(Manager& manager, size_t size);
std::shared_ptr<Tensor> create_tensor_uninit_i32(Manager& manager, size_t size);
std::shared_ptr<Tensor> create_tensor_uninit_u32(Manager& manager, size_t size);
std::shared_ptr<Tensor> create_tensor_uninit_i8(Manager& manager, size_t size);
std::shared_ptr<Tensor> create_tensor_uninit_u8(Manager& manager, size_t size);
size_t tensor_size(Tensor& tensor);
size_t tensor_size_bytes(Tensor& tensor);
rust::Vec<float> tensor_data_f32(Tensor& tensor);
std::shared_ptr<Algorithm> create_algorithm(Manager& manager, rust::Slice<const uint32_t> spirv);
std::shared_ptr<Sequence> create_sequence(Manager& manager);
void sequence_record_tensor_sync_device(Sequence& sequence, const Tensor& tensor);
void sequence_record_tensor_sync_local(Sequence& sequence, const Tensor& tensor);
void sequence_record_algorithm_dispatch(Sequence& sequence, const Algorithm& algorithm);
void sequence_eval(Sequence& sequence);

// Handle-based operations (new approach)
// Tensor operations by handle
uint64_t create_tensor_f32_handle(Manager& manager, rust::Slice<const float> data);
uint64_t create_tensor_uninit_f32_handle(Manager& manager, size_t size);

// Multi-type tensor operations by handle
uint64_t create_tensor_f64_handle(Manager& manager, rust::Slice<const double> data);
uint64_t create_tensor_i32_handle(Manager& manager, rust::Slice<const int32_t> data);
uint64_t create_tensor_u32_handle(Manager& manager, rust::Slice<const uint32_t> data);
uint64_t create_tensor_i8_handle(Manager& manager, rust::Slice<const int8_t> data);
uint64_t create_tensor_u8_handle(Manager& manager, rust::Slice<const uint8_t> data);

uint64_t create_tensor_uninit_f64_handle(Manager& manager, size_t size);
uint64_t create_tensor_uninit_i32_handle(Manager& manager, size_t size);
uint64_t create_tensor_uninit_u32_handle(Manager& manager, size_t size);
uint64_t create_tensor_uninit_i8_handle(Manager& manager, size_t size);
uint64_t create_tensor_uninit_u8_handle(Manager& manager, size_t size);

// Multi-type tensor data retrieval by handle
rust::Vec<double> tensor_data_f64_by_handle(uint64_t handle);
rust::Vec<int32_t> tensor_data_i32_by_handle(uint64_t handle);
rust::Vec<uint32_t> tensor_data_u32_by_handle(uint64_t handle);
rust::Vec<int8_t> tensor_data_i8_by_handle(uint64_t handle);
rust::Vec<uint8_t> tensor_data_u8_by_handle(uint64_t handle);

size_t tensor_size_by_handle(uint64_t handle);
size_t tensor_size_bytes_by_handle(uint64_t handle);
rust::Vec<float> tensor_data_f32_by_handle(uint64_t handle);
uint32_t tensor_data_type_by_handle(uint64_t handle);
void tensor_destroy_handle(uint64_t handle);

// Algorithm operations by handle
uint64_t create_algorithm_handle(Manager& manager, rust::Slice<const uint32_t> spirv);
void algorithm_destroy_handle(uint64_t handle);

// Sequence operations by handle
uint64_t create_sequence_handle(Manager& manager);
void sequence_record_tensor_sync_device_by_handles(uint64_t seq_handle, uint64_t tensor_handle);
void sequence_record_tensor_sync_local_by_handles(uint64_t seq_handle, uint64_t tensor_handle);
void sequence_record_algorithm_dispatch_by_handles(uint64_t seq_handle, uint64_t algo_handle);
void sequence_eval_by_handle(uint64_t handle);
void sequence_eval_async_by_handle(uint64_t handle);
void sequence_eval_await_by_handle(uint64_t handle, uint64_t timeout_ms);
void sequence_destroy_handle(uint64_t handle);

// Queue management operations
uint64_t create_queue_handle(Manager& manager, uint32_t queue_family_index);
uint64_t create_sequence_on_queue_handle(Manager& manager, uint64_t queue_handle);
void queue_submit_sequence(uint64_t queue_handle, uint64_t sequence_handle);
void queue_wait_idle(uint64_t queue_handle);
void queue_destroy_handle(uint64_t handle);
