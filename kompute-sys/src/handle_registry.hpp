#pragma once

#include "rust/cxx.h"
#include "kompute/Manager.hpp"
#include "kompute/Tensor.hpp"
#include "kompute/Algorithm.hpp"
#include "kompute/Sequence.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <mutex>

// Type aliases for cleaner code
using Manager = kp::Manager;
using Tensor = kp::Tensor;
using Algorithm = kp::Algorithm;
using Sequence = kp::Sequence;

/// Thread-safe handle registry for managing GPU resources.
/// This replaces the global variables with a proper RAII class.
class HandleRegistry {
private:
    std::unordered_map<uint64_t, std::shared_ptr<Tensor>> tensor_registry_;
    std::unordered_map<uint64_t, std::shared_ptr<Algorithm>> algorithm_registry_;
    std::unordered_map<uint64_t, std::shared_ptr<Sequence>> sequence_registry_;
    std::atomic<uint64_t> next_handle_{1};
    mutable std::mutex registry_mutex_;

public:
    HandleRegistry() = default;
    ~HandleRegistry() = default;

    // Delete copy constructor and assignment to prevent accidental copying
    HandleRegistry(const HandleRegistry&) = delete;
    HandleRegistry& operator=(const HandleRegistry&) = delete;

    // Move constructor and assignment are deleted due to atomic member
    HandleRegistry(HandleRegistry&&) = delete;
    HandleRegistry& operator=(HandleRegistry&&) = delete;

    /// Generate a unique handle
    uint64_t generate_handle() {
        return next_handle_.fetch_add(1);
    }

    /// Register a tensor and return its handle
    uint64_t register_tensor(std::shared_ptr<Tensor> tensor) {
        // Pattern 1: Input Validation at API Boundary
        if (!tensor) {
            throw std::invalid_argument("Cannot register null tensor");
        }

        // Pattern 2: Resource Limit Checking
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            if (tensor_registry_.size() >= 100000) { // Prevent memory exhaustion
                throw std::runtime_error("Tensor registry size limit exceeded");
            }
        }

        // Pattern 3: Atomic Handle Generation
        uint64_t handle = generate_handle();

        // Pattern 4: Exception Safety - Strong Guarantee
        try {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            tensor_registry_[handle] = std::move(tensor);
            return handle;
        } catch (...) {
            // If insertion fails, we've "leaked" a handle number, but that's acceptable
            // for safety vs trying to rollback the atomic counter
            throw;
        }
    }

    /// Register an algorithm and return its handle
    uint64_t register_algorithm(std::shared_ptr<Algorithm> algorithm) {
        uint64_t handle = generate_handle();
        std::lock_guard<std::mutex> lock(registry_mutex_);
        algorithm_registry_[handle] = std::move(algorithm);
        return handle;
    }

    /// Register a sequence and return its handle
    uint64_t register_sequence(std::shared_ptr<Sequence> sequence) {
        uint64_t handle = generate_handle();
        std::lock_guard<std::mutex> lock(registry_mutex_);
        sequence_registry_[handle] = std::move(sequence);
        return handle;
    }

    /// Get a tensor by handle (thread-safe)
    std::shared_ptr<Tensor> get_tensor(uint64_t handle) const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto it = tensor_registry_.find(handle);
        return (it != tensor_registry_.end()) ? it->second : nullptr;
    }

    /// Get an algorithm by handle (thread-safe)
    std::shared_ptr<Algorithm> get_algorithm(uint64_t handle) const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto it = algorithm_registry_.find(handle);
        return (it != algorithm_registry_.end()) ? it->second : nullptr;
    }

    /// Get a sequence by handle (thread-safe)
    std::shared_ptr<Sequence> get_sequence(uint64_t handle) const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto it = sequence_registry_.find(handle);
        return (it != sequence_registry_.end()) ? it->second : nullptr;
    }

    /// Remove a tensor handle
    bool remove_tensor(uint64_t handle) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        return tensor_registry_.erase(handle) > 0;
    }

    /// Remove an algorithm handle
    bool remove_algorithm(uint64_t handle) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        return algorithm_registry_.erase(handle) > 0;
    }

    /// Remove a sequence handle
    bool remove_sequence(uint64_t handle) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        return sequence_registry_.erase(handle) > 0;
    }

    /// Find tensor handle by object address (thread-safe)
    uint64_t find_tensor_handle(const Tensor* tensor_ptr) const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        for (const auto& pair : tensor_registry_) {
            if (pair.second.get() == tensor_ptr) {
                return pair.first;
            }
        }
        return 0; // Invalid handle
    }

    /// Find algorithm handle by object address (thread-safe)
    uint64_t find_algorithm_handle(const Algorithm* algo_ptr) const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        for (const auto& pair : algorithm_registry_) {
            if (pair.second.get() == algo_ptr) {
                return pair.first;
            }
        }
        return 0; // Invalid handle
    }

    /// Find sequence handle by object address (thread-safe)
    uint64_t find_sequence_handle(const Sequence* seq_ptr) const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        for (const auto& pair : sequence_registry_) {
            if (pair.second.get() == seq_ptr) {
                return pair.first;
            }
        }
        return 0; // Invalid handle
    }

    /// Get statistics (thread-safe)
    struct RegistryStats {
        size_t tensor_count;
        size_t algorithm_count;
        size_t sequence_count;
        uint64_t next_handle_value;
    };

    RegistryStats get_stats() const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        return RegistryStats{
            .tensor_count = tensor_registry_.size(),
            .algorithm_count = algorithm_registry_.size(),
            .sequence_count = sequence_registry_.size(),
            .next_handle_value = next_handle_.load()
        };
    }
};

/// Get the global registry instance (thread-safe singleton)
HandleRegistry& get_global_registry();
