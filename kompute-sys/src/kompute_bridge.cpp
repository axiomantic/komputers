#include "kompute_bridge.hpp"
#include "handle_registry.hpp"
#include <stdexcept>
#include <cstdlib>
#include <string>

// Helper function to configure Vulkan environment for macOS
void configure_vulkan_environment() {
    const char* vulkan_sdk = std::getenv("VULKAN_SDK");
    if (vulkan_sdk) {
        // Set ICD file for MoltenVK
        std::string icd_path = std::string(vulkan_sdk) + "/share/vulkan/icd.d/MoltenVK_icd.json";
        setenv("VK_ICD_FILENAMES", icd_path.c_str(), 1);

        // Set layer path
        std::string layer_path = std::string(vulkan_sdk) + "/share/vulkan/explicit_layer.d";
        setenv("VK_LAYER_PATH", layer_path.c_str(), 1);

        // Disable validation layers for basic usage
        unsetenv("VK_INSTANCE_LAYERS");
    }
}

// Manager operations
std::unique_ptr<Manager> create_manager(uint32_t device_id) {
    try {
        // Configure Vulkan environment first
        configure_vulkan_environment();

        // First verify device_id is valid
        int32_t available_devices = get_device_count();
        if (available_devices <= 0) {
            throw std::runtime_error("No Vulkan devices available. Check that MoltenVK is properly installed and configured.");
        }
        if (device_id >= static_cast<uint32_t>(available_devices)) {
            throw std::runtime_error(std::string("Device ID ") + std::to_string(device_id) +
                " is invalid. Only " + std::to_string(available_devices) + " devices available.");
        }

        // Try to create with explicit error handling
        auto manager = std::make_unique<kp::Manager>(device_id);

        // Verify the manager was created successfully by testing a basic operation
        if (!manager) {
            throw std::runtime_error("Manager creation returned null pointer");
        }

        // Test if manager is functional by trying to list devices
        try {
            auto devices = manager->listDevices();
            if (devices.empty()) {
                throw std::runtime_error("Manager created but found no devices");
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Manager created but not functional: ") + e.what());
        }

        return manager;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create manager with device ") +
                                std::to_string(device_id) + ": " + e.what());
    }
}

int32_t get_device_count() {
    try {
        // Configure Vulkan environment first
        configure_vulkan_environment();

        // Use direct Vulkan calls to enumerate devices without creating a full Manager
        // This avoids potential initialization issues with the Manager constructor

        // Create Vulkan instance for device enumeration only
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Kompute Device Enumeration";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "Kompute";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // Enable portability enumeration for macOS/MoltenVK
        std::vector<const char*> extensions = {
            VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
        };
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        createInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

        VkInstance instance;
        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
        if (result != VK_SUCCESS) {
            return -1;
        }

        // Enumerate physical devices
        uint32_t deviceCount = 0;
        result = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        vkDestroyInstance(instance, nullptr);

        if (result != VK_SUCCESS) {
            return -1;
        }

        return static_cast<int32_t>(deviceCount);
    } catch (const std::exception&) {
        return -1;
    }
}

// Tensor operations
std::shared_ptr<Tensor> create_tensor_f32(Manager& manager, rust::Slice<const float> data) {
    try {
        std::vector<float> cpp_data(data.begin(), data.end());
        return manager.tensor(cpp_data);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create tensor: ") + e.what());
    }
}

std::shared_ptr<Tensor> create_tensor_uninit_f32(Manager& manager, size_t size) {
    try {
        std::vector<float> empty_data(size, 0.0f);
        return manager.tensor(empty_data);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create uninitialized tensor: ") + e.what());
    }
}

size_t tensor_size(Tensor& tensor) {
    return tensor.size();
}

size_t tensor_size_bytes(Tensor& tensor) {
    return tensor.memorySize();
}

rust::Vec<float> tensor_data_f32(Tensor& tensor) {
    try {
        auto cpp_vector = tensor.vector<float>();
        rust::Vec<float> result;
        result.reserve(cpp_vector.size());
        for (const auto& value : cpp_vector) {
            result.push_back(value);
        }
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to get tensor data: ") + e.what());
    }
}

// Algorithm operations
std::shared_ptr<Algorithm> create_algorithm(Manager& manager, rust::Slice<const uint32_t> spirv) {
    try {
        std::vector<uint32_t> cpp_spirv(spirv.begin(), spirv.end());

        // Create empty tensors vector for now - this would be properly parameterized in real usage
        std::vector<std::shared_ptr<kp::Tensor>> tensors;

        // Default workgroup and constants
        kp::Workgroup workgroup{1, 1, 1};
        std::vector<float> spec_constants;
        std::vector<float> push_constants;

        return manager.algorithm(tensors, cpp_spirv, workgroup, spec_constants, push_constants);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create algorithm: ") + e.what());
    }
}

// Sequence operations
std::shared_ptr<Sequence> create_sequence(Manager& manager) {
    try {
        return manager.sequence();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create sequence: ") + e.what());
    }
}

void sequence_record_tensor_sync_device(Sequence& sequence, const Tensor& tensor) {
    try {
        // Find the handles for both objects using reverse lookup
        auto& registry = get_global_registry();
        uint64_t seq_handle = registry.find_sequence_handle(&sequence);
        uint64_t tensor_handle = registry.find_tensor_handle(&tensor);

        if (seq_handle == 0) {
            throw std::runtime_error("Sequence not found in registry - must be created through handle system");
        }
        if (tensor_handle == 0) {
            throw std::runtime_error("Tensor not found in registry - must be created through handle system");
        }

        // Delegate to the handle-based implementation
        sequence_record_tensor_sync_device_by_handles(seq_handle, tensor_handle);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to record tensor sync device: ") + e.what());
    }
}

void sequence_record_tensor_sync_local(Sequence& sequence, const Tensor& tensor) {
    try {
        // Find the handles for both objects using reverse lookup
        auto& registry = get_global_registry();
        uint64_t seq_handle = registry.find_sequence_handle(&sequence);
        uint64_t tensor_handle = registry.find_tensor_handle(&tensor);

        if (seq_handle == 0) {
            throw std::runtime_error("Sequence not found in registry - must be created through handle system");
        }
        if (tensor_handle == 0) {
            throw std::runtime_error("Tensor not found in registry - must be created through handle system");
        }

        // Delegate to the handle-based implementation
        sequence_record_tensor_sync_local_by_handles(seq_handle, tensor_handle);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to record tensor sync local: ") + e.what());
    }
}

void sequence_record_algorithm_dispatch(Sequence& sequence, const Algorithm& algorithm) {
    try {
        // Find the handles for both objects using reverse lookup
        auto& registry = get_global_registry();
        uint64_t seq_handle = registry.find_sequence_handle(&sequence);
        uint64_t algo_handle = registry.find_algorithm_handle(&algorithm);

        if (seq_handle == 0) {
            throw std::runtime_error("Sequence not found in registry - must be created through handle system");
        }
        if (algo_handle == 0) {
            throw std::runtime_error("Algorithm not found in registry - must be created through handle system");
        }

        // Delegate to the handle-based implementation
        sequence_record_algorithm_dispatch_by_handles(seq_handle, algo_handle);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to record algorithm dispatch: ") + e.what());
    }
}

void sequence_eval(Sequence& sequence) {
    try {
        sequence.eval();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to evaluate sequence: ") + e.what());
    }
}

// ================================
// Handle-based operations implementation
// ================================

// Use the thread-safe registry for all operations

// =============================================================================
// C++ Safety Pattern Implementation: Tensor Operations
// =============================================================================

uint64_t create_tensor_f32_handle(Manager& manager, rust::Slice<const float> data) {
    // Pattern 1: Input Validation at API Boundary
    if (data.empty()) {
        throw std::invalid_argument("Cannot create tensor with empty data");
    }

    if (!validation::is_safe_size(data.size(), safety_limits::MAX_TENSOR_SIZE)) {
        throw std::invalid_argument("Tensor size exceeds safety limits: " +
                                  std::to_string(data.size()) + " elements");
    }

    try {
        // Pattern 2: Exception Safety - Strong Guarantee
        // All operations succeed or no side effects occur
        std::vector<float> cpp_data;
        cpp_data.reserve(data.size()); // Pre-allocate to avoid reallocation

        // Pattern 3: Safe Memory Copy with Iterator Safety
        std::copy(data.begin(), data.end(), std::back_inserter(cpp_data));

        // Pattern 4: RAII - Automatic cleanup if tensor creation fails
        auto tensor = manager.tensor(cpp_data);

        if (!tensor) {
            throw std::runtime_error("Manager returned null tensor");
        }

        // Pattern 5: Atomic Operation Registration
        auto handle = get_global_registry().register_tensor(tensor);

        if (!validation::is_valid_handle(handle)) {
            throw std::runtime_error("Registry returned invalid handle");
        }

        return handle;

    } catch (const std::exception& e) {
        // Pattern 6: Exception Translation with Context
        throw std::runtime_error(std::string("Failed to create tensor handle: ") + e.what());
    }
}

uint64_t create_tensor_uninit_f32_handle(Manager& manager, size_t size) {
    // Pattern 1: Input Validation at API Boundary
    if (size == 0) {
        throw std::invalid_argument("Cannot create tensor with zero size");
    }

    if (!validation::is_safe_size(size, safety_limits::MAX_TENSOR_SIZE)) {
        throw std::invalid_argument("Tensor size exceeds safety limits: " +
                                  std::to_string(size) + " elements");
    }

    try {
        // Pattern 2: Exception Safety - Strong Guarantee
        std::vector<float> empty_data;
        empty_data.reserve(size); // Pre-allocate for safety
        empty_data.resize(size, 0.0f); // Initialize to zero

        // Pattern 4: RAII - Automatic cleanup if tensor creation fails
        auto tensor = manager.tensor(empty_data);

        if (!tensor) {
            throw std::runtime_error("Manager returned null tensor");
        }

        // Pattern 5: Atomic Operation Registration
        auto handle = get_global_registry().register_tensor(tensor);

        if (!validation::is_valid_handle(handle)) {
            throw std::runtime_error("Registry returned invalid handle");
        }

        return handle;

    } catch (const std::exception& e) {
        // Pattern 6: Exception Translation with Context
        throw std::runtime_error(std::string("Failed to create uninitialized tensor handle: ") + e.what());
    }
}

size_t tensor_size_by_handle(uint64_t handle) {
    // Pattern 1: Input Validation at API Boundary
    if (!validation::is_valid_handle(handle)) {
        throw std::invalid_argument("Invalid tensor handle: " + std::to_string(handle));
    }

    // Pattern 2: Safe Resource Access with Null Checking
    auto tensor = get_global_registry().get_tensor(handle);
    if (!tensor) {
        throw std::runtime_error("Tensor handle not found in registry: " + std::to_string(handle));
    }

    // Pattern 3: Exception Safety - No side effects on failure
    try {
        return tensor->size();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to get tensor size for handle " +
                               std::to_string(handle) + ": " + e.what());
    }
}

size_t tensor_size_bytes_by_handle(uint64_t handle) {
    auto tensor = get_global_registry().get_tensor(handle);
    if (!tensor) {
        throw std::runtime_error("Invalid tensor handle");
    }
    return tensor->memorySize();
}

rust::Vec<float> tensor_data_f32_by_handle(uint64_t handle) {
    try {
        auto tensor = get_global_registry().get_tensor(handle);
        if (!tensor) {
            throw std::runtime_error("Invalid tensor handle");
        }

        auto cpp_vector = tensor->vector<float>();
        rust::Vec<float> result;
        result.reserve(cpp_vector.size());
        for (const auto& value : cpp_vector) {
            result.push_back(value);
        }
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to get tensor data by handle: ") + e.what());
    }
}

void tensor_destroy_handle(uint64_t handle) {
    get_global_registry().remove_tensor(handle);
}

// Algorithm operations by handle
uint64_t create_algorithm_handle(Manager& manager, rust::Slice<const uint32_t> spirv) {
    // Pattern 1: Input Validation at API Boundary
    if (spirv.empty()) {
        throw std::invalid_argument("Cannot create algorithm with empty SPIR-V data");
    }

    if (!validation::is_valid_spirv(spirv)) {
        throw std::invalid_argument("Invalid SPIR-V data: size=" + std::to_string(spirv.size()) +
                                  ", magic=" + (spirv.size() > 0 ?
                                    std::to_string(spirv[0]) : "N/A"));
    }

    try {
        // Pattern 2: Exception Safety - Strong Guarantee
        std::vector<uint32_t> cpp_spirv;
        cpp_spirv.reserve(spirv.size()); // Pre-allocate for safety

        // Pattern 3: Safe Memory Copy
        std::copy(spirv.begin(), spirv.end(), std::back_inserter(cpp_spirv));

        // Pattern 4: Default Construction with Safe Values
        std::vector<std::shared_ptr<kp::Tensor>> tensors; // Empty by design
        kp::Workgroup workgroup{1, 1, 1}; // Safe default workgroup size
        std::vector<float> spec_constants; // Empty by design
        std::vector<float> push_constants; // Empty by design

        // Pattern 5: RAII - Automatic cleanup if algorithm creation fails
        auto algorithm = manager.algorithm(tensors, cpp_spirv, workgroup,
                                         spec_constants, push_constants);

        if (!algorithm) {
            throw std::runtime_error("Manager returned null algorithm");
        }

        // Pattern 6: Atomic Operation Registration
        auto handle = get_global_registry().register_algorithm(algorithm);

        if (!validation::is_valid_handle(handle)) {
            throw std::runtime_error("Registry returned invalid handle");
        }

        return handle;

    } catch (const std::exception& e) {
        // Pattern 7: Exception Translation with Context
        throw std::runtime_error(std::string("Failed to create algorithm handle: ") + e.what());
    }
}

void algorithm_destroy_handle(uint64_t handle) {
    get_global_registry().remove_algorithm(handle);
}

// Sequence operations by handle
uint64_t create_sequence_handle(Manager& manager) {
    try {
        auto sequence = manager.sequence();
        return get_global_registry().register_sequence(sequence);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create sequence handle: ") + e.what());
    }
}

void sequence_record_tensor_sync_device_by_handles(uint64_t seq_handle, uint64_t tensor_handle) {
    try {
        auto sequence = get_global_registry().get_sequence(seq_handle);
        if (!sequence) {
            throw std::runtime_error("Invalid sequence handle");
        }

        auto tensor = get_global_registry().get_tensor(tensor_handle);
        if (!tensor) {
            throw std::runtime_error("Invalid tensor handle");
        }

        // Record sync device operation
        sequence->record<kp::OpTensorSyncDevice>({tensor});
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to record tensor sync device by handles: ") + e.what());
    }
}

void sequence_record_tensor_sync_local_by_handles(uint64_t seq_handle, uint64_t tensor_handle) {
    try {
        auto sequence = get_global_registry().get_sequence(seq_handle);
        if (!sequence) {
            throw std::runtime_error("Invalid sequence handle");
        }

        auto tensor = get_global_registry().get_tensor(tensor_handle);
        if (!tensor) {
            throw std::runtime_error("Invalid tensor handle");
        }

        // Record sync local operation
        sequence->record<kp::OpTensorSyncLocal>({tensor});
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to record tensor sync local by handles: ") + e.what());
    }
}

void sequence_record_algorithm_dispatch_by_handles(uint64_t seq_handle, uint64_t algo_handle) {
    try {
        auto sequence = get_global_registry().get_sequence(seq_handle);
        if (!sequence) {
            throw std::runtime_error("Invalid sequence handle");
        }

        auto algorithm = get_global_registry().get_algorithm(algo_handle);
        if (!algorithm) {
            throw std::runtime_error("Invalid algorithm handle");
        }

        // Record algorithm dispatch operation
        sequence->record<kp::OpAlgoDispatch>(algorithm);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to record algorithm dispatch by handles: ") + e.what());
    }
}

void sequence_eval_by_handle(uint64_t handle) {
    try {
        auto sequence = get_global_registry().get_sequence(handle);
        if (!sequence) {
            throw std::runtime_error("Invalid sequence handle");
        }

        sequence->eval();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to evaluate sequence by handle: ") + e.what());
    }
}

void sequence_eval_async_by_handle(uint64_t handle) {
    try {
        auto sequence = get_global_registry().get_sequence(handle);
        if (!sequence) {
            throw std::runtime_error("Invalid sequence handle");
        }

        sequence->evalAsync();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to evaluate sequence async by handle: ") + e.what());
    }
}

void sequence_eval_await_by_handle(uint64_t handle, uint64_t timeout_ms) {
    try {
        auto sequence = get_global_registry().get_sequence(handle);
        if (!sequence) {
            throw std::runtime_error("Invalid sequence handle");
        }

        if (timeout_ms == 0) {
            // Default timeout (UINT64_MAX)
            sequence->evalAwait();
        } else {
            sequence->evalAwait(timeout_ms);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to await sequence by handle: ") + e.what());
    }
}

void sequence_destroy_handle(uint64_t handle) {
    get_global_registry().remove_sequence(handle);
}
