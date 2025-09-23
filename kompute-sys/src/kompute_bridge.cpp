#include "kompute_bridge.hpp"
#include "handle_registry.hpp"
#include "platform_detection.hpp"
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <vector>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include <mutex>
#include <atomic>

// Thread-safe environment configuration to prevent race conditions in async contexts
static std::once_flag vulkan_env_configured;

// Intelligent Vulkan environment configuration with platform detection
void configure_vulkan_environment() {
    // Ensure environment is configured only once, thread-safely
    std::call_once(vulkan_env_configured, []() {
        using namespace platform_detection;

        // Load platform configuration from environment
        auto config = load_platform_config();

        // Detect platform capabilities
        auto platform_info = detect_platform();

        // Make intelligent decisions based on platform and user preferences
        auto decision = make_platform_decision(platform_info, config);

        // Print detection results if verbose mode enabled
        print_platform_info(platform_info, decision, config.verbose_detection);

        // Configure MoltenVK if decided to use it
        if (decision.use_moltenvk) {
            const char* vulkan_sdk = std::getenv("VULKAN_SDK");
            if (vulkan_sdk) {
                // Check if VK_ICD_FILENAMES is already set and points to an existing file
                const char* existing_icd = std::getenv("VK_ICD_FILENAMES");
                if (existing_icd && access(existing_icd, R_OK) == 0) {
                    // Already correctly set, don't override
                    std::cerr << "🔧 Using existing VK_ICD_FILENAMES: " << existing_icd << std::endl;
                } else {
                    // Try common locations for MoltenVK ICD
                    std::vector<std::string> icd_candidates = {
                        std::string(vulkan_sdk) + "/etc/vulkan/icd.d/MoltenVK_icd.json",
                        std::string(vulkan_sdk) + "/share/vulkan/icd.d/MoltenVK_icd.json"
                    };

                    for (const auto& icd_path : icd_candidates) {
                        if (access(icd_path.c_str(), R_OK) == 0) {
                            setenv("VK_ICD_FILENAMES", icd_path.c_str(), 1);
                            std::cerr << "🔧 Set VK_ICD_FILENAMES: " << icd_path << std::endl;
                            break;
                        }
                    }
                }
            }
        }

        // Configure validation layers based on decision - simple and direct
        if (decision.use_validation) {
            // Only set layer path, don't set VK_INSTANCE_LAYERS (let Kompute decide)
            const char* vulkan_sdk = std::getenv("VULKAN_SDK");
            if (vulkan_sdk) {
                const char* existing_layer = std::getenv("VK_LAYER_PATH");
                if (!existing_layer) {
                    std::string layer_path = std::string(vulkan_sdk) + "/share/vulkan/explicit_layer.d";
                    if (access(layer_path.c_str(), R_OK) == 0) {
                        setenv("VK_LAYER_PATH", layer_path.c_str(), 1);
                        std::cerr << "🔧 Set VK_LAYER_PATH for validation layers: " << layer_path << std::endl;
                    }
                }
            }
            std::cerr << "🔧 Validation layers available - letting Kompute decide usage" << std::endl;
        } else {
            // Ensure validation layers are disabled - clear ALL possible layer environment variables
            unsetenv("VK_INSTANCE_LAYERS");
            unsetenv("VK_LAYER_PATH");
            unsetenv("VK_LOADER_LAYERS_ENABLE");
            unsetenv("VK_LAYER_ENABLES");
            unsetenv("VK_LOADER_LAYERS_DISABLE");
            unsetenv("VK_LAYER_SETTINGS_PATH");
            unsetenv("VK_KHRONOS_VALIDATION_LAYER");
            std::cerr << "🔧 Validation layers disabled - comprehensive environment cleanup" << std::endl;
        }

        // Print any warnings
        for (const auto& warning : decision.warnings) {
            std::cerr << warning << std::endl;
        }
    }); // End of std::call_once lambda
}

// OPTION 2: Custom Manager creation with PR #353 MoltenVK compatibility fix
std::unique_ptr<Manager> create_manager_moltenvk_compatible(uint32_t device_id) {
    try {
        std::cerr << "🔧 OPTION 2: Creating MoltenVK-compatible Manager with PR #353 fixes" << std::endl;

        // Create Vulkan instance using the exact same fix from PR #353
        vk::ApplicationInfo applicationInfo;
        applicationInfo.pApplicationName = "Kompute";
        applicationInfo.pEngineName = "Kompute";
        applicationInfo.apiVersion = VK_MAKE_VERSION(1, 2, 0);  // Force API 1.2!
        applicationInfo.engineVersion = VK_MAKE_VERSION(1, 2, 0);
        applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 2, 0);

        // PR #353 FIX: Apply the exact same changes as in Manager.cpp
        std::vector<const char*> applicationExtensions;
        vk::InstanceCreateInfo computeInstanceCreateInfo;
        computeInstanceCreateInfo.pApplicationInfo = &applicationInfo;

        // Intelligent platform-specific extension configuration
        auto config = platform_detection::load_platform_config();
        auto platform_info = platform_detection::detect_platform();
        auto decision = platform_detection::make_platform_decision(platform_info, config);

        if (decision.use_portability_extension) {
            applicationExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            computeInstanceCreateInfo.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
            std::cerr << "🔧 Added VK_KHR_PORTABILITY_ENUMERATION_EXTENSION (" << decision.reasoning << ")" << std::endl;
        }

        computeInstanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(applicationExtensions.size());
        computeInstanceCreateInfo.ppEnabledExtensionNames = applicationExtensions.empty() ? nullptr : applicationExtensions.data();

        std::cerr << "🔧 Creating Vulkan instance with PR #353 compatibility fixes..." << std::endl;

        // PR #353 FIX: Add error handling for Vulkan instance creation
        std::cerr << "🔧 About to create Vulkan instance..." << std::endl;
        auto instance = std::make_shared<vk::Instance>(vk::createInstance(computeInstanceCreateInfo));
        std::cerr << "🔧 Vulkan instance created successfully!" << std::endl;

        // Enumerate physical devices
        std::cerr << "🔧 Enumerating physical devices..." << std::endl;
        std::vector<vk::PhysicalDevice> physicalDevices = instance->enumeratePhysicalDevices();

        if (physicalDevices.empty()) {
            throw std::runtime_error("No Vulkan physical devices found");
        }

        if (device_id >= physicalDevices.size()) {
            throw std::runtime_error("Device ID " + std::to_string(device_id) + " out of range");
        }

        auto physicalDevice = std::make_shared<vk::PhysicalDevice>(physicalDevices[device_id]);

        // Get queue family properties
        std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice->getQueueFamilyProperties();

        uint32_t computeQueueFamilyIndex = UINT32_MAX;
        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute) {
                computeQueueFamilyIndex = i;
                break;
            }
        }

        if (computeQueueFamilyIndex == UINT32_MAX) {
            throw std::runtime_error("No compute queue family found");
        }

        // Create logical device
        std::cerr << "🔧 Creating logical device..." << std::endl;
        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo;
        queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        vk::DeviceCreateInfo deviceCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

        auto device = std::make_shared<vk::Device>(physicalDevice->createDevice(deviceCreateInfo));

        // Create Manager with our custom Vulkan objects
        std::cerr << "🔧 Creating Manager with custom Vulkan objects using PR #353 fixes..." << std::endl;
        auto manager = std::make_unique<kp::Manager>(instance, physicalDevice, device);

        std::cerr << "✅ OPTION 2: MoltenVK-compatible Manager created successfully with PR #353 fixes!" << std::endl;
        return manager;

    } catch (const std::exception& e) {
        std::cerr << "❌ OPTION 2 failed: " << e.what() << std::endl;
        throw;
    }
}

// Manager operations
std::unique_ptr<Manager> create_manager(uint32_t device_id) {
    try {
        std::cerr << "DEBUG: Starting manager creation for device " << device_id << std::endl;

        // Configure Vulkan environment first (critical for validation layers)
        configure_vulkan_environment();

        // Apply PR #353 and #380 fixes for MoltenVK compatibility
        // Intelligent MoltenVK configuration based on platform detection
        auto config = platform_detection::load_platform_config();
        auto platform_info = platform_detection::detect_platform();
        auto decision = platform_detection::make_platform_decision(platform_info, config);

        if (decision.use_moltenvk) {
            std::cerr << "🔧 Applying MoltenVK compatibility fixes (" << decision.reasoning << ")" << std::endl;

            // Set MoltenVK environment variables for API 1.2 compatibility
            setenv("MVK_VULKAN_API_VERSION_MAJOR", "1", 1);
            setenv("MVK_VULKAN_API_VERSION_MINOR", "2", 1);
            setenv("MVK_VULKAN_API_VERSION", "1.2.0", 1);
            setenv("MVK_CONFIG_API_VERSION_TO_ADVERTISE", "1.2.0", 1);

            std::cerr << "🔧 Set MoltenVK environment variables for API 1.2 compatibility" << std::endl;
        } else {
            std::cerr << "🔧 MoltenVK disabled (" << decision.reasoning << ")" << std::endl;
        }

        // Note: PR #353 fixes are applied in Kompute's internal Manager constructor
        // via the KOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK flag we set in build.rs
        std::cerr << "🔧 Using default Manager constructor with version check configuration" << std::endl;

        // Create Manager with properly configured environment
        std::cerr << "DEBUG: About to create kp::Manager with default constructor" << std::endl;
        auto manager = std::make_unique<kp::Manager>();
        std::cerr << "✅ kp::Manager constructor completed successfully" << std::endl;

        // Verify the manager was created successfully by testing a basic operation
        if (!manager) {
            throw std::runtime_error("Manager creation returned null pointer");
        }

        std::cerr << "DEBUG: About to test manager functionality" << std::endl;

        // Test if manager is functional by trying to list devices
        try {
            auto devices = manager->listDevices();
            std::cerr << "DEBUG: listDevices() completed, found " << devices.size() << " devices" << std::endl;
            if (devices.empty()) {
                throw std::runtime_error("Manager created but found no devices");
            }
        } catch (const std::exception& e) {
            std::cerr << "DEBUG: listDevices() failed: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Manager created but not functional: ") + e.what());
        }

        std::cerr << "DEBUG: Manager creation successful" << std::endl;
        return manager;
    } catch (const std::exception& e) {
        std::cerr << "DEBUG: Manager creation failed: " << e.what() << std::endl;
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
        appInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // Intelligent portability extension configuration
        std::vector<const char*> extensions;
        auto config = platform_detection::load_platform_config();
        auto platform_info = platform_detection::detect_platform();
        auto decision = platform_detection::make_platform_decision(platform_info, config);

        if (decision.use_portability_extension) {
            extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            createInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            std::cerr << "🔧 Device enumeration using portability extension (" << decision.reasoning << ")" << std::endl;
        }

        // Configure validation layers if enabled and available
        std::vector<const char*> layers;
        if (decision.use_validation) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            std::cerr << "🔧 Device enumeration using validation layers" << std::endl;
        }

        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
        createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
        createInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();

        VkInstance instance;
        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
        if (result != VK_SUCCESS) {
            // If validation layers were requested but failed, try without them
            if (decision.use_validation && result == VK_ERROR_LAYER_NOT_PRESENT) {
                std::cerr << "⚠️  Validation layers requested but not available, retrying without them..." << std::endl;

                // Retry without validation layers
                createInfo.enabledLayerCount = 0;
                createInfo.ppEnabledLayerNames = nullptr;

                result = vkCreateInstance(&createInfo, nullptr, &instance);
                if (result == VK_SUCCESS) {
                    std::cerr << "✅ Vulkan instance created successfully without validation layers" << std::endl;
                } else {
                    std::cerr << "❌ Failed to create Vulkan instance even without validation layers: " << result << std::endl;
                    return -1;
                }
            } else {
                std::cerr << "❌ Failed to create Vulkan instance for device enumeration: " << result << std::endl;
                return -1;
            }
        } else if (decision.use_validation) {
            std::cerr << "✅ Vulkan instance created successfully with validation layers" << std::endl;
        }

        // Enumerate physical devices
        uint32_t deviceCount = 0;
        result = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        vkDestroyInstance(instance, nullptr);

        if (result != VK_SUCCESS) {
            std::cerr << "Failed to enumerate physical devices: " << result << std::endl;
            return -1;
        }

        return static_cast<int32_t>(deviceCount);
    } catch (const std::exception& e) {
        std::cerr << "Exception in get_device_count: " << e.what() << std::endl;
        return -1;
    }
}

// Tensor operations
std::shared_ptr<Tensor> create_tensor_f32(Manager& manager, rust::Slice<const float> data) {
    try {
        std::vector<float> cpp_data(data.begin(), data.end());
        return manager.tensorT<float>(cpp_data);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create tensor: ") + e.what());
    }
}

std::shared_ptr<Tensor> create_tensor_uninit_f32(Manager& manager, size_t size) {
    try {
        std::vector<float> empty_data(size, 0.0f);
        return manager.tensorT<float>(empty_data);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create uninitialized tensor: ") + e.what());
    }
}

// Multi-type tensor creation functions
std::shared_ptr<Tensor> create_tensor_f64(Manager& manager, rust::Slice<const double> data) {
    try {
        std::vector<double> cpp_data(data.begin(), data.end());
        return manager.tensorT<double>(cpp_data);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create f64 tensor: ") + e.what());
    }
}

std::shared_ptr<Tensor> create_tensor_i32(Manager& manager, rust::Slice<const int32_t> data) {
    try {
        std::vector<int32_t> cpp_data(data.begin(), data.end());
        return manager.tensorT<int32_t>(cpp_data);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create i32 tensor: ") + e.what());
    }
}

std::shared_ptr<Tensor> create_tensor_u32(Manager& manager, rust::Slice<const uint32_t> data) {
    try {
        std::vector<uint32_t> cpp_data(data.begin(), data.end());
        return manager.tensorT<uint32_t>(cpp_data);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create u32 tensor: ") + e.what());
    }
}


// Uninitialized multi-type tensors
std::shared_ptr<Tensor> create_tensor_uninit_f64(Manager& manager, size_t size) {
    try {
        std::vector<double> empty_data(size, 0.0);
        return manager.tensorT<double>(empty_data);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create uninitialized f64 tensor: ") + e.what());
    }
}

std::shared_ptr<Tensor> create_tensor_uninit_i32(Manager& manager, size_t size) {
    try {
        std::vector<int32_t> empty_data(size, 0);
        return manager.tensorT<int32_t>(empty_data);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create uninitialized i32 tensor: ") + e.what());
    }
}

std::shared_ptr<Tensor> create_tensor_uninit_u32(Manager& manager, size_t size) {
    try {
        std::vector<uint32_t> empty_data(size, 0u);
        return manager.tensorT<uint32_t>(empty_data);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create uninitialized u32 tensor: ") + e.what());
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

        // Create empty memory objects vector for now - this would be properly parameterized in real usage
        std::vector<std::shared_ptr<kp::Memory>> memObjects;

        // Default workgroup and constants
        kp::Workgroup workgroup{1, 1, 1};
        std::vector<float> spec_constants;
        std::vector<float> push_constants;

        return manager.algorithm(memObjects, cpp_spirv, workgroup, spec_constants, push_constants);
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
        auto tensor = manager.tensorT<float>(cpp_data);

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
        auto tensor = manager.tensorT<float>(empty_data);

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

// =============================================================================
// Multi-Type Tensor Handle Operations Implementation
// =============================================================================

uint64_t create_tensor_f64_handle(Manager& manager, rust::Slice<const double> data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot create tensor with empty data");
    }

    if (!validation::is_safe_size(data.size(), safety_limits::MAX_TENSOR_SIZE)) {
        throw std::invalid_argument("Tensor size exceeds safety limits: " +
                                  std::to_string(data.size()) + " elements");
    }

    try {
        std::vector<double> cpp_data;
        cpp_data.reserve(data.size());
        std::copy(data.begin(), data.end(), std::back_inserter(cpp_data));

        auto tensor = manager.tensorT<double>(cpp_data);
        if (!tensor) {
            throw std::runtime_error("Manager returned null tensor");
        }

        auto handle = get_global_registry().register_tensor(tensor);
        if (!validation::is_valid_handle(handle)) {
            throw std::runtime_error("Registry returned invalid handle");
        }

        return handle;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create f64 tensor handle: ") + e.what());
    }
}

uint64_t create_tensor_i32_handle(Manager& manager, rust::Slice<const int32_t> data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot create tensor with empty data");
    }

    if (!validation::is_safe_size(data.size(), safety_limits::MAX_TENSOR_SIZE)) {
        throw std::invalid_argument("Tensor size exceeds safety limits: " +
                                  std::to_string(data.size()) + " elements");
    }

    try {
        std::vector<int32_t> cpp_data;
        cpp_data.reserve(data.size());
        std::copy(data.begin(), data.end(), std::back_inserter(cpp_data));

        auto tensor = manager.tensorT<int32_t>(cpp_data);
        if (!tensor) {
            throw std::runtime_error("Manager returned null tensor");
        }

        auto handle = get_global_registry().register_tensor(tensor);
        if (!validation::is_valid_handle(handle)) {
            throw std::runtime_error("Registry returned invalid handle");
        }

        return handle;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create i32 tensor handle: ") + e.what());
    }
}

uint64_t create_tensor_u32_handle(Manager& manager, rust::Slice<const uint32_t> data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot create tensor with empty data");
    }

    if (!validation::is_safe_size(data.size(), safety_limits::MAX_TENSOR_SIZE)) {
        throw std::invalid_argument("Tensor size exceeds safety limits: " +
                                  std::to_string(data.size()) + " elements");
    }

    try {
        std::vector<uint32_t> cpp_data;
        cpp_data.reserve(data.size());
        std::copy(data.begin(), data.end(), std::back_inserter(cpp_data));

        auto tensor = manager.tensorT<uint32_t>(cpp_data);
        if (!tensor) {
            throw std::runtime_error("Manager returned null tensor");
        }

        auto handle = get_global_registry().register_tensor(tensor);
        if (!validation::is_valid_handle(handle)) {
            throw std::runtime_error("Registry returned invalid handle");
        }

        return handle;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create u32 tensor handle: ") + e.what());
    }
}


// =============================================================================
// Uninitialized Multi-Type Tensor Handle Operations
// =============================================================================

uint64_t create_tensor_uninit_f64_handle(Manager& manager, size_t size) {
    if (size == 0) {
        throw std::invalid_argument("Cannot create tensor with zero size");
    }

    if (!validation::is_safe_size(size, safety_limits::MAX_TENSOR_SIZE)) {
        throw std::invalid_argument("Tensor size exceeds safety limits: " +
                                  std::to_string(size) + " elements");
    }

    try {
        std::vector<double> empty_data(size, 0.0);
        auto tensor = manager.tensorT<double>(empty_data);
        if (!tensor) {
            throw std::runtime_error("Manager returned null tensor");
        }

        auto handle = get_global_registry().register_tensor(tensor);
        if (!validation::is_valid_handle(handle)) {
            throw std::runtime_error("Registry returned invalid handle");
        }

        return handle;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create f64 uninit tensor handle: ") + e.what());
    }
}

uint64_t create_tensor_uninit_i32_handle(Manager& manager, size_t size) {
    if (size == 0) {
        throw std::invalid_argument("Cannot create tensor with zero size");
    }

    if (!validation::is_safe_size(size, safety_limits::MAX_TENSOR_SIZE)) {
        throw std::invalid_argument("Tensor size exceeds safety limits: " +
                                  std::to_string(size) + " elements");
    }

    try {
        std::vector<int32_t> empty_data(size, 0);
        auto tensor = manager.tensorT<int32_t>(empty_data);
        if (!tensor) {
            throw std::runtime_error("Manager returned null tensor");
        }

        auto handle = get_global_registry().register_tensor(tensor);
        if (!validation::is_valid_handle(handle)) {
            throw std::runtime_error("Registry returned invalid handle");
        }

        return handle;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create i32 uninit tensor handle: ") + e.what());
    }
}

uint64_t create_tensor_uninit_u32_handle(Manager& manager, size_t size) {
    if (size == 0) {
        throw std::invalid_argument("Cannot create tensor with zero size");
    }

    if (!validation::is_safe_size(size, safety_limits::MAX_TENSOR_SIZE)) {
        throw std::invalid_argument("Tensor size exceeds safety limits: " +
                                  std::to_string(size) + " elements");
    }

    try {
        std::vector<uint32_t> empty_data(size, 0u);
        auto tensor = manager.tensorT<uint32_t>(empty_data);
        if (!tensor) {
            throw std::runtime_error("Manager returned null tensor");
        }

        auto handle = get_global_registry().register_tensor(tensor);
        if (!validation::is_valid_handle(handle)) {
            throw std::runtime_error("Registry returned invalid handle");
        }

        return handle;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create u32 uninit tensor handle: ") + e.what());
    }
}

// =============================================================================
// Filled Tensor Creation Functions
// =============================================================================

uint64_t create_tensor_filled_f32_handle(Manager& manager, size_t size, float fill_value) {
    if (size == 0) {
        throw std::invalid_argument("Cannot create tensor with zero size");
    }

    if (!validation::is_safe_size(size, safety_limits::MAX_TENSOR_SIZE)) {
        throw std::invalid_argument("Tensor size exceeds safety limits: " +
                                  std::to_string(size) + " elements");
    }

    try {
        std::vector<float> filled_data(size, fill_value);
        auto tensor = manager.tensorT<float>(filled_data);
        if (!tensor) {
            throw std::runtime_error("Manager returned null tensor");
        }

        auto handle = get_global_registry().register_tensor(tensor);
        if (!validation::is_valid_handle(handle)) {
            throw std::runtime_error("Registry returned invalid handle");
        }

        return handle;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create f32 filled tensor handle: ") + e.what());
    }
}

uint64_t create_tensor_filled_f64_handle(Manager& manager, size_t size, double fill_value) {
    if (size == 0) {
        throw std::invalid_argument("Cannot create tensor with zero size");
    }

    if (!validation::is_safe_size(size, safety_limits::MAX_TENSOR_SIZE)) {
        throw std::invalid_argument("Tensor size exceeds safety limits: " +
                                  std::to_string(size) + " elements");
    }

    try {
        std::vector<double> filled_data(size, fill_value);
        auto tensor = manager.tensorT<double>(filled_data);
        if (!tensor) {
            throw std::runtime_error("Manager returned null tensor");
        }

        auto handle = get_global_registry().register_tensor(tensor);
        if (!validation::is_valid_handle(handle)) {
            throw std::runtime_error("Registry returned invalid handle");
        }

        return handle;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create f64 filled tensor handle: ") + e.what());
    }
}

uint64_t create_tensor_filled_i32_handle(Manager& manager, size_t size, int32_t fill_value) {
    if (size == 0) {
        throw std::invalid_argument("Cannot create tensor with zero size");
    }

    if (!validation::is_safe_size(size, safety_limits::MAX_TENSOR_SIZE)) {
        throw std::invalid_argument("Tensor size exceeds safety limits: " +
                                  std::to_string(size) + " elements");
    }

    try {
        std::vector<int32_t> filled_data(size, fill_value);
        auto tensor = manager.tensorT<int32_t>(filled_data);
        if (!tensor) {
            throw std::runtime_error("Manager returned null tensor");
        }

        auto handle = get_global_registry().register_tensor(tensor);
        if (!validation::is_valid_handle(handle)) {
            throw std::runtime_error("Registry returned invalid handle");
        }

        return handle;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create i32 filled tensor handle: ") + e.what());
    }
}

uint64_t create_tensor_filled_u32_handle(Manager& manager, size_t size, uint32_t fill_value) {
    if (size == 0) {
        throw std::invalid_argument("Cannot create tensor with zero size");
    }

    if (!validation::is_safe_size(size, safety_limits::MAX_TENSOR_SIZE)) {
        throw std::invalid_argument("Tensor size exceeds safety limits: " +
                                  std::to_string(size) + " elements");
    }

    try {
        std::vector<uint32_t> filled_data(size, fill_value);
        auto tensor = manager.tensorT<uint32_t>(filled_data);
        if (!tensor) {
            throw std::runtime_error("Manager returned null tensor");
        }

        auto handle = get_global_registry().register_tensor(tensor);
        if (!validation::is_valid_handle(handle)) {
            throw std::runtime_error("Registry returned invalid handle");
        }

        return handle;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create u32 filled tensor handle: ") + e.what());
    }
}

// =============================================================================
// Multi-Type Tensor Data Retrieval Functions
// =============================================================================

rust::Vec<double> tensor_data_f64_by_handle(uint64_t handle) {
    try {
        auto tensor = get_global_registry().get_tensor(handle);
        if (!tensor) {
            return rust::Vec<double>();
        }

        auto cpp_data = tensor->vector<double>();
        rust::Vec<double> rust_data;
        rust_data.reserve(cpp_data.size());
        for (const auto& item : cpp_data) {
            rust_data.push_back(item);
        }
        return rust_data;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to get f64 tensor data by handle: ") + e.what());
    }
}

rust::Vec<int32_t> tensor_data_i32_by_handle(uint64_t handle) {
    try {
        auto tensor = get_global_registry().get_tensor(handle);
        if (!tensor) {
            return rust::Vec<int32_t>();
        }

        auto cpp_data = tensor->vector<int32_t>();
        rust::Vec<int32_t> rust_data;
        rust_data.reserve(cpp_data.size());
        for (const auto& item : cpp_data) {
            rust_data.push_back(item);
        }
        return rust_data;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to get i32 tensor data by handle: ") + e.what());
    }
}

rust::Vec<uint32_t> tensor_data_u32_by_handle(uint64_t handle) {
    try {
        auto tensor = get_global_registry().get_tensor(handle);
        if (!tensor) {
            return rust::Vec<uint32_t>();
        }

        auto cpp_data = tensor->vector<uint32_t>();
        rust::Vec<uint32_t> rust_data;
        rust_data.reserve(cpp_data.size());
        for (const auto& item : cpp_data) {
            rust_data.push_back(item);
        }
        return rust_data;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to get u32 tensor data by handle: ") + e.what());
    }
}

uint32_t tensor_data_type_by_handle(uint64_t handle) {
    try {
        auto tensor = get_global_registry().get_tensor(handle);
        if (!tensor) {
            throw std::runtime_error("Invalid tensor handle: " + std::to_string(handle));
        }

        // Get the tensor's data type through Kompute's type system
        // Since Kompute tensors are templated, we need to determine the type at runtime
        // This is a simplified approach - in a full implementation, we'd store type info in the registry

        // For now, we'll use memory size and element count to infer the type
        size_t memory_size = tensor->memorySize();
        size_t element_count = tensor->size();

        if (element_count == 0) {
            return 0; // Unknown type for empty tensor
        }

        size_t bytes_per_element = memory_size / element_count;

        // Map bytes per element to data types
        // 0: Unknown, 1: F32, 2: F64, 3: I32, 4: U32, 5: I8, 6: U8
        switch (bytes_per_element) {
            case 4:
                // Could be f32, i32, or u32 - default to f32 for backward compatibility
                return 1; // F32
            case 8:
                return 2; // F64
            case 1:
                return 5; // I8 (could also be U8, but we'll default to I8)
            default:
                return 0; // Unknown
        }

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to get tensor data type by handle: ") + e.what());
    }
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
        std::vector<std::shared_ptr<kp::Memory>> memObjects; // Empty by design
        kp::Workgroup workgroup{1, 1, 1}; // Safe default workgroup size
        std::vector<float> spec_constants; // Empty by design
        std::vector<float> push_constants; // Empty by design

        // Pattern 5: RAII - Automatic cleanup if algorithm creation fails
        auto algorithm = manager.algorithm(memObjects, cpp_spirv, workgroup,
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

bool algorithm_set_specialization_constants(uint64_t handle, rust::Slice<const float> /* constants */) {
    try {
        auto algorithm = get_global_registry().get_algorithm(handle);
        if (!algorithm) {
            std::cerr << "Invalid algorithm handle: " << handle << std::endl;
            return false;
        }

        // Note: Kompute Algorithm doesn't currently support setting specialization constants
        // after creation. This would need to be implemented in the Kompute C++ library.
        // For now, we'll store the constants but they won't be used until Kompute supports it.

        std::cerr << "Warning: Specialization constants not yet supported by Kompute C++ library" << std::endl;
        return false;

    } catch (const std::exception& e) {
        std::cerr << "Failed to set specialization constants: " << e.what() << std::endl;
        return false;
    }
}

bool algorithm_set_push_constants(uint64_t handle, rust::Slice<const float> constants) {
    try {
        auto algorithm = get_global_registry().get_algorithm(handle);
        if (!algorithm) {
            std::cerr << "Invalid algorithm handle: " << handle << std::endl;
            return false;
        }

        // Convert Rust slice to C++ vector
        std::vector<float> push_constants(constants.begin(), constants.end());

        // Use the correct Kompute setPushConstants signature
        uint32_t size = static_cast<uint32_t>(push_constants.size());
        uint32_t memory_size = size * sizeof(float);

        // Cast to void* as required by Kompute API
        algorithm->setPushConstants(
            static_cast<void*>(push_constants.data()),
            size,
            memory_size
        );

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to set push constants: " << e.what() << std::endl;
        return false;
    }
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
        sequence->record<kp::OpSyncDevice>({tensor});
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
        sequence->record<kp::OpSyncLocal>({tensor});
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

// =============================================================================
// Queue Management Operations Implementation
// =============================================================================

uint64_t create_queue_handle(Manager& /* manager */, uint32_t queue_family_index) {
    try {
        // For now, we'll create a wrapper around Kompute's primary queue
        // In a full implementation, this would create queues from the manager's device

        // Get the manager's device to access queue families
        // This is a simplified implementation that assumes the manager has access to its device

        // For the current Kompute API, we'll create a conceptual queue handle
        // that represents the manager's primary queue family
        auto queue_wrapper = std::make_shared<QueueHandle>(
            VK_NULL_HANDLE, // We don't have direct access to VkQueue from Kompute Manager
            queue_family_index,
            "primary_queue_" + std::to_string(queue_family_index)
        );

        return get_global_registry().register_queue(queue_wrapper);

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create queue handle: ") + e.what());
    }
}

uint64_t create_sequence_on_queue_handle(Manager& manager, uint64_t queue_handle) {
    try {
        // Validate queue handle exists
        auto queue = get_global_registry().get_queue(queue_handle);
        if (!queue) {
            throw std::runtime_error("Invalid queue handle: " + std::to_string(queue_handle));
        }

        // Create sequence using the manager (Kompute sequences are tied to the manager's primary queue)
        auto sequence = manager.sequence();
        if (!sequence) {
            throw std::runtime_error("Manager returned null sequence");
        }

        return get_global_registry().register_sequence(sequence);

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create sequence on queue: ") + e.what());
    }
}

void queue_submit_sequence(uint64_t queue_handle, uint64_t sequence_handle) {
    try {
        // Validate handles
        auto queue = get_global_registry().get_queue(queue_handle);
        if (!queue) {
            throw std::runtime_error("Invalid queue handle: " + std::to_string(queue_handle));
        }

        auto sequence = get_global_registry().get_sequence(sequence_handle);
        if (!sequence) {
            throw std::runtime_error("Invalid sequence handle: " + std::to_string(sequence_handle));
        }

        // For now, submit directly via sequence eval (Kompute handles queue submission internally)
        sequence->eval();

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to submit sequence to queue: ") + e.what());
    }
}

void queue_wait_idle(uint64_t queue_handle) {
    try {
        // Validate queue handle
        auto queue = get_global_registry().get_queue(queue_handle);
        if (!queue) {
            throw std::runtime_error("Invalid queue handle: " + std::to_string(queue_handle));
        }

        // For Kompute, we don't have direct queue access, so this is a no-op
        // In a full Vulkan implementation, this would call vkQueueWaitIdle

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to wait for queue idle: ") + e.what());
    }
}

void queue_destroy_handle(uint64_t handle) {
    try {
        get_global_registry().remove_queue(handle);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to destroy queue handle: ") + e.what());
    }
}
