# Komputers Detailed Implementation Plan

**Generated:** 2024-01-15
**Version:** 1.0
**Purpose:** Systematic, function-level implementation plan following CONTRIBUTING.md and GRIMOIRE.md patterns
**Framework:** Three Rings of Validation (Unit → Integration → Quality Gates)

---

## Implementation Methodology

Following CONTRIBUTING.md and GRIMOIRE.md, each work item MUST:

1. **MUST NOT BE STUBBED** - Full functional implementation required
2. **MUST BE FIXED WITHOUT WORKAROUNDS** - Proper solutions, not shortcuts
3. **MUST ASK FOR HELP** - If implementation requires workarounds, ask user for guidance
4. **Test-First Development** - Write failing tests before implementation
5. **Evidence-Based Completion** - Concrete proof of functionality required

Each phase includes:

- Function-level work items with full implementation requirements
- Phase completion: Update SYSTEM_ARCHITECTURE.md
- Phase completion: Add comprehensive API documentation comments

---

## PHASE 0: Test Infrastructure Completion (`kompute/tests/`)

### Critical Foundation Requirement

**Before implementing new features, comprehensive test coverage must be established** for all existing functionality. Current tests are incomplete with significant gaps in thread safety, type support, error conditions, and C++ memory safety validation.

### C++ Memory Sanitization Configuration

#### Required Cargo.toml additions:

```toml
[features]
default = []
async = ["tokio"]
asan = []
test-utils = []

[dev-dependencies]
tokio = { version = "1.0", features = ["full"], optional = true }
proptest = "1.0"
criterion = "0.5"
```

#### Required build.rs configuration:

```rust
// In kompute-sys/build.rs
fn main() {
    if cfg!(feature = "asan") {
        println!("cargo:rustc-link-arg=-fsanitize=address");
        println!("cargo:rustc-env=ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer");

        let mut build = cc::Build::new();
        build.flag("-fsanitize=address");
        build.flag("-fno-omit-frame-pointer");
        build.flag("-g");
    }
}
```

#### Required CI configuration (.github/workflows/sanitizer-tests.yml):

```yaml
name: Sanitizer Tests
on: [push, pull_request]

jobs:
  sanitizer-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: nightly
          components: rust-src
      - name: Install LLVM
        run: sudo apt-get install -y llvm
      - name: Run AddressSanitizer tests
        run: |
          export RUSTFLAGS="-Z sanitizer=address"
          cargo +nightly test --features asan --target x86_64-unknown-linux-gnu
```

### Work Items

#### 0.1: Manager Module Unit Test Completion

- **Function:** Complete missing unit tests for all Manager functionality
- **File:** `kompute/tests/manager_unit_tests.rs`
- **Requirements:**
  - MUST test `test_manager_thread_safety_concurrent_creation()` with 8+ threads
  - MUST test `test_manager_resource_limits_tensor_creation()` with 1000+ tensors
  - MUST test `test_manager_device_switching_all_available()` across all device IDs
  - MUST test `test_manager_memory_pressure_allocation_failure()` under low GPU memory
  - MUST test `test_manager_factory_methods_invalid_inputs()` for all factory methods
  - MUST NOT BE STUBBED - test actual GPU operations where applicable
  - MUST ASK FOR HELP if GPU hardware not available for required tests
- **Implementation Example:**

```rust
#[test]
fn test_manager_thread_safety_concurrent_creation() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let thread_count = 8;
    let barrier = Arc::new(Barrier::new(thread_count));
    let mut handles = vec![];

    for thread_id in 0..thread_count {
        let barrier_clone = Arc::clone(&barrier);
        let handle = thread::spawn(move || {
            barrier_clone.wait(); // Synchronize start
            let result = Manager::new();
            (thread_id, result.is_ok())
        });
        handles.push(handle);
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let success_count = results.iter().filter(|(_, success)| *success).count();
    assert!(success_count > 0, "No managers created successfully in concurrent test");
}
```

- **Tests Required:** 15+ comprehensive unit tests covering all Manager methods
- **Dependencies:** None

#### 0.2: Tensor Module Multi-Type Support Testing

- **Function:** Test all declared TensorData types (f32, f64, i32, u32, i8, u8)
- **File:** `kompute/tests/tensor_multitype_tests.rs`
- **Requirements:**
  - MUST test `test_tensor_f64_data_consistency()` with high precision values
  - MUST test `test_tensor_i32_integer_overflow_handling()` with MIN/MAX values
  - MUST test `test_tensor_u32_unsigned_wraparound()` with large unsigned values
  - MUST test `test_tensor_i8_u8_byte_alignment()` with memory alignment validation
  - MUST test `test_tensor_memory_alignment_requirements()` across all types
  - MUST test `test_tensor_handle_lifecycle_invalidation()` after Drop
  - MUST NOT BE STUBBED - test actual tensor creation for all types
  - MUST ASK FOR HELP if non-f32 types not yet implemented in C++ bridge
- **Implementation Example:**

```rust
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_tensor_f64_data_consistency() {
    if let Ok(mut manager) = Manager::new() {
        let original_data = vec![3.14159265358979323846f64, 2.71828182845904523536];
        match Tensor::new(&mut manager, &original_data) {
            Ok(tensor) => {
                if let Ok(retrieved) = tensor.to_vec() {
                    for (orig, retr) in original_data.iter().zip(retrieved.iter()) {
                        assert!((orig - retr).abs() < f64::EPSILON,
                               "f64 precision lost: {} != {}", orig, retr);
                    }
                }
            }
            Err(e) => {
                assert!(e.to_string().contains("f32") || e.to_string().contains("supported"),
                       "Error should mention type support limitation: {}", e);
            }
        }
    }
}
```

- **Tests Required:** 20+ tests covering all declared tensor types
- **Dependencies:** Work Item 0.1

#### 0.3: Algorithm Module SPIR-V Validation Testing

- **Function:** Comprehensive SPIR-V validation and algorithm lifecycle testing
- **File:** `kompute/tests/algorithm_spirv_tests.rs`
- **Requirements:**
  - MUST test `test_algorithm_spirv_corruption_detection()` with invalid SPIR-V patterns
  - MUST test `test_algorithm_tensor_binding_verification()` (will fail until Phase 2)
  - MUST test `test_algorithm_workgroup_specification_limits()` with GPU limits
  - MUST test `test_algorithm_multiple_creation_resource_limits()` with 100+ algorithms
  - MUST test `test_algorithm_reuse_different_tensors()` with algorithm rebinding
  - MUST NOT BE STUBBED - test actual SPIR-V parsing and GPU shader compilation
  - MUST ASK FOR HELP if SPIR-V validation requires external compiler tools
- **Implementation Example:**

```rust
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_algorithm_spirv_corruption_detection() {
    if let Ok(mut manager) = Manager::new() {
        let corruption_patterns = vec![
            vec![0x07230203, 0x00010000], // Truncated header
            vec![0x12345678, 0x00010000, 0x00080001, 0x0000000b, 0x00000000], // Wrong magic
            vec![0x07230203; 1000], // Extremely long but invalid
        ];

        for (i, corrupted_spirv) in corruption_patterns.into_iter().enumerate() {
            match Algorithm::new(&mut manager, &corrupted_spirv) {
                Ok(_) => panic!("Corrupted SPIR-V pattern {} should not succeed", i),
                Err(e) => {
                    assert!(e.to_string().contains("SPIR-V") || e.to_string().contains("shader"),
                           "Error should mention SPIR-V: {}", e);
                }
            }
        }
    }
}
```

- **Tests Required:** 25+ tests covering SPIR-V validation and algorithm configuration
- **Dependencies:** Work Item 0.1, 0.2

#### 0.4: Sequence Module Command Ordering Testing

- **Function:** Verify command recording, ordering, and execution correctness
- **File:** `kompute/tests/sequence_command_tests.rs`
- **Requirements:**
  - MUST test `test_sequence_command_execution_ordering()` with multiple operations
  - MUST test `test_sequence_complex_workflow_tensor_algorithm_chain()` end-to-end
  - MUST test `test_sequence_async_completion_timing()` with eval_async/eval_await
  - MUST test `test_sequence_error_recovery_partial_failure()` with invalid operations
  - MUST test `test_sequence_command_buffer_limits_maximum_operations()` with 1000+ ops
  - MUST NOT BE STUBBED - test actual GPU command submission and execution
  - MUST ASK FOR HELP if GPU command buffer limits cause issues
- **Implementation Example:**

```rust
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_sequence_command_execution_ordering() {
    if let Ok(mut manager) = Manager::new() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        if let Ok(tensor) = Tensor::new(&mut manager, &data) {
            if let Ok(mut sequence) = Sequence::new(&mut manager) {
                sequence.record_tensor_sync_device(tensor.handle()).unwrap();
                sequence.record_tensor_sync_local(tensor.handle()).unwrap();

                let start = std::time::Instant::now();
                match sequence.eval() {
                    Ok(()) => {
                        let duration = start.elapsed();
                        assert!(duration < std::time::Duration::from_secs(5),
                               "Sequence execution took too long: {:?}", duration);

                        if let Ok(retrieved) = tensor.to_vec() {
                            assert_eq!(data, retrieved, "Data corrupted during ordered execution");
                        }
                    }
                    Err(e) => println!("Sequence execution failed (may be expected): {}", e),
                }
            }
        }
    }
}
```

- **Tests Required:** 30+ tests covering all sequence operations and edge cases
- **Dependencies:** Work Item 0.1, 0.2, 0.3

#### 0.5: C++ Memory Sanitization Integration

- **Function:** Integrate AddressSanitizer and LeakSanitizer for C++ bridge code
- **File:** `kompute/tests/cpp_sanitization_tests.rs`
- **Requirements:**
  - MUST test `test_cpp_memory_leak_detection()` with AddressSanitizer subprocess execution
  - MUST test `test_cpp_use_after_free_detection()` with handle lifecycle validation
  - MUST test `test_cpp_buffer_overflow_detection()` with tensor data boundary checks
  - MUST test `test_handle_registry_memory_safety()` with registry operations
  - MUST test `test_vulkan_resource_cleanup_verification()` with GPU resource lifecycle
  - MUST NOT BE STUBBED - integrate real C++ sanitizers with actual detection
  - MUST ASK FOR HELP if sanitizer integration requires complex build configuration
- **Implementation Example:**

```rust
#[test]
fn test_cpp_memory_leak_detection() {
    if !cfg!(feature = "asan") {
        println!("Skipping C++ sanitization test - asan feature not enabled");
        return;
    }

    env::set_var("ASAN_OPTIONS", "detect_leaks=1:abort_on_error=1:print_stats=1");

    let output = Command::new("cargo")
        .args(&["test", "--features", "asan", "--", "--exact", "test_manager_creation"])
        .env("RUSTFLAGS", "-Z sanitizer=address")
        .output()
        .expect("Failed to execute sanitized test");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("ERROR: LeakSanitizer"),
           "Memory leak detected in C++ code:\n{}", stderr);
    assert!(!stderr.contains("ERROR: AddressSanitizer"),
           "Memory safety violation detected:\n{}", stderr);
}
```

- **Tests Required:** 10+ sanitization tests covering all C++ bridge operations
- **Dependencies:** Work Item 0.1, 0.2, 0.3, 0.4

#### 0.6: HandleTracker Module Statistics Verification

- **Function:** Verify thread-safe handle tracking accuracy and performance
- **File:** `kompute/tests/handle_tracker_tests.rs`
- **Requirements:**
  - MUST test `test_handle_tracker_statistics_accuracy()` with multiple categories
  - MUST test `test_handle_tracker_concurrent_register_unregister()` with 8+ threads
  - MUST test `test_handle_tracker_memory_leak_prevention()` with large handle counts
  - MUST test `test_handle_tracker_race_condition_safety()` with rapid register/unregister
  - MUST test `test_handle_tracker_large_scale_operations()` with 10000+ handles
  - MUST NOT BE STUBBED - test actual concurrent handle tracking operations
  - MUST ASK FOR HELP if thread safety verification requires additional synchronization
- **Implementation Example:**

```rust
#[test]
fn test_handle_tracker_statistics_accuracy() {
    let tracker = HandleTracker::new();
    let categories = vec!["tensors", "algorithms", "sequences"];
    let handles_per_category = vec![5, 3, 7];

    for (category, count) in categories.iter().zip(handles_per_category.iter()) {
        for i in 0..*count {
            let handle = tracker.generate_id();
            tracker.register_handle(category, handle);
        }
    }

    let stats = tracker.statistics();
    assert_eq!(stats.total_handles, 15, "Total handle count incorrect");

    for (category, expected_count) in categories.iter().zip(handles_per_category.iter()) {
        let actual_count = stats.categories.get(*category).unwrap_or(&0);
        assert_eq!(actual_count, expected_count,
                  "Handle count for category '{}' incorrect", category);
    }
}
```

- **Tests Required:** 15+ tests covering all HandleTracker functionality
- **Dependencies:** Work Item 0.1

#### 0.7: VulkanSetup Platform-Specific Testing

- **Function:** Test cross-platform Vulkan environment configuration
- **File:** `kompute/tests/vulkan_setup_platform_tests.rs`
- **Requirements:**
  - MUST test `test_vulkan_setup_macos_paths()` with SDK and homebrew paths
  - MUST test `test_vulkan_setup_linux_system_paths()` with system Vulkan locations
  - MUST test `test_vulkan_setup_windows_registry_fallback()` with registry detection
  - MUST test `test_vulkan_setup_missing_dependencies_handling()` without Vulkan installed
  - MUST test `test_vulkan_setup_corrupted_installation_detection()` with invalid paths
  - MUST NOT BE STUBBED - test actual platform filesystem and environment detection
  - MUST ASK FOR HELP if platform-specific detection requires OS-specific APIs
- **Implementation Example:**

```rust
#[cfg(target_os = "macos")]
#[test]
fn test_vulkan_setup_macos_paths() {
    let original_vulkan_sdk = env::var("VULKAN_SDK").ok();
    env::set_var("VULKAN_SDK", "/opt/vulkan-sdk");

    match crate::vulkan_setup::configure_vulkan_environment() {
        Ok(()) => {
            let icd_filenames = env::var("VK_ICD_FILENAMES");
            if let Ok(icd) = icd_filenames {
                assert!(icd.contains("MoltenVK"), "ICD path should reference MoltenVK: {}", icd);
                assert!(icd.contains(".json"), "ICD path should be JSON file: {}", icd);
            }
        }
        Err(e) => {
            println!("Vulkan setup failed (expected if no Vulkan installed): {}", e);
            assert!(!e.is_empty(), "Error message should not be empty");
        }
    }

    match original_vulkan_sdk {
        Some(original) => env::set_var("VULKAN_SDK", original),
        None => env::remove_var("VULKAN_SDK"),
    }
}
```

- **Tests Required:** 20+ tests covering all platform-specific configurations
- **Dependencies:** Work Item 0.1, 0.6

#### 0.8: Performance Regression Test Suite

- **Function:** Establish performance baselines for detecting regressions
- **File:** `kompute/tests/performance_regression_tests.rs`
- **Requirements:**
  - MUST test `test_performance_manager_creation_baseline()` with timing measurements
  - MUST test `test_performance_tensor_data_transfer_throughput()` with bandwidth metrics
  - MUST test `test_performance_sequence_execution_latency()` with GPU timing
  - MUST test `test_performance_algorithm_compilation_timing()` with SPIR-V compilation
  - MUST test `test_performance_memory_allocation_overhead()` with allocation timing
  - MUST NOT BE STUBBED - measure actual GPU performance with real operations
  - MUST ASK FOR HELP if GPU performance measurement requires specialized tools
- **Implementation Example:**

```rust
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_performance_manager_creation_baseline() {
    const ITERATIONS: usize = 10;
    const MAX_CREATION_TIME_MS: u64 = 1000;

    let mut creation_times = Vec::new();
    let mut successful_creations = 0;

    for i in 0..ITERATIONS {
        let start = Instant::now();
        match Manager::new() {
            Ok(_manager) => {
                let duration = start.elapsed();
                creation_times.push(duration);
                successful_creations += 1;
                assert!(duration < Duration::from_millis(MAX_CREATION_TIME_MS),
                       "Manager creation {} took too long: {:?}", i, duration);
            }
            Err(e) => println!("Manager creation {} failed: {}", i, e),
        }
    }

    if successful_creations > 0 {
        let avg_time = creation_times.iter().sum::<Duration>() / successful_creations as u32;
        println!("Average manager creation time: {:?}", avg_time);
        assert!(avg_time < Duration::from_millis(MAX_CREATION_TIME_MS / 2),
               "Average manager creation time regression: {:?}", avg_time);
    }
}
```

- **Tests Required:** 25+ performance tests establishing baselines for all operations
- **Dependencies:** Work Item 0.1, 0.2, 0.3, 0.4

#### 0.9: Comprehensive Error Path Coverage

- **Function:** Ensure all error conditions are testable and properly handled
- **File:** `kompute/tests/error_path_coverage_tests.rs`
- **Requirements:**
  - MUST test `test_all_error_variants_triggerable()` ensuring every KomputeError variant can be triggered
  - MUST test `test_error_context_preservation_through_call_stack()` with error chaining
  - MUST test `test_error_recovery_scenarios_resource_cleanup()` with partial failures
  - MUST test `test_error_message_quality_user_actionability()` with helpful error text
  - MUST test `test_error_chaining_source_error_preservation()` with std::error::Error integration
  - MUST NOT BE STUBBED - trigger actual error conditions, not synthetic errors
  - MUST ASK FOR HELP if error conditions cannot be reliably triggered
- **Implementation Example:**

```rust
#[test]
fn test_all_error_variants_triggerable() {
    let mut triggered_variants = std::collections::HashSet::new();

    // Test VulkanError
    match Manager::device_count() {
        Err(e) if matches!(e, KomputeError::VulkanError(_)) => {
            triggered_variants.insert("VulkanError");
        }
        _ => {}
    }

    // Test DeviceError
    match Manager::with_device(u32::MAX) {
        Err(e) if matches!(e, KomputeError::DeviceError(_)) => {
            triggered_variants.insert("DeviceError");
        }
        _ => {}
    }

    // Test TensorError
    if let Ok(mut manager) = Manager::new() {
        match Tensor::<f32>::new(&mut manager, &[]) {
            Err(e) if matches!(e, KomputeError::TensorError(_)) => {
                triggered_variants.insert("TensorError");
            }
            _ => {}
        }
    }

    println!("Triggered error variants: {:?}", triggered_variants);
    assert!(triggered_variants.len() >= 2,
           "Should be able to trigger multiple error conditions");
}
```

- **Tests Required:** 30+ tests covering all error paths and recovery scenarios
- **Dependencies:** Work Item 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8

---

## PHASE 1: Manager Multi-Queue Support (`kompute/src/manager.rs`)

### Critical Missing Functionality

The MultiQueueManager currently calls `self.manager.create_queue()` which **does not exist**. This phase implements the missing Manager queue management API.

### Work Items

#### 1.1: Implement `Manager::create_queue()` method

- **Function:** `pub fn create_queue(&mut self, queue_name: &str) -> Result<u32>`
- **Requirements:**
  - MUST call actual Kompute C++ Manager queue creation methods
  - MUST return valid queue index for subsequent operations
  - MUST validate queue_name is non-empty and unique
  - MUST NOT BE STUBBED - implement real C++ integration
  - MUST ASK FOR HELP if C++ Kompute Manager doesn't support queue creation
- **Tests Required:**
  - Unit test: Valid queue creation returns non-zero index
  - Unit test: Duplicate queue names fail with clear error
  - Unit test: Empty queue name fails with validation error
  - Integration test: Created queue can be used for sequences

#### 1.2: Implement `Manager::with_queues()` constructor

- **Function:** `pub fn with_queues(device_id: u32, queue_families: &[u32]) -> Result<Self>`
- **Requirements:**
  - MUST call Kompute C++ Manager constructor with queue family parameters
  - MUST validate queue_families array is non-empty and valid indices
  - MUST NOT BE STUBBED - implement real multi-queue initialization
  - MUST ASK FOR HELP if C++ API doesn't match expected signature
- **Tests Required:**
  - Unit test: Valid queue families create manager successfully
  - Unit test: Empty queue families array fails validation
  - Unit test: Invalid queue family indices fail with clear error
  - Integration test: Multi-queue manager can create sequences on different queues

#### 1.3: Implement `Manager::with_extensions()` constructor

- **Function:** `pub fn with_extensions(device_id: u32, queue_families: &[u32], extensions: &[&str]) -> Result<Self>`
- **Requirements:**
  - MUST call Kompute C++ Manager constructor with Vulkan extensions
  - MUST validate extension names are valid Vulkan extension strings
  - MUST NOT BE STUBBED - implement real Vulkan extension support
  - MUST ASK FOR HELP if Vulkan extension loading requires additional C++ work
- **Tests Required:**
  - Unit test: Valid extensions create manager successfully
  - Unit test: Invalid extension names fail with descriptive error
  - Unit test: Empty extensions array works (optional extensions)
  - Integration test: Manager with extensions can perform extended operations

#### 1.4: Implement `Manager::sequence_on_queue()` method

- **Function:** `pub fn sequence_on_queue(&mut self, queue_index: u32) -> Result<Sequence>`
- **Requirements:**
  - MUST create sequence on specific queue index from create_queue()
  - MUST validate queue_index exists and was previously created
  - MUST NOT BE STUBBED - implement real queue-specific sequence creation
  - MUST ASK FOR HELP if C++ Kompute Sequence constructor needs queue parameter
- **Tests Required:**
  - Unit test: Valid queue index creates sequence successfully
  - Unit test: Invalid queue index fails with clear error
  - Unit test: Queue index 0 (default) always works
  - Integration test: Sequences on different queues can run in parallel

#### 1.5: Add C++ FFI support functions

- **Function:** Add to `kompute-sys/src/bridge.rs` and `kompute_bridge.cpp`
- **Requirements:**
  - MUST add `fn create_queue_handle(manager: Pin<&mut Manager>, queue_name: &str) -> u32`
  - MUST add `fn create_manager_with_queues(device_id: u32, queue_families: &[u32]) -> UniquePtr<Manager>`
  - MUST add `fn create_manager_with_extensions(device_id: u32, queue_families: &[u32], extensions: &[&str]) -> UniquePtr<Manager>`
  - MUST add `fn create_sequence_on_queue_handle(manager: Pin<&mut Manager>, queue_index: u32) -> u64`
  - MUST NOT BE STUBBED - implement real C++ Kompute API calls
  - MUST ASK FOR HELP if Kompute C++ API doesn't support these operations
- **Tests Required:**
  - Unit test: FFI functions don't crash with valid inputs
  - Unit test: FFI functions return error codes for invalid inputs
  - Integration test: FFI functions integrate correctly with Rust wrappers

#### 1.6: Update SYSTEM_ARCHITECTURE.md for multi-queue support

- **Requirements:**
  - MUST document multi-queue architecture in handle-based system
  - MUST update data flow diagrams to show queue management
  - MUST document queue naming and lifecycle management
  - MUST update implementation status to reflect completed multi-queue features

#### 1.7: Add comprehensive API documentation comments

- **Requirements:**
  - MUST add rustdoc comments to all new public functions
  - MUST include usage examples for each queue management function
  - MUST document error conditions and recovery strategies
  - MUST provide complete parameter documentation with types and constraints

---

## PHASE 2: Algorithm-Tensor Binding System (`kompute/src/algorithm.rs`)

### Critical Missing Functionality

Algorithms are currently created with empty tensor lists, making GPU compute impossible. This phase implements proper algorithm-tensor binding.

### Work Items

#### 2.1: Implement `Algorithm::with_tensors()` method

- **Function:** `pub fn with_tensors(manager: &mut Manager, tensors: &[&Tensor<f32>], spirv: &[u32]) -> Result<Self>`
- **Requirements:**
  - MUST bind actual tensor handles to algorithm creation
  - MUST validate tensors array is non-empty for compute operations
  - MUST call C++ Kompute Algorithm constructor with tensor shared_ptrs
  - MUST NOT BE STUBBED - implement real tensor binding
  - MUST ASK FOR HELP if C++ Algorithm constructor needs different tensor format
- **Tests Required:**
  - Unit test: Algorithm with valid tensors creates successfully
  - Unit test: Algorithm with empty tensors fails validation
  - Unit test: Algorithm with invalid tensor handles fails clearly
  - Integration test: Algorithm can dispatch and modify bound tensors

#### 2.2: Implement `Algorithm::set_tensors()` method

- **Function:** `pub fn set_tensors(&mut self, tensors: &[&Tensor<f32>]) -> Result<()>`
- **Requirements:**
  - MUST update algorithm's bound tensors after creation
  - MUST validate new tensors are compatible with existing algorithm
  - MUST call C++ Kompute Algorithm rebinding methods
  - MUST NOT BE STUBBED - implement real runtime tensor rebinding
  - MUST ASK FOR HELP if C++ Algorithm doesn't support tensor rebinding
- **Tests Required:**
  - Unit test: Rebinding with compatible tensors succeeds
  - Unit test: Rebinding with incompatible tensors fails validation
  - Unit test: Rebinding preserves algorithm configuration (workgroup, etc.)
  - Integration test: Algorithm dispatch works with rebound tensors

#### 2.3: Implement `Algorithm::set_workgroup()` method

- **Function:** `pub fn set_workgroup(&mut self, workgroup: Workgroup) -> Result<()>`
- **Requirements:**
  - MUST configure algorithm's GPU workgroup size (x, y, z dimensions)
  - MUST validate workgroup dimensions are within GPU limits
  - MUST call C++ Kompute Algorithm workgroup configuration
  - MUST NOT BE STUBBED - implement real workgroup specification
  - MUST ASK FOR HELP if C++ Algorithm workgroup setting requires different approach
- **Tests Required:**
  - Unit test: Valid workgroup dimensions configure successfully
  - Unit test: Zero or negative dimensions fail validation
  - Unit test: Workgroup dimensions exceeding GPU limits fail clearly
  - Integration test: Algorithm dispatch uses configured workgroup size

#### 2.4: Implement `Algorithm::set_specialization_constants()` method

- **Function:** `pub fn set_specialization_constants(&mut self, constants: &[f32]) -> Result<()>`
- **Requirements:**
  - MUST set SPIR-V specialization constants for shader compilation
  - MUST validate constants array matches shader requirements
  - MUST call C++ Kompute Algorithm specialization constant methods
  - MUST NOT BE STUBBED - implement real specialization constant support
  - MUST ASK FOR HELP if C++ Algorithm specialization requires different data types
- **Tests Required:**
  - Unit test: Valid specialization constants set successfully
  - Unit test: Mismatched constant count fails with clear error
  - Unit test: Invalid constant values fail validation
  - Integration test: Algorithm uses specialization constants in GPU execution

#### 2.5: Implement `Algorithm::set_push_constants()` method

- **Function:** `pub fn set_push_constants(&mut self, constants: &[f32]) -> Result<()>`
- **Requirements:**
  - MUST set Vulkan push constants for efficient parameter passing
  - MUST validate push constants size within Vulkan limits
  - MUST call C++ Kompute Algorithm push constant methods
  - MUST NOT BE STUBBED - implement real push constant support
  - MUST ASK FOR HELP if C++ Algorithm push constants need different format
- **Tests Required:**
  - Unit test: Valid push constants set successfully
  - Unit test: Oversized push constants fail with size limit error
  - Unit test: Push constants persist across algorithm reuse
  - Integration test: Algorithm receives push constants in GPU shader

#### 2.6: Add C++ FFI support for algorithm configuration

- **Function:** Add to `kompute-sys/src/bridge.rs` and `kompute_bridge.cpp`
- **Requirements:**
  - MUST add `fn create_algorithm_with_tensors_handle(manager: Pin<&mut Manager>, tensor_handles: &[u64], spirv: &[u32]) -> u64`
  - MUST add `fn algorithm_set_tensors_by_handles(algo_handle: u64, tensor_handles: &[u64]) -> Result<()>`
  - MUST add `fn algorithm_set_workgroup(algo_handle: u64, x: u32, y: u32, z: u32) -> Result<()>`
  - MUST add `fn algorithm_set_specialization_constants(algo_handle: u64, constants: &[f32]) -> Result<()>`
  - MUST add `fn algorithm_set_push_constants(algo_handle: u64, constants: &[f32]) -> Result<()>`
  - MUST NOT BE STUBBED - implement real C++ Kompute Algorithm API calls
  - MUST ASK FOR HELP if Kompute C++ Algorithm API doesn't support these operations
- **Tests Required:**
  - Unit test: All FFI functions handle valid inputs without crashing
  - Unit test: FFI functions return proper error codes for invalid inputs
  - Integration test: FFI functions properly integrate with Rust Algorithm wrapper

#### 2.7: Update SYSTEM_ARCHITECTURE.md for algorithm-tensor binding

- **Requirements:**
  - MUST document algorithm-tensor binding architecture
  - MUST update data flow to show tensor binding in compute pipeline
  - MUST document workgroup and constant configuration patterns
  - MUST update implementation status to reflect completed algorithm features

#### 2.8: Add comprehensive API documentation comments

- **Requirements:**
  - MUST add rustdoc comments to all algorithm configuration functions
  - MUST include examples showing algorithm creation and configuration
  - MUST document workgroup sizing guidelines and GPU limits
  - MUST provide specialization and push constant usage examples

---

## PHASE 3: Multi-Type Tensor Support (`kompute/src/tensor.rs`)

### Current Limitation

Only f32 tensors are fully implemented. This phase adds support for other numeric types essential for GPU computing.

### Work Items

#### 3.1: Implement `Tensor<f64>` full support

- **Function:** Extend existing Tensor<T> implementation for f64
- **Requirements:**
  - MUST add `fn create_tensor_f64_handle()` to C++ bridge
  - MUST add `fn tensor_data_f64_by_handle()` to C++ bridge
  - MUST implement all Tensor<f64> methods (new, to_vec, size, etc.)
  - MUST NOT BE STUBBED - implement complete f64 tensor lifecycle
  - MUST ASK FOR HELP if C++ Kompute Tensor doesn't support f64 data type
- **Tests Required:**
  - Unit test: f64 tensor creation with valid data
  - Unit test: f64 tensor data round-trip maintains precision
  - Unit test: f64 tensor size calculations correct
  - Integration test: f64 tensors work in sequences and algorithms

#### 3.2: Implement `Tensor<i32>` full support

- **Function:** Extend existing Tensor<T> implementation for i32
- **Requirements:**
  - MUST add `fn create_tensor_i32_handle()` to C++ bridge
  - MUST add `fn tensor_data_i32_by_handle()` to C++ bridge
  - MUST implement all Tensor<i32> methods with proper integer handling
  - MUST NOT BE STUBBED - implement complete i32 tensor lifecycle
  - MUST ASK FOR HELP if C++ Kompute Tensor integer support requires special handling
- **Tests Required:**
  - Unit test: i32 tensor creation with positive and negative integers
  - Unit test: i32 tensor data preserves exact integer values
  - Unit test: i32 tensor size and byte calculations correct
  - Integration test: i32 tensors work correctly in GPU computations

#### 3.3: Implement `Tensor<u32>` full support

- **Function:** Extend existing Tensor<T> implementation for u32
- **Requirements:**
  - MUST add `fn create_tensor_u32_handle()` to C++ bridge
  - MUST add `fn tensor_data_u32_by_handle()` to C++ bridge
  - MUST implement all Tensor<u32> methods with unsigned integer handling
  - MUST NOT BE STUBBED - implement complete u32 tensor lifecycle
  - MUST ASK FOR HELP if C++ Kompute unsigned integer tensors need special consideration
- **Tests Required:**
  - Unit test: u32 tensor creation with large unsigned values
  - Unit test: u32 tensor data preserves full unsigned range
  - Unit test: u32 tensor byte size calculations handle unsigned correctly
  - Integration test: u32 tensors work in GPU index and mask operations

#### 3.4: Implement `Manager::tensor_filled()` method

- **Function:** `pub fn tensor_filled<T: TensorData>(&mut self, size: usize, fill_value: T) -> Result<Tensor<T>>`
- **Requirements:**
  - MUST create tensor filled with specified value for all supported types
  - MUST validate size is within reasonable limits
  - MUST optimize for large tensors by using GPU fill operations if available
  - MUST NOT BE STUBBED - implement efficient tensor filling
  - MUST ASK FOR HELP if C++ Kompute doesn't provide efficient tensor filling
- **Tests Required:**
  - Unit test: Filled tensor has correct size and uniform values
  - Unit test: Fill value is preserved exactly for all data types
  - Unit test: Large filled tensors complete within reasonable time
  - Integration test: Filled tensors work correctly as algorithm inputs

#### 3.5: Implement `Tensor::data_type()` method

- **Function:** `pub fn data_type(&self) -> DataType`
- **Requirements:**
  - MUST return runtime type information for tensor data type
  - MUST distinguish between f32, f64, i32, u32, i8, u8 types
  - MUST query C++ Kompute Tensor for actual data type
  - MUST NOT BE STUBBED - implement real runtime type detection
  - MUST ASK FOR HELP if C++ Kompute Tensor doesn't provide type introspection
- **Tests Required:**
  - Unit test: Each tensor type returns correct DataType enum
  - Unit test: DataType enum values match TensorData trait implementations
  - Unit test: Runtime type detection works across FFI boundary
  - Integration test: Type-safe operations use correct data type information

#### 3.6: Add C++ FFI support for multi-type tensors

- **Function:** Add to `kompute-sys/src/bridge.rs` and `kompute_bridge.cpp`
- **Requirements:**
  - MUST add complete tensor operations for f64, i32, u32 types
  - MUST add `fn create_tensor_filled_*_handle()` for each type
  - MUST add `fn tensor_data_type_by_handle()` for runtime type detection
  - MUST maintain same safety patterns as existing f32 implementation
  - MUST NOT BE STUBBED - implement full multi-type tensor support
  - MUST ASK FOR HELP if C++ Kompute template system needs different approach
- **Tests Required:**
  - Unit test: All FFI functions work correctly for each data type
  - Unit test: Type detection FFI returns correct C++ tensor type information
  - Unit test: Memory layout matches between Rust and C++ for all types
  - Integration test: Multi-type tensors work correctly through full pipeline

#### 3.7: Update SYSTEM_ARCHITECTURE.md for multi-type tensor support

- **Requirements:**
  - MUST document multi-type tensor architecture and type safety
  - MUST update data model documentation to show generic type support
  - MUST document type coercion and conversion policies
  - MUST update implementation status to reflect completed multi-type features

#### 3.8: Add comprehensive API documentation comments

- **Requirements:**
  - MUST add rustdoc comments documenting supported data types
  - MUST include examples for each tensor type (f32, f64, i32, u32)
  - MUST document type-specific considerations and limitations
  - MUST provide guidance on choosing appropriate data types for GPU operations

---

## PHASE 4: Real Async GPU Integration (`kompute/src/async_executor.rs`)

### Current Limitation

Async implementation is polling-based simulation, not real GPU async. This phase implements true GPU async integration.

### Work Items

#### 4.1: Implement real `AsyncSequenceExecution::poll()` with GPU events

- **Function:** `fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>`
- **Requirements:**
  - MUST replace polling loop with actual Vulkan event/semaphore integration
  - MUST use C++ Kompute async completion callbacks or Vulkan synchronization
  - MUST integrate with Rust async runtime for proper task waking
  - MUST NOT BE STUBBED - implement real GPU async, not polling simulation
  - MUST ASK FOR HELP if Kompute C++ doesn't provide async completion APIs
- **Tests Required:**
  - Unit test: Async sequence completes without blocking thread
  - Unit test: Multiple async sequences can run concurrently
  - Unit test: Async completion integrates correctly with tokio runtime
  - Integration test: Real GPU work completes asynchronously with correct results

#### 4.2: Implement `sequence_eval_async_with_callback()` C++ bridge

- **Function:** Add to `kompute-sys/src/bridge.rs` and `kompute_bridge.cpp`
- **Requirements:**
  - MUST add C++ callback mechanism for GPU completion notification
  - MUST integrate with Vulkan VkEvent or VkSemaphore for real async detection
  - MUST provide thread-safe callback invocation to wake Rust futures
  - MUST NOT BE STUBBED - implement real GPU async detection mechanism
  - MUST ASK FOR HELP if Vulkan async integration requires additional C++ infrastructure
- **Tests Required:**
  - Unit test: C++ callback fires when GPU work actually completes
  - Unit test: Callback thread safety doesn't cause crashes or race conditions
  - Unit test: Multiple concurrent callbacks work correctly
  - Integration test: Callback mechanism integrates with Rust async executor

#### 4.3: Implement `AsyncGpuExecutor::execute_parallel()` with real concurrency

- **Function:** `pub async fn execute_parallel(&mut self, sequences: Vec<AsyncSequence>, timeout: Duration) -> Result<()>`
- **Requirements:**
  - MUST execute sequences on separate GPU queues for true parallelism
  - MUST use actual GPU queue submission and synchronization
  - MUST coordinate completion across multiple GPU queues efficiently
  - MUST NOT BE STUBBED - implement real multi-queue GPU parallelism
  - MUST ASK FOR HELP if Kompute C++ multi-queue execution needs different approach
- **Tests Required:**
  - Unit test: Parallel execution actually uses multiple GPU queues
  - Unit test: Parallel sequences complete faster than sequential execution
  - Unit test: Queue synchronization prevents data races between sequences
  - Integration test: Complex multi-queue workflows execute correctly with proper results

#### 4.4: Implement async timeout with GPU work cancellation

- **Function:** Enhance existing timeout mechanisms with GPU work cancellation
- **Requirements:**
  - MUST cancel pending GPU work when timeout expires, not just fail the future
  - MUST use Vulkan command buffer reset or queue idle operations
  - MUST clean up GPU resources properly when cancellation occurs
  - MUST NOT BE STUBBED - implement real GPU work cancellation
  - MUST ASK FOR HELP if Vulkan GPU work cancellation requires complex synchronization
- **Tests Required:**
  - Unit test: Timeout properly cancels in-flight GPU operations
  - Unit test: Cancelled operations don't affect subsequent GPU work
  - Unit test: GPU resource cleanup occurs correctly after cancellation
  - Integration test: System remains stable after multiple timeout cancellations

#### 4.5: Update SYSTEM_ARCHITECTURE.md for real async GPU integration

- **Requirements:**
  - MUST document real async GPU architecture vs polling simulation
  - MUST explain Vulkan synchronization primitive integration
  - MUST document async runtime integration patterns
  - MUST update implementation status to reflect completed real async features

#### 4.6: Add comprehensive API documentation comments

- **Requirements:**
  - MUST document async GPU execution model and performance characteristics
  - MUST provide examples of effective async GPU programming patterns
  - MUST document timeout and cancellation behavior clearly
  - MUST explain integration with popular async runtimes (tokio, async-std)

---

## Implementation Order and Dependencies

### Dependency Chain

1. **Phase 0** (Test Infrastructure Completion) → Establishes quality gates and validates existing functionality
2. **Phase 1** (Manager Multi-Queue) → Enables MultiQueueManager functionality
3. **Phase 2** (Algorithm-Tensor Binding) → Enables actual GPU compute operations
4. **Phase 3** (Multi-Type Tensors) → Enables production-ready data type support
5. **Phase 4** (Real Async GPU) → Enables high-performance async GPU computing

### Critical Success Factors

- Each phase MUST be completed fully before proceeding to next phase
- All work items within a phase MUST be implemented without stubs or workarounds
- If any implementation requires shortcuts, MUST ask user for guidance before proceeding
- Evidence-based completion required for each work item (tests pass, functionality proven)

### Quality Gates (Applied to Each Phase)

1. **Code Quality**: `cargo fmt --check && cargo clippy -- -D warnings`
2. **Test Coverage**: All unit and integration tests pass
3. **Performance**: No regressions vs baseline C++ performance
4. **Documentation**: Complete rustdoc comments for all public APIs
5. **Integration**: Clean integration with existing components

This plan provides systematic, function-level implementation guidance while maintaining the quality standards and methodology established in CONTRIBUTING.md and GRIMOIRE.md.
