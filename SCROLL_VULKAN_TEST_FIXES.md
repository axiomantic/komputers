# Implementation Scroll: Vulkan Test Infrastructure & Example Integration

**Version:** 1.0
**Summoned:** 2025-01-19
**Component:** Vulkan Test Infrastructure Repairs & Example Test Integration
**Sacred Promise:** I shall enable all GPU tests on macOS/MoltenVK, fix failing tests, and create comprehensive example testing infrastructure

---

## 1. The Sacred Invocation

**Your Persona:** Master Test Infrastructure Engineer specializing in GPU compute validation and cross-platform Vulkan compatibility.

**Primary Goal:** Enable and fix all 22 ignored Vulkan tests on macOS with MoltenVK, repair failing integration tests, and establish comprehensive example testing.

**The Sacred Promise:**

> "I promise to systematically enable GPU testing on macOS, fix all test failures with proper diagnostics, establish example testing infrastructure, and validate all work through comprehensive evidence. I will not declare tests 'ignored' when we have a functioning Vulkan implementation through MoltenVK."

---

## 2. Current State Analysis

### Test Infrastructure Status:

- **22 tests ignored** with message "requires Vulkan-capable device"
- **1 failing test** in integration suite (`test_manager_error_paths`)
- **7 examples created** but not properly registered or tested
- **MoltenVK available** on macOS but tests not utilizing it

### Root Cause Analysis:

1. **Vulkan Detection Issue**: Tests likely checking for native Vulkan, not detecting MoltenVK
2. **Environment Configuration**: Missing or incorrect Vulkan environment variables
3. **Example Organization**: Examples scattered between directories, not registered in Cargo.toml
4. **No Example Tests**: Examples lack integration testing infrastructure

### CRITICAL DESIGN DECISION:

**NO TEST SKIPPING**: We will NEVER skip tests due to missing Vulkan. The absence of Vulkan/MoltenVK is considered a CATASTROPHIC FAILURE of the test environment. All systems running these tests MUST have Vulkan available. This is non-negotiable.

### NEW REQUIREMENT: Intelligent GPU Configuration & User-Friendly Error Handling

**Auto-MoltenVK Configuration:**

- **Happy Path Default**: Automatically detect and configure MoltenVK on systems with Metal GPU support
- **Multi-GPU Support**: Handle systems with both Metal and discrete GPUs (e.g., NVIDIA on macOS, Linux on M1)
- **User Override**: Provide `KOMPUTE_DISABLE_MOLTENVK_AUTO` environment variable to disable auto-configuration
- **Cross-Platform**: Support Metal GPU detection beyond just macOS (future Linux ARM support)

**Comprehensive Error Handling:**

- **Device Initialization Failure**: When device enumeration returns 0 devices, provide detailed diagnostics
- **Metal-Specific Guidance**: On Metal-capable systems, guide users to:
  - Install MoltenVK if missing
  - Set proper environment variables
  - Use override env var if preferring other GPUs
- **Platform-Specific Messages**: Tailored error messages for macOS, Linux, Windows
- **Clear Exit Strategy**: Fail fast with actionable error messages, never silently degrade

**Implementation Priority**: Add as Phase 0 (foundational requirement before test enablement)

---

## 3. Implementation Plan

### Phase 0: Intelligent GPU Configuration & Error Handling (NEW)

#### Task 0.1: Implement Metal GPU Detection

```rust
// In kompute/src/platform.rs or new gpu_detection.rs
pub fn detect_metal_gpu_support() -> bool {
    #[cfg(target_os = "macos")]
    {
        // Check for Metal framework and GPU capabilities
        // Use metal-rs or system_profiler to detect Metal support
        true // Assume Metal on macOS for now
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Future: detect ARM Linux systems with Metal support
        false
    }
}

pub fn should_auto_configure_moltenvk() -> bool {
    // Check KOMPUTE_DISABLE_MOLTENVK_AUTO environment variable
    if std::env::var("KOMPUTE_DISABLE_MOLTENVK_AUTO").is_ok() {
        return false;
    }
    detect_metal_gpu_support()
}
```

#### Task 0.2: Enhanced Vulkan Configuration Logic

```rust
// In kompute/src/vulkan_setup.rs
pub fn configure_vulkan_environment_intelligent() -> Result<(), String> {
    if should_auto_configure_moltenvk() {
        configure_moltenvk_auto()?;
    }

    // Fallback to standard configuration
    configure_vulkan_environment()?;

    // Validate that devices are available
    validate_device_availability()
}

fn validate_device_availability() -> Result<(), String> {
    let device_count = crate::Manager::device_count()?;
    if device_count == 0 {
        return Err(generate_helpful_device_error());
    }
    Ok(())
}

fn generate_helpful_device_error() -> String {
    let mut error = String::from("No Vulkan devices found. ");

    if detect_metal_gpu_support() {
        error.push_str("On this Metal-capable system:\n");
        error.push_str("  1. Install MoltenVK: brew install molten-vk\n");
        error.push_str("  2. Ensure VK_ICD_FILENAMES points to MoltenVK_icd.json\n");
        error.push_str("  3. To use discrete GPUs instead, set KOMPUTE_DISABLE_MOLTENVK_AUTO=1\n");
    } else {
        error.push_str("Ensure Vulkan drivers are installed for your GPU.");
    }

    error
}
```

#### Task 0.3: Update Manager Creation

```rust
// In kompute/src/manager.rs
impl Manager {
    pub fn new() -> Result<Self> {
        // Use intelligent configuration before attempting device creation
        configure_vulkan_environment_intelligent()
            .map_err(|e| KomputeError::device(format!("Vulkan configuration failed: {}", e)))?;

        Self::with_device(0)
    }
}
```

#### Task 0.4: Environment Variable Documentation

- Add `KOMPUTE_DISABLE_MOLTENVK_AUTO` to environment variable documentation
- Update examples to show multi-GPU configuration scenarios
- Add troubleshooting guide for different GPU setups

### Phase 1: Vulkan/MoltenVK Detection & Configuration

#### Task 1.1: Diagnose Current Vulkan Setup

```rust
// Check current Vulkan environment and availability
- Run vulkan_setup::verify_vulkan_availability()
- Check VULKAN_SDK, VK_ICD_FILENAMES, VK_LAYER_PATH env vars
- Verify MoltenVK library presence at expected paths
```

#### Task 1.2: Fix Vulkan Detection for Tests

```rust
// Update test infrastructure to properly detect MoltenVK
impl Manager {
    #[cfg(test)]
    pub fn new_for_test() -> Result<Self> {
        // Special initialization for test environment
        // Configure for MoltenVK on macOS
        if cfg!(target_os = "macos") {
            vulkan_setup::configure_vulkan_environment();
        }
        Self::new()
    }
}
```

#### Task 1.3: Create Test Helper Module

```rust
// kompute/src/test_utils.rs
#[cfg(test)]
pub mod test_utils {
    use crate::vulkan_setup;
    use crate::error::KomputeError;

    /// CRITICAL: Vulkan MUST be available for ALL tests
    /// We NEVER skip tests - absence of Vulkan is a FAILURE
    pub fn require_vulkan() -> Result<(), KomputeError> {
        match vulkan_setup::verify_vulkan_availability() {
            Ok(_) => Ok(()),
            Err(e) => {
                panic!(
                    "CRITICAL FAILURE: Vulkan/MoltenVK not available!\n\
                    This is a complete test infrastructure failure.\n\
                    All tests REQUIRE a Vulkan device.\n\
                    Error: {}\n\
                    \n\
                    On macOS: Ensure MoltenVK is properly installed and configured.\n\
                    Environment variables to check:\n\
                    - VULKAN_SDK\n\
                    - VK_ICD_FILENAMES\n\
                    - VK_LAYER_PATH",
                    e
                );
            }
        }
    }

    /// Ensure Vulkan is available at test module initialization
    pub fn init_test_environment() {
        // This will panic and fail the entire test suite if Vulkan is not available
        require_vulkan().expect("Test environment initialization failed");
    }
}
```

### Phase 2: Enable & Fix GPU Tests

#### Task 2.1: Remove Ignore Attributes Systematically

```rust
// Replace all instances of:
#[test]
#[ignore = "requires Vulkan-capable device"]

// With:
#[test]
fn test_name() {
    // MANDATORY: Vulkan MUST be available
    test_utils::require_vulkan().unwrap();
    // ... rest of test
}

// Or at module level for all tests in a module:
#[cfg(test)]
mod tests {
    use super::*;

    // This ensures ALL tests in this module have Vulkan
    // Fails entire test run if Vulkan not available
    static _VULKAN_REQUIRED: std::sync::Once = std::sync::Once::new();

    fn ensure_vulkan() {
        _VULKAN_REQUIRED.call_once(|| {
            test_utils::init_test_environment();
        });
    }

    #[test]
    fn test_gpu_operation() {
        ensure_vulkan();
        // Test implementation
    }
}
```

#### Task 2.2: Fix Individual Test Failures

1. Run each previously ignored test
2. Diagnose specific failure modes
3. Apply targeted fixes:
   - Environment configuration issues
   - Resource initialization problems
   - MoltenVK-specific compatibility

#### Task 2.3: Fix Integration Test Failure

```rust
// Fix test_manager_error_paths expectation
// Update error message validation to match actual error format
```

### Phase 3: Example Organization & Registration

#### Task 3.1: Consolidate Examples

```bash
# Move all examples to kompute/examples/
mv examples/*.rs kompute/examples/
```

#### Task 3.2: Register in Cargo.toml

```toml
[[example]]
name = "basic_gpu_computing"
path = "examples/basic_gpu_computing.rs"

[[example]]
name = "multi_type_tensors"
path = "examples/multi_type_tensors.rs"

[[example]]
name = "advanced_gpu_operations"
path = "examples/advanced_gpu_operations.rs"

[[example]]
name = "async_gpu_operations"
path = "examples/async_gpu_operations.rs"
required-features = ["async"]

[[example]]
name = "machine_learning_operations"
path = "examples/machine_learning_operations.rs"

[[example]]
name = "scientific_computing"
path = "examples/scientific_computing.rs"

[[example]]
name = "image_processing_pipeline"
path = "examples/image_processing_pipeline.rs"
```

### Phase 4: Example Testing Infrastructure

#### Task 4.1: Make Examples Testable

```rust
// Modify each example to support both interactive and test modes
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let test_mode = args.contains(&"--test".to_string());

    if test_mode {
        run_example_test()
    } else {
        run_interactive()
    }
}

fn run_example_test() -> Result<()> {
    // Deterministic test with verifiable output
    let result = perform_computation()?;

    // Print parseable output for integration test
    println!("TEST_RESULT: {}", serde_json::to_string(&result)?);
    Ok(())
}
```

#### Task 4.2: Create Integration Tests for Examples

```rust
// kompute/tests/example_integration_tests.rs
#[test]
fn test_basic_gpu_computing_example() {
    let output = std::process::Command::new("cargo")
        .args(&["run", "--example", "basic_gpu_computing", "--", "--test"])
        .output()
        .expect("Failed to run example");

    assert!(output.status.success());

    // Parse and validate output
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("TEST_RESULT:"));

    // Validate specific computation results
    let result_line = stdout.lines()
        .find(|line| line.starts_with("TEST_RESULT:"))
        .unwrap();

    // Parse JSON and validate
    let json_str = result_line.strip_prefix("TEST_RESULT: ").unwrap();
    let result: serde_json::Value = serde_json::from_str(json_str).unwrap();

    // Validate computation correctness
    assert_eq!(result["computation"], "success");
}
```

---

## 4. Validation Strategy

### Test Coverage Requirements:

1. **Intelligent GPU Configuration** must work correctly:
   - Auto-detect Metal GPU support on appropriate systems
   - Auto-configure MoltenVK when appropriate
   - Respect `KOMPUTE_DISABLE_MOLTENVK_AUTO` environment variable
   - Provide helpful error messages when no devices found
2. **All 22 GPU tests** must run successfully on macOS
3. **Integration test** must pass with corrected expectations
4. **All 7 examples** must:
   - Build without warnings
   - Run in test mode with verifiable output
   - Pass integration tests

### Success Criteria:

```bash
# Phase 0: Intelligent GPU Configuration Validation
# Test auto-MoltenVK configuration
cargo test test_intelligent_gpu_config -- --nocapture

# Test environment variable override
KOMPUTE_DISABLE_MOLTENVK_AUTO=1 cargo test test_moltenvk_disabled -- --nocapture

# Test helpful error messages
# (simulate no devices scenario)
cargo test test_helpful_device_errors -- --nocapture

# All tests pass including GPU tests - NO IGNORED TESTS
# This MUST show 0 ignored tests and all passing
cargo test --all

# Verify no tests are being skipped
cargo test --all 2>&1 | grep -E "test result:" | grep "0 ignored"

# All examples build
cargo build --examples

# All examples run in test mode
for example in $(cargo build --examples 2>&1 | grep -o '\w\+$'); do
    cargo run --example $example -- --test
done

# Integration tests pass
cargo test --test example_integration_tests

# CRITICAL VERIFICATION: Ensure Vulkan is actually being used
# This should show device enumeration and successful GPU operations
RUST_LOG=debug cargo test -- --nocapture 2>&1 | grep -i "vulkan\|gpu\|device"
```

---

## 5. Implementation Order

1. **Fix Vulkan detection** (Task 1.1-1.3)
2. **Enable one GPU test** as proof of concept
3. **Fix the failing integration test** (Task 2.3)
4. **Enable remaining GPU tests** (Task 2.1-2.2)
5. **Organize examples** (Task 3.1-3.2)
6. **Add test mode to examples** (Task 4.1)
7. **Create integration tests** (Task 4.2)

---

## 6. Risk Mitigation

### Potential Issues:

1. **MoltenVK Limitations**: Some Vulkan features may not be supported

   - Mitigation: Fix the implementation to work with MoltenVK's available features
   - NO DEGRADATION - if a feature is needed, make it work

2. **Environment Variables**: Complex Vulkan setup on different systems

   - Mitigation: Auto-configure environment in test setup
   - Provide diagnostic output when Vulkan init fails

3. **Test Flakiness**: GPU tests may be non-deterministic

   - Mitigation: Add retry logic and tolerance for minor variations
   - But NEVER skip a test

4. **CI/CD Compatibility**: GitHub Actions may not have GPU
   - Mitigation: CI can use a different test profile, but local MUST run all tests
   - Use `--features ci-skip-gpu` ONLY for CI, never for local development

---

## 7. The Sacred Validation

Upon completion, provide evidence:

1. Full test output showing all tests passing
2. Example execution logs in test mode
3. Integration test results
4. Coverage report showing no ignored tests on local machine

**The implementation shall proceed with systematic precision, validating each phase before advancing.**
