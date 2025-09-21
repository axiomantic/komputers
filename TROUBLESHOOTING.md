# Komputers Troubleshooting Guide

This guide helps you resolve common issues when using Komputers for GPU computing in Rust.

## Quick Diagnostics

### Check Your Environment

```bash
# Verify Rust installation
rustc --version
cargo --version

# Check Vulkan availability
vulkaninfo | head -20

# Test Komputers installation
cargo run --example basic_gpu_computing
```

### Environment Variables

```bash
# Enable debug logging
export RUST_LOG=debug

# macOS: Disable auto MoltenVK configuration (if needed)
export KOMPUTE_DISABLE_MOLTENVK_AUTO=1

# Vulkan validation layers (for debugging)
export VK_LAYER_ENABLES=VK_LAYER_KHRONOS_validation
```

## Platform-Specific Issues

### macOS Issues

#### Problem: "Manager constructor segfaults"

**Symptoms:**

- Segmentation fault when creating `Manager::new()`
- Crashes during GPU initialization

**Solution:**

```bash
# 1. Verify MoltenVK installation
ls /opt/homebrew/lib/libMoltenVK.dylib
ls /usr/local/lib/libMoltenVK.dylib

# 2. Check Vulkan SDK
echo $VULKAN_SDK
vulkaninfo | head -10

# 3. Try manual MoltenVK setup
export KOMPUTE_DISABLE_MOLTENVK_AUTO=1
export VK_ICD_FILENAMES=/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json
export VK_DRIVER_FILES=/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json
```

#### Problem: "No Vulkan drivers found"

**Symptoms:**

- `vulkaninfo` fails
- "No compatible Vulkan device found"

**Solution:**

```bash
# Install Vulkan SDK and MoltenVK
brew install vulkan-headers vulkan-loader molten-vk

# Verify installation
vulkaninfo --summary
```

#### Problem: "Metal GPU not detected"

**Symptoms:**

- GPU operations fail on Apple Silicon
- MoltenVK not working

**Solution:**

```bash
# Check Metal support
system_profiler SPDisplaysDataType | grep "Metal"

# Verify Apple GPU
system_profiler SPDisplaysDataType | grep -A 5 "Apple"

# Test MoltenVK directly
vulkaninfo | grep -i metal
```

### Linux Issues

#### Problem: "Build fails with missing Vulkan headers"

**Symptoms:**

- `fatal error: vulkan/vulkan.h: No such file or directory`
- CMake configuration errors

**Solution:**

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install vulkan-tools libvulkan-dev vulkan-validationlayers-dev

# Fedora/RHEL
sudo dnf install vulkan-tools vulkan-devel vulkan-validation-layers-devel

# Arch Linux
sudo pacman -S vulkan-tools vulkan-headers vulkan-validation-layers
```

#### Problem: "No GPU device found"

**Symptoms:**

- `vulkaninfo` shows no devices
- Driver issues

**Solution:**

```bash
# NVIDIA
sudo apt install nvidia-driver-XXX vulkan-utils
nvidia-smi  # Verify driver

# AMD
sudo apt install mesa-vulkan-drivers vulkan-utils
vulkaninfo | grep -i amd

# Intel
sudo apt install intel-media-va-driver-non-free vulkan-utils
vulkaninfo | grep -i intel
```

### Windows Issues

#### Problem: "Build fails with CMake errors"

**Symptoms:**

- CMake can't find Visual Studio
- Build tools not found

**Solution:**

1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
2. Install [CMake](https://cmake.org/download/)
3. Restart terminal and try again

#### Problem: "Vulkan not found"

**Symptoms:**

- `vulkaninfo` command not found
- Vulkan SDK missing

**Solution:**

1. Download and install [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
2. Add to PATH: `C:\VulkanSDK\[version]\Bin`
3. Verify: `vulkaninfo.exe`

## Build and Compilation Issues

### Problem: "Linking errors with kompute-sys"

**Symptoms:**

- Undefined symbol errors
- Linking phase fails

**Solution:**

```bash
# Clean rebuild
cargo clean
rm -rf target/

# Rebuild with verbose output
cargo build --all-features --verbose

# Check CMake output
RUST_LOG=debug cargo build --features kompute-master 2>&1 | grep -i cmake
```

### Problem: "Feature flag conflicts"

**Symptoms:**

- Multiple version conflicts
- Feature resolution errors

**Solution:**

```bash
# Use only supported features for v0.1.0
cargo build --features kompute-master

# For async support
cargo build --features "kompute-master,async"

# Check dependency tree
cargo tree --features kompute-master
```

### Problem: "Out of memory during compilation"

**Symptoms:**

- Compilation killed by system
- Out of memory errors

**Solution:**

```bash
# Limit parallel compilation
export CARGO_BUILD_JOBS=2
cargo build --release

# Use less memory-intensive build
cargo build --release --no-default-features --features kompute-master
```

## Runtime Issues

### Problem: "GPU operations fail at runtime"

**Symptoms:**

- Manager creation succeeds but operations fail
- Vulkan validation errors

**Solution:**

```bash
# Enable validation layers
export VK_LAYER_ENABLES=VK_LAYER_KHRONOS_validation
export RUST_LOG=debug

# Run with debug output
cargo run --example basic_gpu_computing --features debug 2>&1 | head -100

# Check GPU memory
vulkaninfo | grep -i memory
```

### Problem: "Tensor operations return wrong results"

**Symptoms:**

- GPU computation results incorrect
- Data corruption

**Solution:**

```rust
// Add explicit synchronization
manager.eval_sequence(&sequence)?;
sequence.eval_await(5000)?; // Wait up to 5 seconds

// Verify data transfer
let data_before = tensor.to_vec()?;
// ... GPU operation ...
let data_after = tensor.to_vec()?;
println!("Before: {:?}, After: {:?}", data_before, data_after);
```

### Problem: "Memory leaks detected"

**Symptoms:**

- GPU memory usage grows
- Resource cleanup warnings

**Solution:**

```rust
// Ensure proper Drop implementation
{
    let manager = Manager::new()?;
    let tensor = Tensor::new(&mut manager, &[1.0f32; 1000])?;
    // Resources cleaned up automatically when scope ends
}

// Test with memory tracking
cargo test --features debug memory_safety_resource_leak_tests -- --nocapture
```

## Performance Issues

### Problem: "GPU operations are slow"

**Symptoms:**

- Operations take longer than expected
- Poor GPU utilization

**Solution:**

```rust
// Optimize workgroup size
let workgroup = WorkgroupConfig::optimal_1d(data.len());

// Use batched operations
let mut sequence = manager.sequence()?;
sequence.record_tensor_sync(&tensor_a)?;
sequence.record_algorithm(&algorithm)?;
sequence.record_tensor_sync(&tensor_out)?;
sequence.eval()?; // Single GPU submission
```

### Problem: "Memory transfer bottlenecks"

**Symptoms:**

- Slow data upload/download
- High memory usage

**Solution:**

```rust
// Use memory-mapped tensors for large data
let tensor = manager.tensor_uninit::<f32>(large_size)?;

// Minimize CPU-GPU transfers
// ... process multiple operations on GPU before downloading ...
let final_result = output_tensor.to_vec()?;
```

## Testing and Debugging

### Enable Debug Features

```bash
# Run with all debug features
cargo test --features "kompute-master,debug" -- --nocapture

# Specific test categories
cargo test --test gpu_compute_integration -- --nocapture
cargo test --test memory_safety_resource_leak_tests -- --nocapture
```

### Validation Layers

```bash
# Enable Vulkan validation
export VK_LAYER_ENABLES=VK_LAYER_KHRONOS_validation
export VK_LOADER_DEBUG=all

# Run example with validation
cargo run --example basic_gpu_computing --features validation
```

### Performance Profiling

```bash
# Run benchmarks
cargo test --release benchmark_tests -- --nocapture

# Profile specific operations
cargo run --example basic_gpu_computing --release
```

## Getting Additional Help

### Diagnostic Information to Collect

When reporting issues, include:

```bash
# System information
uname -a
rustc --version

# Vulkan information
vulkaninfo --summary

# GPU information
# macOS
system_profiler SPDisplaysDataType | grep -A 10 "Chipset Model"

# Linux
lspci | grep -i vga
lscpu

# Windows
dxdiag /t dxdiag.txt
```

### Useful Environment Variables

```bash
# Maximum debug output
export RUST_LOG=debug
export VK_LOADER_DEBUG=all
export VK_LAYER_ENABLES=VK_LAYER_KHRONOS_validation

# Komputers-specific
export KOMPUTE_DISABLE_MOLTENVK_AUTO=1  # macOS only

# Test with minimal features
cargo test --no-default-features --features kompute-master
```

### Community Resources

- **GitHub Issues**: [Report bugs and get help](https://github.com/axiomantic/komputers/issues)
- **Documentation**: [docs.rs/kompute](https://docs.rs/kompute)
- **Examples**: Check the `examples/` directory for working code
- **Architecture**: See `SYSTEM_ARCHITECTURE.md` for technical details

### Known Limitations

1. **Kompute Version Support**: v0.1.0 only supports Kompute master branch
2. **Platform Testing**: Extensive testing on Apple Silicon macOS, limited on other platforms
3. **GPU Vendor Support**: Comprehensive Apple Silicon support, others community-tested
4. **AddressSanitizer**: Disabled on macOS due to linker compatibility

---

If none of these solutions work, please [open an issue](https://github.com/axiomantic/komputers/issues/new/choose) with:

- Complete error output
- Environment information (`rustc --version`, `vulkaninfo --summary`)
- Minimal code that reproduces the problem
- Steps you've already tried
