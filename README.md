# Komputers: Experimental GPU Computing for Rust

[![Crates.io](https://img.shields.io/crates/v/komputers.svg)](https://crates.io/crates/komputers)
[![Documentation](https://docs.rs/komputers/badge.svg)](https://docs.rs/komputers)
[![CI](https://github.com/axiomantic/komputers/workflows/CI/badge.svg)](https://github.com/axiomantic/komputers/actions/workflows/ci.yml)
[![Security Audit](https://github.com/axiomantic/komputers/workflows/Security%20Audit/badge.svg)](https://github.com/axiomantic/komputers/actions/workflows/security.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Experimental Rust bindings for the [Kompute](https://github.com/KomputeProject/kompute) GPU compute framework.**

Komputers provides memory-safe GPU computing with Vulkan. Features multi-type tensors, error recovery, and memory optimization. Currently in active development and used in our own projects.

## ⚠️ Experimental Status

**🧪 EXPERIMENTAL - ALPHA QUALITY**

Komputers is an **experimental GPU computing library** that we believe is close to being ready but needs community testing. We are currently using it in our own projects, but it should be considered alpha quality.

**Version Support**: Currently supports Kompute master branch only (unreleased). Official releases v0.8.1 and v0.9.0 are not supported due to compatibility issues.

**Current Features**:

- ✅ **Multi-Type Tensor Support**: f32, f64, i32, u32 with full GPU integration
- ✅ **Algorithm-Tensor Binding**: Real GPU compute operations with SPIR-V shaders
- ✅ **Multi-Queue GPU Operations**: Parallel GPU execution across multiple queues
- ✅ **Advanced Error Recovery**: Production-grade error handling and recovery
- ✅ **Memory Optimization**: Intelligent GPU memory management and optimization
- ✅ **Cross-Platform**: Native support for Linux, Windows, macOS (via MoltenVK)
- ✅ **Async GPU Operations**: Futures-based async API for non-blocking GPU operations
- ✅ **Comprehensive Testing**: 40+ test suites including performance regression and memory safety

## Features

### Core GPU Computing

- **Memory Safety**: All GPU resource management through RAII patterns with automatic cleanup
- **Type Safety**: Generic tensor types with compile-time validation and runtime type detection
- **Zero-Cost Abstractions**: Minimal overhead compared to C++ Kompute
- **Handle-Based Architecture**: Efficient resource management with automatic lifecycle tracking

### Advanced Capabilities

- **Multi-Type Tensors**: Full support for f32, f64, i32, u32 data types
- **Algorithm Configuration**: SPIR-V shader binding with tensor parameters and workgroups
- **Memory Optimization**: Intelligent allocation strategies and fragmentation management
- **Error Recovery**: Automatic error detection, diagnosis, and recovery strategies
- **Platform Integration**: Vulkan environment auto-configuration and validation

### GPU Vendor Support

- **NVIDIA**: Native Vulkan with optimized performance characteristics
- **AMD**: Full RDNA architecture support with vendor-specific optimizations
- **Intel**: Integrated graphics support with shared memory optimization
- **Apple Silicon**: MoltenVK integration with unified memory architecture

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
komputers = "0.1"

# Optional: Enable async GPU operations
# komputers = { version = "0.1", features = ["async"] }
```

### Basic GPU Computing

```rust
use komputers::{Manager, Tensor, Algorithm};

fn main() -> komputers::Result<()> {
    // Create GPU manager (automatically selects best available device)
    let mut manager = Manager::new()?;

    // Create multi-type tensors
    let input_a = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0, 4.0])?;
    let input_b = Tensor::new(&mut manager, &[5.0f32, 6.0, 7.0, 8.0])?;
    let mut output: Tensor<f32> = Tensor::uninit(&mut manager, 4)?;

    // Load SPIR-V compute shader and bind tensors
    let spirv_data = include_bytes!("shaders/vector_add.spv");
    let algorithm = Algorithm::with_tensors(
        &mut manager,
        spirv_data,
        &[&input_a, &input_b, &output]
    )?;

    // Execute GPU computation
    let mut sequence = manager.sequence()?;
    sequence.record_algorithm(&algorithm)?;
    sequence.eval()?;

    // Retrieve results from GPU
    let results = output.to_vec()?;
    println!("GPU Results: {:?}", results); // [6.0, 8.0, 10.0, 12.0]

    Ok(())
}
```

### Multi-Type Tensor Operations

```rust
use komputers::{Manager, Tensor, DataType};

fn multi_type_example() -> komputers::Result<()> {
    let mut manager = Manager::new()?;

    // Create different data type tensors
    let f64_tensor = Tensor::new(&mut manager, &[1.0f64, 2.0, 3.0])?;
    let i32_tensor = Tensor::new(&mut manager, &[10i32, 20, 30])?;
    let u32_tensor = Tensor::new(&mut manager, &[100u32, 200, 300])?;

    // Runtime type detection
    assert_eq!(f64_tensor.data_type()?, DataType::Float64);
    assert_eq!(i32_tensor.data_type()?, DataType::Int32);
    assert_eq!(u32_tensor.data_type()?, DataType::UInt32);

    // Create filled tensors for initialization
    let zeros: Tensor<f32> = manager.tensor_filled(1000, 0.0)?;
    let ones: Tensor<i32> = manager.tensor_filled(500, 1)?;

    println!("Multi-type tensors created successfully!");
    Ok(())
}
```

### High-Level GPU Operations

```rust
use komputers::{GpuOperations, WorkgroupConfig};

fn high_level_example() -> komputers::Result<()> {
    // High-level operations manager with automatic optimization
    let mut ops = GpuOperations::new()?;

    let source = Tensor::new(ops.manager(), &[1.0f32; 1000])?;
    let mut dest = Tensor::uninit::<f32>(ops.manager(), 1000)?;

    // Optimized GPU copy with automatic workgroup sizing
    ops.copy(&source, &mut dest)?;

    // Custom workgroup configuration for advanced use cases
    let workgroup = WorkgroupConfig::optimal_1d(1000);
    println!("Optimal workgroup: {}x{}x{}", workgroup.x, workgroup.y, workgroup.z);

    Ok(())
}
```

### Async GPU Operations

```rust
#[cfg(feature = "async")]
use komputers::{AsyncGpuExecutor, AsyncSequence};

#[cfg(feature = "async")]
async fn async_gpu_example() -> komputers::Result<()> {
    let mut executor = AsyncGpuExecutor::new().await?;

    let input = Tensor::new(executor.manager(), &[1.0f32; 100])?;
    let mut output = Tensor::uninit::<f32>(executor.manager(), 100)?;

    // Non-blocking GPU operations
    let async_sequence = AsyncSequence::new(&mut executor)?;
    async_sequence.record_copy(&input, &mut output).await?;

    // Parallel execution across multiple GPU queues
    executor.execute_parallel(vec![async_sequence], Duration::from_secs(5)).await?;

    println!("Async GPU operations completed!");
    Ok(())
}
```

### Error Recovery and Memory Optimization

```rust
use komputers::{MemoryOptimizer, ErrorRecoveryManager};

fn advanced_features_example() -> komputers::Result<()> {
    // Advanced memory optimization
    let memory_optimizer = MemoryOptimizer::new()?;
    let stats = memory_optimizer.memory_statistics()?;
    println!("GPU Memory: {}MB allocated, {:.1}% fragmentation",
             stats.total_allocated / 1024 / 1024,
             stats.average_fragmentation * 100.0);

    // Error recovery and diagnostics
    let error_recovery = ErrorRecoveryManager::new();
    let diagnostic = error_recovery.generate_diagnostic_report()?;
    println!("System Status: {:?}", diagnostic.overall_health);

    Ok(())
}
```

## Architecture

Komputers follows a three-layer architecture optimized for safety and performance:

### Core Components

- **[`Manager`]**: GPU device management, resource allocation, and multi-queue coordination
- **[`Tensor<T>`]**: Type-safe GPU memory with automatic host-device synchronization
- **[`Algorithm`]**: SPIR-V compute shaders with parameter binding and workgroup configuration
- **[`Sequence`]**: Command buffer for batched GPU operations with async execution support

### Advanced Components

- **[`GpuOperations`]**: High-level GPU computing patterns with automatic optimization
- **[`MemoryOptimizer`]**: Intelligent GPU memory management with fragmentation reduction
- **[`ErrorRecoveryManager`]**: Production-grade error handling with automatic recovery
- **[`MultiQueueManager`]**: Parallel GPU execution across multiple command queues
- **[`HandleTracker`]**: Resource lifecycle tracking with leak detection

## Platform Support

### Vulkan Drivers

- **Linux**: Native Vulkan drivers (NVIDIA, AMD, Intel)
- **Windows**: Native Vulkan drivers with full DirectX integration
- **macOS**: MoltenVK with Metal backend integration

#### MoltenVK Compatibility (macOS)

Komputers includes automatic compatibility fixes for MoltenVK on macOS:

- **Auto-Configuration**: Intelligent MoltenVK detection and environment setup
- **Version Compatibility**: Automatic handling of MoltenVK API version mismatches
- **Metal GPU Detection**: Automatic detection of Metal-capable GPUs with fallback options
- **Build Flags**: Pre-configured with `KOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK=ON`

**Compatibility Notes**: This library targets the Kompute master branch due to compatibility issues with official releases. For details, see [MoltenVK Issue #2292](https://github.com/KhronosGroup/MoltenVK/issues/2292) and [Kompute Issue #378](https://github.com/KomputeProject/kompute/issues/378).

**User Override**: Set `KOMPUTE_DISABLE_MOLTENVK_AUTO=1` to disable automatic MoltenVK configuration.

### GPU Vendors

- **NVIDIA**: GeForce, Quadro, Tesla series with CUDA interop
- **AMD**: RDNA, RDNA2, RDNA3 architectures with ROCm integration
- **Intel**: UHD, Iris, Arc graphics with oneAPI compatibility
- **Apple**: M1, M2, M3 with unified memory optimization

### Automatic Environment Setup

```rust
use komputers::vulkan_setup::{configure_vulkan_environment, diagnose_vulkan_environment};

// Automatic Vulkan configuration
configure_vulkan_environment()?;

// Environment diagnostics
let report = diagnose_vulkan_environment()?;
println!("Vulkan Status: {}", report.status);
```

## Performance

Komputers aims to deliver good performance through:

- **Zero-Cost Abstractions**: Compile-time optimization with minimal runtime overhead
- **Efficient Memory Management**: GPU memory pooling and automatic defragmentation
- **Optimized Data Transfer**: Minimized host-GPU memory copies with staging buffers
- **Workgroup Optimization**: Automatic workgroup sizing based on GPU architecture
- **Multi-Queue Parallelism**: Concurrent GPU execution across multiple command queues

_Performance benchmarks are planned for future releases._

## Safety Guarantees

### Memory Safety

- **No GPU Memory Leaks**: RAII-based automatic resource cleanup
- **No Use-After-Free**: Compile-time lifetime checking with borrow checker
- **No Buffer Overruns**: Bounds checking on all tensor operations
- **Thread Safety**: Safe concurrent access with automatic synchronization

### Error Handling

- **Comprehensive Error Recovery**: Automatic detection and recovery from GPU errors
- **Detailed Error Context**: Clear error messages with actionable guidance
- **Graceful Degradation**: Fallback strategies for hardware compatibility issues
- **Resource Cleanup**: Guaranteed cleanup even in error conditions

## Testing & Validation

Komputers includes comprehensive testing suites:

### Test Coverage

- **40+ Test Suites**: Unit, integration, and end-to-end testing
- **Memory Safety**: AddressSanitizer and LeakSanitizer integration
- **Performance Regression**: Automated benchmarking with regression detection
- **Cross-Platform**: Validation across Linux, Windows, macOS
- **GPU Vendor Matrix**: Testing across NVIDIA, AMD, Intel, Apple GPUs

### Continuous Integration

```bash
# Run comprehensive test suite (excludes benchmarks for faster builds)
cargo test --release --features "async,glsl,debug,validation,test-utils"

# Memory safety validation (if asan feature is available)
# cargo test --features asan

# Performance regression testing
cargo test --test performance_regression_suite -- --nocapture

# Cross-platform compatibility
cargo test --test gpu_vendor_compatibility_matrix

# Run benchmarks (heavier compilation with criterion)
cargo test --features "benchmarks" --benches
```

## Documentation

### API Documentation

- **[docs.rs](https://docs.rs/komputers)**: Official API documentation with examples, tutorials, and version selector

## Examples and Tutorials

### Complete Examples

- **[Basic GPU Computing](komputers/examples/basic_gpu_computing.rs)**: Simple vector operations
- **[Advanced GPU Operations](komputers/examples/advanced_gpu_operations.rs)**: Complex workflows and optimization
- **[Multi-Type Tensors](komputers/examples/multi_type_tensors.rs)**: Working with different data types
- **[Async Operations](komputers/examples/async_gpu_operations.rs)**: Non-blocking GPU computations
- **[Scientific Computing](komputers/examples/scientific_computing.rs)**: Mathematical operations and algorithms
- **[Machine Learning Operations](komputers/examples/machine_learning_operations.rs)**: ML-focused GPU computing
- **[Image Processing Pipeline](komputers/examples/image_processing_pipeline.rs)**: GPU-accelerated image processing
- **[Defensive Guards Demo](komputers/examples/defensive_guards_demo.rs)**: Error handling and safety patterns

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/axiomantic/komputers.git
cd komputers

# Install dependencies (Ubuntu/Debian)
sudo apt install vulkan-dev cmake build-essential

# Install dependencies (macOS)
brew install vulkan-headers vulkan-loader molten-vk cmake

# Run tests (excludes benchmarks for faster builds)
cargo test --features "async,glsl,debug,validation,test-utils"
```

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0)

## Acknowledgments

- [Kompute Project](https://github.com/KomputeProject/kompute) for the excellent C++ GPU computing framework
- [Vulkan](https://www.vulkan.org/) for the cross-platform GPU API
- [MoltenVK](https://github.com/KhronosGroup/MoltenVK) for Metal-to-Vulkan translation on macOS
