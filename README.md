# Komputers: GPU Computing for Rust

[![Crates.io](https://img.shields.io/crates/v/komputers.svg)](https://crates.io/crates/komputers)
[![Documentation](https://docs.rs/komputers/badge.svg)](https://docs.rs/komputers)
[![CI](https://github.com/axiomantic/komputers/workflows/CI/badge.svg)](https://github.com/axiomantic/komputers/actions/workflows/ci.yml)
[![Security Audit](https://github.com/axiomantic/komputers/workflows/Security%20Audit/badge.svg)](https://github.com/axiomantic/komputers/actions/workflows/security.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Rust bindings for the [Kompute](https://github.com/KomputeProject/kompute) GPU compute framework.

Komputers provides GPU computing with Vulkan through a Rust interface. It includes tensor operations, SPIR-V algorithm support, and memory management for GPU workloads.

## Status

Komputers is under active development. It supports Kompute's master branch. Official Kompute releases (v0.8.1, v0.9.0) are not currently supported due to compatibility requirements.

## Features

- **Multi-Type Tensors**: Support for f32, f64, i32, u32 data types with GPU operations
- **SPIR-V Integration**: Load and execute compute shaders with tensor parameter binding
- **Multi-Queue Support**: Parallel execution across multiple GPU command queues
- **Error Handling**: Structured error types with recovery strategies
- **Memory Management**: GPU memory allocation and cleanup through RAII patterns
- **Cross-Platform**: Linux, Windows, and macOS support (macOS via MoltenVK)
- **Async Operations**: Optional async/await API for non-blocking GPU work
- **Testing Suite**: Unit, integration, and GPU hardware tests

### Implementation Details

- **Memory Safety**: GPU resource management through RAII patterns with automatic cleanup
- **Type Safety**: Generic tensor types with compile-time and runtime type checking
- **Handle-Based Architecture**: Resource management with lifecycle tracking
- **SPIR-V Support**: Compute shader loading and execution with parameter binding
- **Vulkan Backend**: Uses Vulkan API through the Kompute C++ framework

### Platform Support

- **Linux**: Native Vulkan drivers (NVIDIA, AMD, Intel)
- **Windows**: Native Vulkan drivers
- **macOS**: MoltenVK translation layer for Metal backend compatibility
- **GPU Vendors**: NVIDIA, AMD, Intel, and Apple Silicon support

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

Komputers provides a layered architecture for GPU computing:

### Core Components

- **[`Manager`]**: GPU device management and resource allocation
- **[`Tensor<T>`]**: Type-safe GPU memory with host-device synchronization
- **[`Algorithm`]**: SPIR-V compute shader loading and execution
- **[`Sequence`]**: Command buffer for batched GPU operations

### Additional Components

- **[`GpuOperations`]**: High-level GPU operation patterns
- **[`MemoryOptimizer`]**: GPU memory management utilities
- **[`ErrorRecoveryManager`]**: Error handling and recovery
- **[`MultiQueueManager`]**: Multi-queue execution support
- **[`HandleTracker`]**: Resource lifecycle tracking

## Platform Support

### Vulkan Drivers

- **Linux**: Native Vulkan drivers (NVIDIA, AMD, Intel)
- **Windows**: Native Vulkan drivers with full DirectX integration
- **macOS**: MoltenVK with Metal backend integration

#### MoltenVK Compatibility (macOS)

Komputers includes MoltenVK compatibility for macOS:

- **Auto-Configuration**: MoltenVK detection and environment setup
- **Version Compatibility**: Handling of MoltenVK API version differences
- **Metal GPU Detection**: Detection of Metal-capable GPUs with fallback
- **Build Flags**: Pre-configured with `KOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK=ON`

**Compatibility Notes**: This library targets the Kompute master branch due to compatibility issues with official releases. For details, see [MoltenVK Issue #2292](https://github.com/KhronosGroup/MoltenVK/issues/2292) and [Kompute Issue #378](https://github.com/KomputeProject/kompute/issues/378).

**User Override**: Set `KOMPUTE_DISABLE_MOLTENVK_AUTO=1` to disable automatic MoltenVK configuration.

### GPU Vendors

- **NVIDIA**: GeForce, Quadro, Tesla series
- **AMD**: RDNA, RDNA2, RDNA3 architectures
- **Intel**: UHD, Iris, Arc graphics
- **Apple**: M1, M2, M3 processors

### Environment Setup

```rust
use komputers::vulkan_setup::{configure_vulkan_environment, diagnose_vulkan_environment};

// Vulkan configuration
configure_vulkan_environment()?;

// Environment diagnostics
let report = diagnose_vulkan_environment()?;
println!("Vulkan Status: {}", report.status);
```

## Performance

Komputers includes performance features:

- **Rust Zero-Cost Abstractions**: Compile-time optimization with minimal runtime overhead
- **Memory Management**: GPU memory pooling and defragmentation
- **Data Transfer**: Staging buffers to minimize host-GPU memory copies
- **Workgroup Sizing**: Workgroup configuration based on GPU architecture
- **Multi-Queue Support**: Concurrent GPU execution across multiple command queues

## Safety

### Memory Safety

- **GPU Memory Management**: RAII-based automatic resource cleanup
- **Lifetime Checking**: Compile-time lifetime validation with Rust borrow checker
- **Bounds Checking**: Bounds validation on tensor operations
- **Thread Safety**: Safe concurrent access patterns

### Error Handling

- **Error Recovery**: Detection and recovery from GPU errors
- **Error Context**: Error messages with contextual information
- **Hardware Compatibility**: Fallback strategies for compatibility issues
- **Resource Cleanup**: Resource cleanup in error conditions

## Testing

Komputers includes testing suites:

### Test Coverage

- **Test Suites**: Unit, integration, and end-to-end testing
- **Memory Safety**: AddressSanitizer and LeakSanitizer integration
- **Performance Testing**: Benchmarking with regression detection
- **Cross-Platform**: Testing across Linux, Windows, macOS
- **GPU Hardware**: Testing across NVIDIA, AMD, Intel, Apple GPUs

### Test Commands

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

# Run property tests (heavier compilation with proptest)
cargo test --features "property-testing" --test property_tests
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
