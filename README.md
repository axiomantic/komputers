# Kompute-rs

[![Crates.io](https://img.shields.io/crates/v/kompute.svg)](https://crates.io/crates/kompute)
[![Documentation](https://docs.rs/kompute/badge.svg)](https://docs.rs/kompute)
[![Build Status](https://github.com/kompute-rs/kompute-rs/workflows/CI/badge.svg)](https://github.com/kompute-rs/kompute-rs/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Idiomatic Rust bindings for the [Kompute](https://github.com/KomputeProject/kompute) GPU compute framework.

Kompute is a general-purpose GPU computing framework built on Vulkan that provides simple abstractions for GPU memory management and compute shader execution. This crate provides memory-safe, idiomatic Rust bindings with zero-cost abstractions.

## Features

- **Memory Safety**: All GPU resource management handled through RAII patterns
- **Type Safety**: Generic tensor types with compile-time data type validation
- **Cross-Platform**: Supports Linux, Windows, and macOS with Vulkan drivers
- **Zero-Cost Abstractions**: Minimal overhead compared to C++ Kompute
- **Async Support**: Optional futures-based async API (requires `async` feature)

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
kompute = "0.1"
```

### Basic Example

```rust
use kompute::{Manager, Tensor};

fn main() -> kompute::Result<()> {
    // Create a GPU manager
    let mut manager = Manager::new()?;

    // Create tensors with data
    let tensor_a = Tensor::new(&manager, &[1.0f32, 2.0, 3.0])?;
    let tensor_b = Tensor::new(&manager, &[4.0f32, 5.0, 6.0])?;
    let mut tensor_out = Tensor::new(&manager, &[0.0f32; 3])?;

    // Load and execute a compute shader
    let algorithm = manager.algorithm(include_bytes!("shaders/add.spv"))?;
    manager.execute(&algorithm, &[&tensor_a, &tensor_b, &mut tensor_out])?;

    // Retrieve results
    let results = tensor_out.to_vec()?;
    println!("Results: {:?}", results); // [5.0, 7.0, 9.0]

    Ok(())
}
```

## Prerequisites

### Vulkan SDK

You need a Vulkan SDK installed:

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install vulkan-sdk
```

**macOS (Homebrew):**
```bash
brew install vulkan-headers vulkan-loader molten-vk
```

**Windows:**
Download and install the Vulkan SDK from [LunarG](https://www.lunarg.com/vulkan-sdk/).

### GPU Drivers

Ensure you have up-to-date GPU drivers that support Vulkan:

- **NVIDIA**: GeForce 600 series or newer
- **AMD**: Radeon HD 7000 series or newer
- **Intel**: HD Graphics 4000 or newer

## Architecture

The crate is structured around four main components:

- **Manager**: GPU device management and resource allocation
- **Tensor**: Type-safe GPU memory with host synchronization
- **Algorithm**: SPIR-V compute shader with parameter bindings
- **Sequence**: Command buffer for batched GPU operations

## Building from Source

```bash
git clone https://github.com/kompute-rs/kompute-rs.git
cd kompute-rs
git submodule update --init --recursive
cargo build
```

### Build Requirements

- Rust 1.75 or later
- C++17 compatible compiler
- CMake 3.20 or later
- Vulkan SDK

## Features

- `async` - Enable futures-based async API support
- `validation` - Enable Vulkan validation layers (development only)

## Examples

See the [`examples/`](examples/) directory for more comprehensive examples:

- Basic tensor operations
- Custom compute shaders
- Multi-GPU usage
- Async operations

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Kompute Project](https://github.com/KomputeProject/kompute) - The underlying C++ GPU compute framework
- [Vulkan](https://www.vulkan.org/) - The cross-platform GPU API
- [Rust GPU Working Group](https://github.com/rust-gpu/rust-gpu) - Inspiration for GPU computing in Rust