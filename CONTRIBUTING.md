# Contributing to Komputers

We welcome contributions to Komputers! This guide will help you get started with contributing to our GPU computing library for Rust.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Rust 1.70+** with `cargo`
- **Vulkan SDK** installed for your platform
- **CMake 3.15+** for building Kompute C++ dependencies
- **Git** for version control

#### Platform-Specific Requirements

**macOS:**

```bash
brew install vulkan-headers vulkan-loader molten-vk cmake
```

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install vulkan-tools libvulkan-dev vulkan-validationlayers-dev cmake build-essential
```

**Windows:**

- Install [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
- Install [CMake](https://cmake.org/download/)

### Setting Up Your Development Environment

1. **Fork and Clone**

   ```bash
   git clone https://github.com/YOUR_USERNAME/komputers.git
   cd komputers
   ```

2. **Build and Test**

   ```bash
   # Build the project
   cargo build --all-features

   # Run the test suite
   cargo test --all-features

   # Run examples to verify GPU functionality
   cargo run --example basic_gpu_computing
   ```

3. **Verify GPU Integration**

   ```bash
   # Test GPU detection and basic operations
   cargo run --example basic_gpu_computing

   # On macOS, verify MoltenVK is working
   vulkaninfo | head -20
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow Rust best practices and idioms
- Ensure code compiles with `cargo build --all-features`
- Add tests for new functionality
- Update documentation for API changes

### 3. Testing Requirements

All contributions must include appropriate tests:

**Unit Tests:**

```bash
cargo test --lib --all-features
```

**Integration Tests:**

```bash
cargo test --test comprehensive_integration
```

**GPU Hardware Tests:**

```bash
cargo test --test gpu_compute_integration
```

**Documentation Tests:**

```bash
cargo test --doc
```

### 4. Code Quality Standards

Before submitting, ensure your code meets our standards:

```bash
# Format code
cargo fmt

# Check for common mistakes
cargo clippy --all-features -- -D warnings

# Verify documentation builds
cargo doc --all-features --no-deps
```

### 5. Commit Guidelines

Use conventional commit format:

```
feat: add new tensor operation for matrix multiplication
fix: resolve memory leak in algorithm cleanup
docs: update installation guide for Windows
test: add benchmark for large tensor operations
```

## Contribution Areas

### High-Priority Areas

1. **Cross-Platform Testing**

   - Test on Linux distributions
   - Windows compatibility verification
   - GPU vendor compatibility (NVIDIA, AMD, Intel)

2. **Performance Optimization**

   - GPU memory optimization
   - Workgroup size tuning
   - Memory transfer efficiency

3. **Documentation Improvements**
   - More examples and tutorials
   - Platform-specific setup guides
   - Troubleshooting documentation

### Medium-Priority Areas

1. **Additional GPU Operations**

   - New algorithms and kernels
   - Advanced tensor operations
   - ML/AI computational patterns

2. **Developer Experience**

   - Better error messages
   - Debugging utilities
   - Performance profiling tools

3. **Ecosystem Integration**
   - Integration with ML frameworks
   - Interop with other GPU libraries
   - Build system improvements

## API Design Guidelines

### Rust Conventions

- Use idiomatic Rust patterns and naming
- Implement appropriate traits (`Clone`, `Debug`, etc.)
- Use `Result<T, KomputeError>` for fallible operations
- Prefer zero-cost abstractions
- Ensure memory safety through RAII patterns

### GPU Resource Management

- All GPU resources must use RAII cleanup
- Use handle-based APIs for C++ interop
- Implement proper Drop traits for resource cleanup
- Consider async operations for non-blocking GPU work

### Error Handling

- Provide informative error messages with context
- Use appropriate error types from `KomputeError`
- Include platform-specific guidance in error messages
- Ensure graceful cleanup on error conditions

## Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **GPU Tests**: Test actual GPU hardware integration
4. **Benchmark Tests**: Performance regression testing
5. **Property Tests**: Fuzz testing for edge cases

### Writing GPU Tests

When adding GPU-dependent tests:

```rust
#[test]
fn test_gpu_feature() -> kompute::Result<()> {
    // Skip test if no GPU available
    if !kompute::platform::has_vulkan_support()? {
        return Ok(());
    }

    let mut manager = Manager::new()?;
    // Test GPU functionality
    Ok(())
}
```

## Documentation Standards

### API Documentation

- All public APIs must have rustdoc comments
- Include usage examples in doc comments
- Document error conditions and return values
- Explain platform-specific behavior

### Example Code

- All examples must compile and run
- Include error handling in examples
- Show realistic usage patterns
- Test examples in CI/CD

## Review Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code is formatted with `cargo fmt`
- [ ] No clippy warnings
- [ ] Documentation is updated
- [ ] Commit messages follow convention

### Pull Request Requirements

1. **Clear Description**: Explain what and why
2. **Test Coverage**: Include appropriate tests
3. **Documentation**: Update docs for API changes
4. **Platform Testing**: Test on multiple platforms if possible

### Review Criteria

Reviews will focus on:

- **Correctness**: Does the code work as intended?
- **Safety**: Is GPU resource management safe?
- **Performance**: Are there performance implications?
- **API Design**: Is the API consistent and usable?
- **Testing**: Is test coverage adequate?

## Getting Help

### Resources

- **Documentation**: [docs.rs/kompute](https://docs.rs/kompute)
- **Examples**: Check the `examples/` directory
- **Architecture**: See `SYSTEM_ARCHITECTURE.md`

### Community

- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Discord/Matrix**: Real-time community chat (links TBD)

### Common Issues

1. **Build Failures**: Usually Vulkan SDK or CMake issues
2. **GPU Detection**: Verify Vulkan drivers are installed
3. **MoltenVK on macOS**: Check MoltenVK installation
4. **Test Failures**: May require specific GPU hardware

## Release Process

Contributions go through this process:

1. **Feature Development**: Implement and test locally
2. **Pull Request**: Submit for community review
3. **Review Process**: Code review and feedback
4. **Integration**: Merge after approval
5. **Release**: Included in next version release

## License

By contributing to Komputers, you agree that your contributions will be licensed under the Apache License 2.0.

## Recognition

Contributors are recognized in:

- CHANGELOG.md for releases
- GitHub contributor graphs
- Special recognition for significant contributions

---

Thank you for contributing to Komputers! Your help makes GPU computing in Rust better for everyone.
