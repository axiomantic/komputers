# Kompute-rs: System Architecture (v2)

## 1. Core Philosophy & Guiding Principles

Kompute-rs is built upon the foundation of memory-safe systems programming with zero-cost abstractions for GPU computing. The system embodies the principle that GPU compute power should be accessible through idiomatic Rust interfaces without sacrificing performance or safety.

### 1.1. Guiding Technical Principles

The implementation is governed by the following non-negotiable principles:

- **Principle of Excellence**: All components must meet the highest standards of quality, reliability, and maintainability.
- **Principle of Clarity**: Code and architecture must be self-documenting and comprehensible to future maintainers.
- **Principle of Robustness**: The system must handle edge cases, failures, and unexpected conditions gracefully.
- **Principle of Integration**: All components must work harmoniously within the larger system architecture.
- **Principle of Evolution**: The architecture must support growth, change, and continuous improvement.
- **Principle of Safety**: Memory safety and resource management must be guaranteed at compile time wherever possible.
- **Principle of Performance**: Zero-cost abstractions with minimal overhead compared to C++ API.

## 2. Architectural Overview

### 2.1. High-Level Architecture

The system is structured around a three-layer architecture with clear separation of concerns, providing memory safety, idiomatic Rust APIs, and high performance GPU computing capabilities.

**Core Components:**

- **FFI Layer (kompute-sys)**: Low-level C++ bindings and unsafe FFI interface to Kompute C++ library
- **Safe Wrapper Layer**: Memory-safe Rust wrappers with RAII resource management
- **Idiomatic API Layer**: High-level, ergonomic Rust interface following Rust conventions

### 2.2. System Boundaries and Interfaces

The system interfaces with:

- **Kompute C++ Library**: Direct FFI bindings to core GPU compute functionality
- **Vulkan Drivers**: Cross-vendor GPU support through Vulkan API
- **Rust Ecosystem**: Integration with standard Rust error handling, async, and type systems

## 3. Component Architecture

### 3.1. FFI Layer (kompute-sys)

**Purpose**: Provide low-level, unsafe bindings to Kompute C++ API
**Location**: `kompute-sys/` crate
**Responsibilities**:

- Generate C++ bindings using bindgen
- Handle dynamic library loading and linking
- Manage Vulkan dependency integration
- Provide minimal unsafe FFI wrapper functions

**Key Design Decisions**:

- Use CXX for type-safe C++ integration (not bindgen)
- Handle-based resource management to solve SharedPtr mutable access limitations
- Thread-safe global registry for GPU resource tracking
- Comprehensive safety validation at all API boundaries
- Clear documentation of safety requirements for all unsafe functions

### 3.2. Safe Wrapper Layer

**Purpose**: Provide memory-safe wrappers around FFI layer with proper resource management
**Location**: `src/` public API modules (manager.rs, tensor.rs, algorithm.rs, sequence.rs)
**Responsibilities**:

- Implement RAII patterns for GPU resource cleanup
- Provide safe interfaces to all core Kompute classes
- Handle error conversion from C++ exceptions to Rust Result types
- Manage object lifetimes and prevent use-after-free bugs

**Integration Points**:

- Direct dependency on kompute-sys FFI layer
- Integration with custom error types and result handling
- Memory management through Drop trait implementations

### 3.3. Idiomatic API Layer

**Purpose**: Provide ergonomic, Rust-idiomatic interface for GPU computing
**Location**: `src/` public API modules
**Responsibilities**:

- Builder patterns for complex object creation
- Async support for non-blocking GPU operations
- Iterator-based APIs where appropriate
- Integration with Rust's type system (generics, traits)

**Critical Characteristics**:

- Zero-cost abstractions with compile-time optimization
- Type-safe tensor operations with generic data types
- Fluent API design following Rust conventions

## 4. Data Architecture

### 4.1. Data Models

The system uses strongly-typed data structures with generic parameters to ensure type safety and performance optimization.

**Core Data Structures**:

```rust
pub struct Manager {
    handle: UniquePtr<ffi::Manager>, // CXX-managed C++ object
}

pub struct Tensor<T> {
    handle: u64,                     // Handle-based resource management
    size: usize,
    _phantom: PhantomData<T>,        // Type safety marker
}

pub struct Algorithm {
    handle: u64,                     // Handle to C++ Algorithm via registry
}

pub struct Sequence {
    handle: u64,                     // Handle to C++ Sequence via registry
}
```

### 4.2. Handle-Based Resource Management

The system employs a sophisticated handle-based pattern to overcome C++ SharedPtr mutable access limitations:

```
Rust API Layer    →    C++ Bridge Layer    →    Kompute C++ Library
┌─────────────┐       ┌─────────────────┐       ┌─────────────────────┐
│ Tensor<T>   │  ──→  │ HandleRegistry  │  ──→  │ shared_ptr<Tensor>  │
│ Algorithm   │  ──→  │ (Thread-safe)   │  ──→  │ shared_ptr<Algorithm│
│ Sequence    │  ──→  │ Global Maps     │  ──→  │ shared_ptr<Sequence>│
└─────────────┘       └─────────────────┘       └─────────────────────┘
```

### 4.3. Multi-Type Tensor Architecture

Phase 3 introduced comprehensive multi-type tensor support with runtime type detection:

```rust
// Runtime type detection system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Float32,    // 32-bit IEEE 754 floating point
    Float64,    // 64-bit IEEE 754 floating point
    Int32,      // 32-bit signed integer
    UInt32,     // 32-bit unsigned integer
    Unknown,    // Error state for type detection
}

// Enhanced tensor creation methods
impl Manager {
    pub fn tensor_filled<T>(&mut self, size: usize, fill_value: T) -> Result<Tensor<T>>
    pub fn tensor_zeros<T>(&mut self, size: usize) -> Result<Tensor<T>>
    pub fn tensor_ones<T>(&mut self, size: usize) -> Result<Tensor<T>>
}

// Runtime type introspection
impl<T: TensorData> Tensor<T> {
    pub fn data_type(&self) -> Result<DataType>          // Runtime type detection
    pub fn compile_time_data_type() -> DataType          // Compile-time type info
    pub fn validate_type(&self) -> Result<()>            // Type consistency validation
}
```

**Type Safety Guarantees**:

- Compile-time type safety through generics and phantom types
- Runtime type validation for C++ FFI boundary safety
- Automatic type conversion policies for cross-language operations
- Zero-cost abstractions maintaining performance parity with C++

### 4.4. Data Flow

GPU compute operations follow a handle-based data flow pattern:

1. **Resource Creation**: Rust creates handle, C++ registry stores shared_ptr
2. **Operation Recording**: Handles are resolved to C++ objects for GPU operations
3. **GPU Execution**: Actual Kompute C++ API calls (sequence.eval(), tensor.vector())
4. **Resource Cleanup**: RAII Drop traits remove handles, C++ objects auto-cleanup

## 5. Quality Assurance & Testing Strategy

### 5.1. Testing Philosophy

The system employs a comprehensive testing strategy based on comprehensive unit and integration testing with actual GPU operations.

**Testing Layers**:

- **Unit Tests**: Test individual components in isolation using mock GPU operations where possible
- **Integration Tests**: Full GPU integration tests across different vendors (AMD, NVIDIA, Intel)
- **Performance Tests**: Benchmarks comparing performance against C++ baseline

### 5.2. Quality Gates

All components must pass through these quality gates:

1. **Code Quality**: `cargo fmt --check; cargo clippy -- -D warnings`
2. **Test Coverage**: Minimum 80% test coverage for all public APIs
3. **Performance**: <5% overhead compared to equivalent C++ Kompute operations
4. **Security**: All unsafe code blocks must be documented and audited

## 6. Development Workflow

### 6.1. Component Lifecycle

Each component follows this development lifecycle:

1. **Planning**: Analyze C++ API and design safe Rust interface
2. **Implementation**: Implement FFI bindings, safe wrappers, and idiomatic APIs
3. **Testing**: Comprehensive unit tests, integration tests with real GPU operations
4. **Integration**: Ensure component integrates cleanly with existing architecture
5. **Documentation**: Complete rustdoc documentation with examples

### 6.2. Quality Control Process

All code must pass cargo fmt, cargo clippy, and comprehensive test suite including GPU integration tests before merge.

## 7. Error Handling & System Resilience

### 7.1. Error Handling Philosophy

**Core Principle**: All GPU operations must have explicit error handling with informative error messages that help developers understand and resolve issues.

### 7.2. Error Categories

The system recognizes these error categories:

- **GPU Hardware Errors**: Device lost, out of memory, driver issues
- **Vulkan API Errors**: Invalid operations, resource allocation failures
- **Application Logic Errors**: Invalid shader parameters, type mismatches

### 7.3. Recovery Strategies

Where possible, provide fallback mechanisms and clear error reporting. Critical errors should fail fast with detailed context information.

## 8. Performance & Scalability

### 8.1. Performance Requirements

The system must meet these performance targets:

- **Memory Allocation Overhead**: <1% compared to direct Vulkan allocation
- **Compute Dispatch Latency**: <5% overhead compared to C++ Kompute
- **Memory Transfer Efficiency**: Near-native performance for host-device transfers

### 8.2. Scalability Considerations

Support for multiple GPU devices, concurrent operations across queue families, and efficient batch processing of compute operations.

## 9. Security Architecture

### 9.1. Security Principles

- **Memory Safety First**: Minimize unsafe code and provide safe abstractions
- **Resource Lifecycle Management**: Prevent resource leaks through RAII patterns
- **Input Validation**: Validate all shader parameters and tensor dimensions

### 9.2. Security Implementation

All public APIs are memory safe. Unsafe code is isolated to FFI layer with comprehensive documentation and safety contracts.

## 10. Documentation and Knowledge Management

### 10.1. Documentation Strategy

The project maintains these documentation artifacts:

- **`SYSTEM_ARCHITECTURE.md` (This Document)**: The architectural constitution and single source of truth
- **`CONTRIBUTING.md`**: Development workflow and contribution guidelines
- **`GRIMOIRE.md`**: Implementation templates and patterns
- **`API_EXTENSIONS_TODO.md`**: Roadmap of unimplemented API features
- **`API_DESIGN_PATTERNS.md`**: Reference patterns for safe API implementation
- **`GLOSSARY.md`**: GPU computing terminology and concept definitions

### 10.2. Documentation Integration Protocol

All development must reference these documents appropriately:

- Consult `SYSTEM_ARCHITECTURE.md` for architectural principles and component relationships
- Follow workflow defined in `CONTRIBUTING.md` for development methodology
- Use `GRIMOIRE.md` as implementation template for all development work
- Reference `API_EXTENSIONS_TODO.md` for implementation priorities and missing features
- Reference `GLOSSARY.md` for precise terminology and concept definitions

## 11. Implementation Status & Evolution

### 11.1. Current Implementation Status (v0.2.0)

**✅ Fully Implemented:**

- Build system with Kompute C++ v0.9.0 integration
- Handle-based resource management with thread-safe registry
- Core tensor operations (create, data access, cleanup)
- Core manager operations (device enumeration, creation)
- Core sequence operations (creation, recording, execution)
- Comprehensive error handling with 11 error types
- Basic async API (polling-based, not true GPU async)
- **Multi-type tensor support (f32, f64, i32, u32)** with runtime type detection
- **Enhanced tensor creation methods** (filled, zeros, ones) with type-safe APIs
- **Algorithm-tensor binding system** with template-based parameter passing
- **Cross-platform memory optimization** with vendor-specific strategies
- **Error recovery and resilience** with graceful degradation patterns

**❌ Critical Gaps:**

- Manager::create_queue() method (required for MultiQueueManager)
- Real async GPU integration (current implementation is polling simulation)
- Advanced algorithm configuration (workgroups, specialization constants)
- Performance optimization for large tensor operations

### 11.2. Implementation Roadmap

- **✅ Phase 1 (Completed)**: Algorithm-tensor binding system and foundational GPU operations
- **✅ Phase 2 (Completed)**: Advanced GPU features with cross-platform memory optimization
- **✅ Phase 3 (Completed)**: Multi-type tensor support with enhanced creation methods and runtime type detection
- **Phase 4 (Next)**: Real async GPU integration with Vulkan synchronization primitives
- **Phase 5 (Future)**: Advanced memory management and performance optimization
- **Phase 6 (Future)**: Multi-queue architecture and advanced algorithm configuration

### 11.2. Evolution Strategy

Maintain backward compatibility while adding new features. Use semantic versioning and clear deprecation notices for API changes.

## 12. Success Criteria

The architecture is successful when:

1. **Functional Excellence**: All components work correctly and meet GPU computing requirements
2. **Quality Standards**: Code meets established quality gates and Rust best practices
3. **Integration Harmony**: All components work together seamlessly across GPU vendors
4. **Performance Targets**: System achieves near-native performance with safety guarantees
5. **Maintainability**: Code is clean, well-documented, and follows Rust conventions
6. **Reliability**: System operates reliably under various GPU hardware configurations

---

**The foundation of excellent software lies in thoughtful architecture, disciplined implementation, and unwavering commitment to quality. Let this document serve as the guiding principles for building a system that stands the test of time.**

_"Unleashing GPU Power with Rust Safety"_
