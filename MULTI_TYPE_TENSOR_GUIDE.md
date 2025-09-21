# Multi-Type Tensor API Guide

## Overview

Phase 3 of Komputers introduces comprehensive multi-type tensor support with runtime type detection and enhanced creation methods. This guide provides best practices and usage patterns for working with different data types in GPU computing operations.

## Supported Data Types

Komputers supports four fundamental data types for GPU tensor operations:

| Rust Type | DataType Enum       | Description                    | Use Cases                                       |
| --------- | ------------------- | ------------------------------ | ----------------------------------------------- |
| `f32`     | `DataType::Float32` | 32-bit IEEE 754 floating point | General-purpose GPU computing, machine learning |
| `f64`     | `DataType::Float64` | 64-bit IEEE 754 floating point | High-precision scientific computing             |
| `i32`     | `DataType::Int32`   | 32-bit signed integer          | Indexing, counters, signed arithmetic           |
| `u32`     | `DataType::UInt32`  | 32-bit unsigned integer        | Buffer indices, bitwise operations              |

## Enhanced Tensor Creation Methods

### Creating Filled Tensors

```rust
use kompute::{Manager, Result, DataType};

fn create_filled_tensors() -> Result<()> {
    let mut manager = Manager::new()?;

    // Create tensors filled with specific values
    let pi_tensor = manager.tensor_filled(1000, std::f32::consts::PI)?;
    let euler_tensor = manager.tensor_filled(500, std::f64::consts::E)?;
    let answer_tensor = manager.tensor_filled(100, 42i32)?;
    let max_tensor = manager.tensor_filled(200, u32::MAX)?;

    // Verify the filled values
    assert_eq!(pi_tensor.to_vec()?[0], std::f32::consts::PI);
    assert_eq!(euler_tensor.to_vec()?[0], std::f64::consts::E);
    assert_eq!(answer_tensor.to_vec()?[0], 42i32);
    assert_eq!(max_tensor.to_vec()?[0], u32::MAX);

    Ok(())
}
```

### Creating Zero and One Tensors

```rust
use kompute::{Manager, Result};

fn create_special_tensors() -> Result<()> {
    let mut manager = Manager::new()?;

    // Create zero-initialized tensors
    let f32_zeros = manager.tensor_zeros::<f32>(1000)?;
    let i32_zeros = manager.tensor_zeros::<i32>(500)?;

    // Create one-initialized tensors
    let f32_ones = manager.tensor_ones::<f32>(1000)?;
    let u32_ones = manager.tensor_ones::<u32>(500)?;

    // Verify initialization
    assert_eq!(f32_zeros.to_vec()?[0], 0.0f32);
    assert_eq!(i32_zeros.to_vec()?[0], 0i32);
    assert_eq!(f32_ones.to_vec()?[0], 1.0f32);
    assert_eq!(u32_ones.to_vec()?[0], 1u32);

    Ok(())
}
```

## Runtime Type Detection

### Type Introspection

```rust
use kompute::{Manager, Tensor, DataType, Result};

fn demonstrate_type_detection() -> Result<()> {
    let mut manager = Manager::new()?;

    // Create tensors of different types
    let f32_tensor = manager.tensor_filled(100, 3.14f32)?;
    let f64_tensor = manager.tensor_filled(100, 2.718f64)?;
    let i32_tensor = manager.tensor_filled(100, -42i32)?;
    let u32_tensor = manager.tensor_filled(100, 42u32)?;

    // Runtime type detection
    assert_eq!(f32_tensor.data_type()?, DataType::Float32);
    assert_eq!(f64_tensor.data_type()?, DataType::Float64);
    assert_eq!(i32_tensor.data_type()?, DataType::Int32);
    assert_eq!(u32_tensor.data_type()?, DataType::UInt32);

    // Compile-time type information
    assert_eq!(Tensor::<f32>::compile_time_data_type(), DataType::Float32);
    assert_eq!(Tensor::<f64>::compile_time_data_type(), DataType::Float64);
    assert_eq!(Tensor::<i32>::compile_time_data_type(), DataType::Int32);
    assert_eq!(Tensor::<u32>::compile_time_data_type(), DataType::UInt32);

    // Type validation
    f32_tensor.validate_type()?;  // Ensures runtime matches compile-time type
    f64_tensor.validate_type()?;
    i32_tensor.validate_type()?;
    u32_tensor.validate_type()?;

    Ok(())
}
```

### Dynamic Type Handling

```rust
use kompute::{Manager, DataType, Result};

fn handle_dynamic_types(data_type: DataType, size: usize) -> Result<()> {
    let mut manager = Manager::new()?;

    match data_type {
        DataType::Float32 => {
            let tensor = manager.tensor_filled(size, 1.0f32)?;
            println!("Created f32 tensor with {} elements", tensor.size());
        }
        DataType::Float64 => {
            let tensor = manager.tensor_filled(size, 1.0f64)?;
            println!("Created f64 tensor with {} elements", tensor.size());
        }
        DataType::Int32 => {
            let tensor = manager.tensor_filled(size, 1i32)?;
            println!("Created i32 tensor with {} elements", tensor.size());
        }
        DataType::UInt32 => {
            let tensor = manager.tensor_filled(size, 1u32)?;
            println!("Created u32 tensor with {} elements", tensor.size());
        }
        DataType::Unknown => {
            return Err(kompute::KomputeError::tensor("Unknown data type"));
        }
    }

    Ok(())
}
```

## Best Practices for GPU Computing

### Choosing the Right Data Type

1. **f32 (Float32)**: Default choice for most GPU computing tasks

   - Excellent performance on all GPU vendors
   - Sufficient precision for machine learning and graphics
   - Optimal memory bandwidth utilization

2. **f64 (Float64)**: Use for high-precision scientific computing

   - May have reduced performance on consumer GPUs
   - Essential for precision-critical calculations
   - Consider performance implications

3. **i32 (Int32)**: Use for signed integer operations

   - Array indexing with negative values
   - Signed arithmetic operations
   - Counter operations that may go negative

4. **u32 (UInt32)**: Use for unsigned integer operations
   - Buffer indices and array offsets
   - Bitwise operations and flags
   - Always-positive counters

### Performance Considerations

```rust
use kompute::{Manager, Result};

fn performance_best_practices() -> Result<()> {
    let mut manager = Manager::new()?;

    // Prefer larger tensor sizes for better GPU utilization
    let optimal_size = 1024 * 1024; // 1M elements

    // Choose data type based on precision requirements
    let tensor = if requires_high_precision() {
        manager.tensor_filled(optimal_size, 1.0f64)?
    } else {
        manager.tensor_filled(optimal_size, 1.0f32)? // Preferred for performance
    };

    // Validate type consistency for debugging
    #[cfg(debug_assertions)]
    tensor.validate_type()?;

    Ok(())
}

fn requires_high_precision() -> bool {
    // Application-specific logic
    false
}
```

### Error Handling Patterns

```rust
use kompute::{Manager, KomputeError, Result};

fn robust_tensor_creation() -> Result<()> {
    let mut manager = Manager::new()?;

    // Handle potential allocation failures gracefully
    let large_size = 1_000_000_000; // Very large allocation

    match manager.tensor_filled(large_size, 1.0f32) {
        Ok(tensor) => {
            println!("Successfully created large tensor with {} elements", tensor.size());
            tensor.validate_type()?;
        }
        Err(KomputeError::MemoryError(msg)) => {
            eprintln!("Memory allocation failed: {}", msg);
            // Fallback to smaller tensor
            let smaller_tensor = manager.tensor_filled(1000, 1.0f32)?;
            println!("Created fallback tensor with {} elements", smaller_tensor.size());
        }
        Err(e) => return Err(e),
    }

    Ok(())
}
```

## Integration with GPU Operations

### Multi-Type Algorithm Operations

```rust
use kompute::{Manager, Algorithm, Sequence, Result};

fn multi_type_gpu_workflow() -> Result<()> {
    let mut manager = Manager::new()?;

    // Create tensors of different types for different purposes
    let input_data = manager.tensor_filled(1024, 2.0f32)?;    // Input data
    let weights = manager.tensor_filled(1024, 0.5f32)?;       // Model weights
    let indices = manager.tensor_filled(1024, 0u32)?;         // Buffer indices
    let counters = manager.tensor_filled(64, 0i32)?;          // Operation counters

    // Validate all types before GPU operations
    input_data.validate_type()?;
    weights.validate_type()?;
    indices.validate_type()?;
    counters.validate_type()?;

    // Create algorithm and sequence for GPU execution
    let spirv_code = include_bytes!("../examples/shaders/multi_type_compute.spv");
    let algorithm = manager.algorithm(spirv_code)?;
    let mut sequence = manager.sequence()?;

    // Record GPU operations with type-safe tensors
    sequence.record(&algorithm, &[&input_data, &weights, &indices, &counters])?;
    sequence.eval()?;

    // Retrieve results with proper type handling
    let results = input_data.to_vec()?;
    let final_counters = counters.to_vec()?;

    println!("Processed {} elements, final counter value: {}",
             results.len(), final_counters[0]);

    Ok(())
}
```

## Migration from Single-Type Code

### Before (f32 only):

```rust
// Old single-type approach
let tensor = manager.tensor(&[1.0f32, 2.0, 3.0])?;
```

### After (multi-type):

```rust
// New multi-type approach with enhanced creation methods
let f32_tensor = manager.tensor_filled(3, 1.0f32)?;
let f64_tensor = manager.tensor_filled(3, 1.0f64)?;
let i32_tensor = manager.tensor_filled(3, 1i32)?;
let u32_tensor = manager.tensor_filled(3, 1u32)?;

// Or use convenient zero/one initialization
let zeros = manager.tensor_zeros::<f32>(1000)?;
let ones = manager.tensor_ones::<f64>(500)?;
```

## Conclusion

The multi-type tensor system in Komputers Phase 3 provides:

- **Type Safety**: Compile-time and runtime type validation
- **Performance**: Zero-cost abstractions with optimal GPU performance
- **Flexibility**: Support for floating-point and integer operations
- **Reliability**: Comprehensive error handling and graceful degradation
- **Ergonomics**: Convenient creation methods for common patterns

This system enables robust, high-performance GPU computing applications with the safety guarantees of Rust's type system.
