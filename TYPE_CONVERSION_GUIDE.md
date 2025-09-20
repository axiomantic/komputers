# Type Conversion and Compatibility Guide

## Overview

This guide covers type conversion policies, compatibility considerations, and best practices for working with multiple data types in Kompute-rs GPU computing operations.

## Type Safety Architecture

Kompute-rs implements a multi-layered type safety system:

1. **Compile-time Safety**: Rust's type system prevents type errors at compile time
2. **Runtime Validation**: C++ FFI boundary validation ensures type consistency
3. **GPU Compatibility**: Automatic mapping to appropriate Vulkan/SPIR-V types

## Data Type Mapping

### Rust to C++ Type Mapping

| Rust Type | C++ Type   | Vulkan Format          | SPIR-V Type      | Size (bytes) |
| --------- | ---------- | ---------------------- | ---------------- | ------------ |
| `f32`     | `float`    | `VK_FORMAT_R32_SFLOAT` | `OpTypeFloat 32` | 4            |
| `f64`     | `double`   | `VK_FORMAT_R64_SFLOAT` | `OpTypeFloat 64` | 8            |
| `i32`     | `int32_t`  | `VK_FORMAT_R32_SINT`   | `OpTypeInt 32 1` | 4            |
| `u32`     | `uint32_t` | `VK_FORMAT_R32_UINT`   | `OpTypeInt 32 0` | 4            |

### Type ID System

```rust
use kompute::tensor::TensorData;

// Each type has a unique identifier for C++ FFI
assert_eq!(f32::data_type_id(), 0);  // Float32
assert_eq!(f64::data_type_id(), 1);  // Float64
assert_eq!(i32::data_type_id(), 2);  // Int32
assert_eq!(u32::data_type_id(), 3);  // UInt32
```

## Safe Type Conversions

### Explicit Conversions

```rust
use kompute::{Manager, Result, DataType};

fn safe_type_conversions() -> Result<()> {
    let mut manager = Manager::new()?;

    // Create source tensor
    let f32_tensor = manager.tensor_filled(1000, 3.14f32)?;
    let f32_data = f32_tensor.to_vec()?;

    // Explicit conversion to different precision
    let f64_data: Vec<f64> = f32_data.iter().map(|&x| x as f64).collect();
    let f64_tensor = manager.tensor(&f64_data)?;

    // Explicit conversion to integer (with precision loss)
    let i32_data: Vec<i32> = f32_data.iter().map(|&x| x as i32).collect();
    let i32_tensor = manager.tensor(&i32_data)?;

    // Validate the conversions
    f64_tensor.validate_type()?;
    i32_tensor.validate_type()?;

    println!("Original f32: {}", f32_data[0]);
    println!("Converted f64: {}", f64_tensor.to_vec()?[0]);
    println!("Converted i32: {}", i32_tensor.to_vec()?[0]);

    Ok(())
}
```

### Precision-Safe Conversions

```rust
use kompute::{Manager, Result};

fn precision_safe_conversions() -> Result<()> {
    let mut manager = Manager::new()?;

    // Safe widening conversions (no precision loss)
    let f32_data = vec![1.5f32, 2.5f32, 3.5f32];
    let f32_tensor = manager.tensor(&f32_data)?;

    // f32 to f64: Always safe (widening)
    let f64_data: Vec<f64> = f32_tensor.to_vec()?.iter().map(|&x| x as f64).collect();
    let f64_tensor = manager.tensor(&f64_data)?;

    // i32 to f64: Safe for most values (sufficient precision)
    let i32_data = vec![1_000_000i32, 2_000_000i32, 3_000_000i32];
    let i32_tensor = manager.tensor(&i32_data)?;
    let i32_to_f64: Vec<f64> = i32_tensor.to_vec()?.iter().map(|&x| x as f64).collect();

    // Verify precision is maintained
    assert_eq!(f64_data[0], 1.5f64);
    assert_eq!(i32_to_f64[0], 1_000_000.0f64);

    Ok(())
}
```

## Unsafe Conversions and Precision Loss

### Narrowing Conversions

```rust
use kompute::{Manager, Result};

fn narrowing_conversions() -> Result<()> {
    let mut manager = Manager::new()?;

    // f64 to f32: Potential precision loss
    let high_precision = std::f64::consts::PI; // 3.141592653589793
    let f64_tensor = manager.tensor_filled(100, high_precision)?;
    let f64_data = f64_tensor.to_vec()?;

    // Convert to f32 (precision loss)
    let f32_data: Vec<f32> = f64_data.iter().map(|&x| x as f32).collect();
    let f32_tensor = manager.tensor(&f32_data)?;

    println!("Original f64 PI: {:.15}", high_precision);
    println!("Converted f32 PI: {:.15}", f32_data[0] as f64);
    println!("Precision difference: {:.15}", high_precision - (f32_data[0] as f64));

    // Floating point to integer: Major precision loss
    let float_data = vec![3.14159f32, -2.71828f32, 1.41421f32];
    let float_tensor = manager.tensor(&float_data)?;

    let int_data: Vec<i32> = float_tensor.to_vec()?.iter().map(|&x| x as i32).collect();
    assert_eq!(int_data, vec![3, -2, 1]); // Truncation occurs

    Ok(())
}
```

### Overflow Protection

```rust
use kompute::{Manager, Result};

fn overflow_protection() -> Result<()> {
    let mut manager = Manager::new()?;

    // u32 to i32: Potential overflow
    let large_u32 = u32::MAX;
    let u32_tensor = manager.tensor_filled(100, large_u32)?;
    let u32_data = u32_tensor.to_vec()?;

    // Safe conversion with overflow check
    let i32_data: Vec<i32> = u32_data
        .iter()
        .map(|&x| {
            if x <= i32::MAX as u32 {
                x as i32
            } else {
                i32::MAX // Clamp to maximum safe value
            }
        })
        .collect();

    println!("Original u32: {}", large_u32);
    println!("Clamped i32: {}", i32_data[0]);

    // Alternative: Result-based conversion
    let safe_conversion = |x: u32| -> std::result::Result<i32, &'static str> {
        if x <= i32::MAX as u32 {
            Ok(x as i32)
        } else {
            Err("Overflow: u32 value too large for i32")
        }
    };

    match safe_conversion(large_u32) {
        Ok(value) => println!("Converted safely: {}", value),
        Err(error) => println!("Conversion error: {}", error),
    }

    Ok(())
}
```

## GPU Shader Compatibility

### SPIR-V Type Considerations

```glsl
// Example SPIR-V shader with multiple types
#version 450

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) buffer FloatBuffer {
    float float_data[];
};

layout(set = 0, binding = 1) buffer DoubleBuffer {
    double double_data[];
};

layout(set = 0, binding = 2) buffer IntBuffer {
    int int_data[];
};

layout(set = 0, binding = 3) buffer UintBuffer {
    uint uint_data[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;

    // Type-specific operations
    float_data[index] *= 2.0;
    double_data[index] = double(float_data[index]); // Explicit conversion
    int_data[index] = int(float_data[index]);       // Truncation
    uint_data[index] = uint(abs(int_data[index]));  // Safe conversion
}
```

### Rust GPU Binding

```rust
use kompute::{Manager, Algorithm, Sequence, Result};

fn multi_type_gpu_operations() -> Result<()> {
    let mut manager = Manager::new()?;

    // Create tensors matching shader expectations
    let float_tensor = manager.tensor_filled(1024, 1.5f32)?;
    let double_tensor = manager.tensor_zeros::<f64>(1024)?;
    let int_tensor = manager.tensor_zeros::<i32>(1024)?;
    let uint_tensor = manager.tensor_zeros::<u32>(1024)?;

    // Load and create algorithm
    let spirv_bytes = include_bytes!("shaders/multi_type_compute.spv");
    let algorithm = manager.algorithm(spirv_bytes)?;

    // Execute with type-matched tensors
    let mut sequence = manager.sequence()?;
    sequence.record(&algorithm, &[&float_tensor, &double_tensor, &int_tensor, &uint_tensor])?;
    sequence.eval()?;

    // Retrieve results with proper types
    let float_results = float_tensor.to_vec()?;
    let double_results = double_tensor.to_vec()?;
    let int_results = int_tensor.to_vec()?;
    let uint_results = uint_tensor.to_vec()?;

    // Verify type consistency
    assert_eq!(double_results[0], float_results[0] as f64);
    assert_eq!(int_results[0], float_results[0] as i32);
    assert_eq!(uint_results[0], int_results[0].abs() as u32);

    Ok(())
}
```

## Performance Implications

### Memory Bandwidth Considerations

```rust
use kompute::{Manager, Result};
use std::time::Instant;

fn performance_comparison() -> Result<()> {
    let mut manager = Manager::new()?;
    let size = 1_000_000;

    // f32: 4 bytes per element = 4MB total
    let start = Instant::now();
    let f32_tensor = manager.tensor_filled(size, 1.0f32)?;
    let _f32_data = f32_tensor.to_vec()?;
    let f32_time = start.elapsed();

    // f64: 8 bytes per element = 8MB total (2x memory bandwidth)
    let start = Instant::now();
    let f64_tensor = manager.tensor_filled(size, 1.0f64)?;
    let _f64_data = f64_tensor.to_vec()?;
    let f64_time = start.elapsed();

    println!("f32 transfer time: {:?} (4MB)", f32_time);
    println!("f64 transfer time: {:?} (8MB)", f64_time);
    println!("f64/f32 ratio: {:.2}", f64_time.as_nanos() as f64 / f32_time.as_nanos() as f64);

    Ok(())
}
```

### GPU Compute Performance

```rust
use kompute::{Manager, DataType, Result};

fn choose_optimal_type(precision_required: bool, data_range: f64) -> DataType {
    match (precision_required, data_range) {
        // High precision required: use f64
        (true, _) => DataType::Float64,

        // Integer operations with large values: use appropriate int type
        (false, range) if range.fract() == 0.0 => {
            if range >= 0.0 && range <= u32::MAX as f64 {
                DataType::UInt32
            } else if range >= i32::MIN as f64 && range <= i32::MAX as f64 {
                DataType::Int32
            } else {
                DataType::Float64 // Fallback for large integers
            }
        }

        // Default to f32 for best GPU performance
        (false, _) => DataType::Float32,
    }
}

fn optimized_tensor_creation(data: &[f64]) -> Result<()> {
    let mut manager = Manager::new()?;

    // Analyze data characteristics
    let max_value = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_value = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let has_fractional = data.iter().any(|&x| x.fract() != 0.0);
    let range = max_value - min_value;

    // Choose optimal type based on data characteristics
    let optimal_type = if has_fractional {
        if range > f32::MAX as f64 || data.iter().any(|&x| (x as f32 as f64 - x).abs() > 1e-6) {
            DataType::Float64 // Requires double precision
        } else {
            DataType::Float32 // Single precision sufficient
        }
    } else {
        choose_optimal_type(false, range)
    };

    // Create tensor with optimal type
    match optimal_type {
        DataType::Float32 => {
            let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            let tensor = manager.tensor(&f32_data)?;
            println!("Created optimized f32 tensor");
        }
        DataType::Float64 => {
            let tensor = manager.tensor(data)?;
            println!("Created f64 tensor for high precision");
        }
        DataType::Int32 => {
            let i32_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();
            let tensor = manager.tensor(&i32_data)?;
            println!("Created optimized i32 tensor");
        }
        DataType::UInt32 => {
            let u32_data: Vec<u32> = data.iter().map(|&x| x as u32).collect();
            let tensor = manager.tensor(&u32_data)?;
            println!("Created optimized u32 tensor");
        }
        _ => unreachable!(),
    }

    Ok(())
}
```

## Best Practices Summary

### 1. Type Selection Guidelines

- **Default to f32**: Best performance on most GPUs
- **Use f64**: Only when precision is critical
- **Use i32**: For signed integer operations
- **Use u32**: For indices, flags, and unsigned operations

### 2. Conversion Safety

- **Always validate**: Use `validate_type()` in debug builds
- **Handle overflow**: Check ranges before narrowing conversions
- **Document precision loss**: Comment when precision may be lost
- **Test edge cases**: Verify behavior at type boundaries

### 3. Performance Optimization

- **Minimize transfers**: Batch operations to reduce CPU-GPU transfers
- **Choose appropriate types**: Balance precision vs. performance
- **Consider memory bandwidth**: f64 uses 2x memory bandwidth of f32
- **Profile actual workloads**: Measure performance with real data

### 4. Error Handling

```rust
use kompute::{Manager, KomputeError, Result};

fn robust_type_handling() -> Result<()> {
    let mut manager = Manager::new()?;

    // Always handle potential type mismatches
    let tensor = manager.tensor_filled(1000, 1.0f32)?;

    match tensor.validate_type() {
        Ok(()) => {
            // Safe to proceed with operations
            let data = tensor.to_vec()?;
            println!("Tensor validation successful, {} elements", data.len());
        }
        Err(KomputeError::TensorError(msg)) => {
            eprintln!("Type validation failed: {}", msg);
            return Err(KomputeError::tensor("Type consistency error"));
        }
        Err(e) => return Err(e),
    }

    Ok(())
}
```

This comprehensive type system provides the foundation for safe, efficient GPU computing operations across different data types while maintaining Rust's safety guarantees.
