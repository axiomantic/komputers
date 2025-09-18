# API Extensions TODO - Unimplemented Kompute Features

**Purpose:** Reference list of unimplemented API features extracted from comprehensive analysis
**Status:** Implementation roadmap for missing functionality
**Priority:** Core compute features first, advanced features second

---

## HIGH PRIORITY: Core GPU Compute Features

### Manager Multi-Queue Support

```rust
// kompute/src/manager.rs - MISSING IMPLEMENTATIONS
impl Manager {
    pub fn with_queues(device_id: u32, queue_families: &[u32]) -> Result<Self>;
    pub fn with_extensions(device_id: u32, queue_families: &[u32], extensions: &[&str]) -> Result<Self>;
    pub fn create_queue(&mut self, queue_name: &str) -> Result<u32>; // CRITICAL: Called by MultiQueueManager
    pub fn sequence_on_queue(&self, queue_index: u32) -> Result<Sequence>;
    pub fn sequence_with_buffers(&self, queue_index: u32, buffer_count: u32) -> Result<Sequence>;
}
```

### Algorithm-Tensor Binding System

```rust
// kompute/src/manager.rs - CRITICAL GAP
impl Manager {
    pub fn algorithm(
        &self,
        tensors: &[&Tensor<f32>], // MUST ACCEPT ACTUAL TENSORS
        spirv: &[u32],
        workgroup: Workgroup,
        spec_constants: &[f32],
        push_constants: &[f32]
    ) -> Result<Algorithm>;

    pub fn algorithm_from_file(
        &self,
        tensors: &[&Tensor<f32>],
        shader_path: &Path,
        workgroup: Workgroup,
        spec_constants: &[f32],
        push_constants: &[f32]
    ) -> Result<Algorithm>;
}

#[derive(Debug, Clone)]
pub struct Workgroup {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
```

### Algorithm Runtime Configuration

```rust
// kompute/src/algorithm.rs - MISSING FUNCTIONALITY
impl Algorithm {
    pub fn set_tensors(&mut self, tensors: &[&Tensor<f32>]) -> Result<()>;
    pub fn set_workgroup(&mut self, workgroup: Workgroup) -> Result<()>;
    pub fn set_specialization_constants(&mut self, constants: &[f32]) -> Result<()>;
    pub fn set_push_constants(&mut self, constants: &[f32]) -> Result<()>;
    pub fn get_workgroup(&self) -> Workgroup;
    pub fn get_tensors(&self) -> &[u64]; // Tensor handles
}
```

---

## MEDIUM PRIORITY: Enhanced Tensor Operations

### Multi-Type Tensor Support

```rust
// kompute/src/manager.rs - PARTIALLY IMPLEMENTED
impl Manager {
    pub fn tensor_f64(&mut self, data: &[f64]) -> Result<Tensor<f64>>;
    pub fn tensor_i32(&mut self, data: &[i32]) -> Result<Tensor<i32>>;
    pub fn tensor_u32(&mut self, data: &[u32]) -> Result<Tensor<u32>>;
    pub fn tensor_i8(&mut self, data: &[i8]) -> Result<Tensor<i8>>;
    pub fn tensor_u8(&mut self, data: &[u8]) -> Result<Tensor<u8>>;
    pub fn tensor_filled<T: TensorData>(&mut self, size: usize, fill_value: T) -> Result<Tensor<T>>;
}
```

### Direct Tensor Data Access

```rust
// kompute/src/tensor.rs - MISSING METHODS
impl<T: TensorData> Tensor<T> {
    pub fn data(&self) -> Result<&[T]>; // Direct data access
    pub fn data_mut(&mut self) -> Result<&mut [T]>; // Mutable data access
    pub fn as_slice(&self) -> Result<&[T]>;
    pub fn as_mut_slice(&mut self) -> Result<&mut [T]>;
    pub fn get(&self, index: usize) -> Result<Option<&T>>;
    pub fn get_mut(&mut self, index: usize) -> Result<Option<&mut T>>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn data_type(&self) -> DataType; // Runtime type information
    pub fn into_vec(self) -> Result<Vec<T>>; // Move semantics
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    UInt32,
    Int8,
    UInt8,
}
```

---

## MEDIUM PRIORITY: Device Information

### Device Enumeration and Properties

```rust
// kompute/src/manager.rs - MISSING DETAILED INFO
impl Manager {
    pub fn list_devices() -> Result<Vec<DeviceInfo>>;
    pub fn list_queue_families(device_id: u32) -> Result<Vec<QueueFamilyInfo>>;
    pub fn device_properties(&self) -> &DeviceProperties;
    pub fn queue_families(&self) -> &[QueueFamilyInfo];
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub id: u32,
    pub name: String,
    pub device_type: DeviceType,
    pub vendor_id: u32,
    pub api_version: u32,
    pub driver_version: u32,
    pub memory_size: u64,
    pub max_compute_workgroup_size: [u32; 3],
}

#[derive(Debug, Clone)]
pub struct QueueFamilyInfo {
    pub index: u32,
    pub flags: QueueFlags,
    pub queue_count: u32,
}

#[derive(Debug, Clone)]
pub enum DeviceType {
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
    Other,
}
```

---

## LOW PRIORITY: Advanced Features

### Memory Management Control

```rust
// kompute/src/tensor.rs - ADVANCED GPU MEMORY
impl<T: TensorData> Tensor<T> {
    pub fn map_to_host(&mut self) -> Result<()>;
    pub fn map_to_device(&mut self) -> Result<()>;
    pub fn memory_type(&self) -> MemoryType;
    pub fn copy_from(&mut self, other: &Tensor<T>) -> Result<()>;
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryType {
    Device,
    Host,
    Staging,
}
```

### Async GPU Integration (Real, not polling)

```rust
// kompute/src/async_executor.rs - REAL ASYNC GPU
impl AsyncSequenceExecution {
    // Replace polling implementation with actual GPU async integration
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // MUST: Use actual Vulkan semaphores/events, not polling
        // MUST: Integrate with GPU async completion callbacks
        // MUST: Proper async runtime integration
    }
}
```

### GLSL Shader Support

```rust
// kompute/src/manager.rs - SHADER COMPILATION
impl Manager {
    pub fn algorithm_from_glsl(
        &self,
        tensors: &[&Tensor<f32>],
        glsl_source: &str,
        workgroup: Workgroup,
        spec_constants: &[f32],
        push_constants: &[f32]
    ) -> Result<Algorithm>;

    pub fn compile_glsl_to_spirv(glsl_source: &str) -> Result<Vec<u32>>;
}
```

---

## IMPLEMENTATION NOTES

### Critical Dependencies

1. `Manager::create_queue()` - Required for MultiQueueManager to function
2. Algorithm-tensor binding - Required for any actual GPU compute
3. Workgroup specification - Required for algorithm dispatch
4. Multi-type tensors - Required for production use

### Implementation Order

1. **First**: Manager::create_queue() and algorithm-tensor binding
2. **Second**: Workgroup and algorithm configuration
3. **Third**: Multi-type tensor support
4. **Fourth**: Advanced memory management and real async GPU

### Testing Requirements

Each feature MUST include:

- Unit tests with mock GPU operations
- Integration tests with real GPU hardware
- Error condition testing
- Memory safety verification
- Performance benchmarks vs C++ baseline

---

**Status**: This represents approximately 60% of remaining implementation work to achieve full Kompute API coverage. Priority should be given to core compute functionality before advanced features.
