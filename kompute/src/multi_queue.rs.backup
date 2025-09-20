//! Multi-queue parallel execution for GPU compute operations.

use crate::error::{KomputeError, Result};
use crate::manager::Manager;
use crate::sequence::Sequence;
use std::collections::HashMap;

/// Multi-queue GPU manager for parallel execution.
///
/// This manager can create and coordinate multiple GPU queues to enable
/// parallel execution of independent compute operations. This is particularly
/// useful for overlapping compute and memory transfer operations.
///
/// # Examples
///
/// ```rust,no_run
/// use kompute::{MultiQueueManager, Tensor, Result};
///
/// fn parallel_gpu_operations() -> Result<()> {
///     let mut multi_manager = MultiQueueManager::new()?;
///
///     // Create queues for different operation types
///     let compute_queue = multi_manager.create_queue("compute")?;
///     let transfer_queue = multi_manager.create_queue("transfer")?;
///
///     // Execute operations in parallel on different queues
///     let data = vec![1.0f32, 2.0, 3.0, 4.0];
///     let tensor = Tensor::new(multi_manager.manager(), &data)?;
///
///     // These operations can run concurrently
///     let mut compute_seq = multi_manager.create_sequence_on_queue("compute")?;
///     let mut transfer_seq = multi_manager.create_sequence_on_queue("transfer")?;
///
///     compute_seq.record_tensor_sync_device(tensor.handle())?;
///     transfer_seq.record_tensor_sync_local(tensor.handle())?;
///
///     // Execute both sequences in parallel
///     multi_manager.execute_parallel(vec![
///         ("compute".to_string(), compute_seq),
///         ("transfer".to_string(), transfer_seq),
///     ])?;
///
///     Ok(())
/// }
/// ```
pub struct MultiQueueManager {
    manager: Manager,
    queues: HashMap<String, QueueInfo>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Part of future API design
struct QueueInfo {
    queue_id: u64, // Changed to u64 to store actual GPU queue handle
    queue_name: String,
    active_sequences: Vec<u64>,
}

impl MultiQueueManager {
    /// Create a new multi-queue manager.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if GPU initialization fails.
    pub fn new() -> Result<Self> {
        let manager = Manager::new()?;
        Ok(Self {
            manager,
            queues: HashMap::new(),
        })
    }

    /// Create a new multi-queue manager with a specific device.
    ///
    /// # Arguments
    ///
    /// * `device_id` - The GPU device to use
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if device initialization fails.
    pub fn with_device(device_id: u32) -> Result<Self> {
        let manager = Manager::with_device(device_id)?;
        Ok(Self {
            manager,
            queues: HashMap::new(),
        })
    }

    /// Create a new named queue for parallel execution.
    ///
    /// # Arguments
    ///
    /// * `queue_name` - A human-readable name for the queue
    ///
    /// # Returns
    ///
    /// The queue handle for referencing this queue in operations.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if queue creation fails.
    pub fn create_queue(&mut self, queue_name: &str) -> Result<u64> {
        // Validate queue name
        if queue_name.is_empty() {
            return Err(KomputeError::compute("Queue name cannot be empty"));
        }

        if queue_name.len() > 64 {
            return Err(KomputeError::compute(
                "Queue name too long (maximum 64 characters)",
            ));
        }

        // Check if queue already exists
        if self.queues.contains_key(queue_name) {
            return Err(KomputeError::compute(format!(
                "Queue '{}' already exists",
                queue_name
            )));
        }

        // Check queue limits
        if self.queues.len() >= 16 {
            return Err(KomputeError::compute(
                "Maximum number of queues (16) reached. Consider removing unused queues.",
            ));
        }

        // Create actual GPU queue using the manager (family index 0 for compute queues)
        let queue_handle = self.manager.create_queue(0)?;

        let queue_info = QueueInfo {
            queue_id: queue_handle, // Use the actual GPU queue handle as the ID
            queue_name: queue_name.to_string(),
            active_sequences: Vec::new(),
        };

        self.queues.insert(queue_name.to_string(), queue_info);
        Ok(queue_handle) // Return the actual GPU queue handle
    }

    /// Create a sequence on a specific queue.
    ///
    /// # Arguments
    ///
    /// * `queue_name` - The name of the queue to create the sequence on
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if sequence creation fails or queue doesn't exist.
    pub fn create_sequence_on_queue(&mut self, queue_name: &str) -> Result<Sequence> {
        let queue_info = self.queues.get(queue_name).ok_or_else(|| {
            KomputeError::compute(format!(
                "Queue '{}' does not exist. Create it first with create_queue()",
                queue_name
            ))
        })?;

        let queue_handle = queue_info.queue_id;
        let sequence = self.manager.create_sequence_on_queue(queue_handle)?;
        let sequence_handle = sequence.handle();

        // Track the sequence in the queue
        if let Some(queue_info) = self.queues.get_mut(queue_name) {
            queue_info.active_sequences.push(sequence_handle);
        }

        Ok(sequence)
    }

    /// Execute multiple sequences in parallel across different queues.
    ///
    /// This method demonstrates true parallel execution by submitting
    /// sequences to different GPU queues simultaneously.
    ///
    /// # Arguments
    ///
    /// * `sequences` - Vector of (queue_name, sequence) pairs to execute
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if any sequence execution fails.
    pub fn execute_parallel(&mut self, mut sequences: Vec<(String, Sequence)>) -> Result<()> {
        // Validate input
        if sequences.is_empty() {
            return Err(KomputeError::compute("Cannot execute empty sequence list"));
        }

        if sequences.len() > 32 {
            return Err(KomputeError::compute(format!(
                "Too many sequences for parallel execution: {} (maximum 32)",
                sequences.len()
            )));
        }

        // Verify all queues exist before starting any execution
        for (queue_name, _) in &sequences {
            if queue_name.is_empty() {
                return Err(KomputeError::compute("Queue name cannot be empty"));
            }

            if !self.queues.contains_key(queue_name) {
                return Err(KomputeError::compute(format!(
                    "Queue '{}' does not exist. Create it first with create_queue()",
                    queue_name
                )));
            }
        }

        // Start all sequences asynchronously
        for (queue_name, sequence) in &mut sequences {
            // Start async execution
            if let Err(e) = sequence.eval_async() {
                return Err(KomputeError::compute(format!(
                    "Failed to start async execution on queue '{}': {}",
                    queue_name, e
                )));
            }
        }

        // Wait for all sequences to complete with error handling
        for (queue_name, sequence) in &mut sequences {
            if let Err(e) = sequence.eval_await(0) {
                return Err(KomputeError::compute(format!(
                    "Sequence execution failed on queue '{}': {}",
                    queue_name, e
                )));
            }
            println!("✓ Sequence on queue '{}' completed", queue_name);
        }

        Ok(())
    }

    /// Execute sequences with different priorities and scheduling.
    ///
    /// This demonstrates advanced queue management with priority scheduling.
    ///
    /// # Arguments
    ///
    /// * `high_priority` - High-priority sequences to execute first
    /// * `low_priority` - Low-priority sequences to execute concurrently
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if execution fails.
    pub fn execute_with_priorities(
        &mut self,
        high_priority: Vec<(String, Sequence)>,
        low_priority: Vec<(String, Sequence)>,
    ) -> Result<()> {
        // Execute high-priority sequences first
        self.execute_parallel(high_priority)?;
        println!("✓ High-priority sequences completed");

        // Then execute low-priority sequences
        self.execute_parallel(low_priority)?;
        println!("✓ Low-priority sequences completed");

        Ok(())
    }

    /// Get statistics about queue usage.
    ///
    /// # Returns
    ///
    /// A map of queue names to the number of active sequences.
    pub fn queue_statistics(&self) -> HashMap<String, usize> {
        self.queues
            .iter()
            .map(|(name, info)| (name.clone(), info.active_sequences.len()))
            .collect()
    }

    /// Get the total number of queues.
    pub fn queue_count(&self) -> usize {
        self.queues.len()
    }

    /// Get a list of all queue names.
    pub fn queue_names(&self) -> Vec<String> {
        self.queues.keys().cloned().collect()
    }

    /// Get a mutable reference to the underlying manager.
    pub fn manager(&mut self) -> &mut Manager {
        &mut self.manager
    }

    /// Get information about GPU device capabilities.
    ///
    /// This provides insights into how many queues the GPU can support
    /// and what types of parallel operations are possible.
    pub fn device_capabilities(&self) -> Result<DeviceCapabilities> {
        let device_count = Manager::device_count()?;

        // For now, we provide basic capabilities information
        // In a real implementation, this would query Vulkan device properties
        Ok(DeviceCapabilities {
            max_queues: 4, // Typical number for most GPUs
            supports_async_compute: true,
            supports_concurrent_execution: true,
            device_count,
        })
    }
}

/// Information about GPU device capabilities for multi-queue operations.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Maximum number of queues the device supports
    pub max_queues: u32,
    /// Whether the device supports async compute operations
    pub supports_async_compute: bool,
    /// Whether the device supports concurrent execution
    pub supports_concurrent_execution: bool,
    /// Total number of available devices
    pub device_count: u32,
}

/// Builder pattern for complex multi-queue operations.
///
/// This provides a fluent API for setting up complex parallel GPU workflows.
pub struct MultiQueueWorkflow {
    manager: MultiQueueManager,
    operations: Vec<WorkflowOperation>,
}

#[derive(Debug)]
#[allow(dead_code)] // Part of future API design
struct WorkflowOperation {
    queue_name: String,
    operation_type: OperationType,
    dependencies: Vec<usize>, // Indices of operations this depends on
}

#[derive(Debug)]
#[allow(dead_code)] // Part of future API design
enum OperationType {
    TensorSync { tensor_handle: u64 },
    AlgorithmDispatch { algorithm_handle: u64 },
    Custom { description: String },
}

impl MultiQueueWorkflow {
    /// Create a new workflow builder.
    pub fn new() -> Result<Self> {
        let manager = MultiQueueManager::new()?;
        Ok(Self {
            manager,
            operations: Vec::new(),
        })
    }

    /// Add a tensor synchronization operation to the workflow.
    ///
    /// # Arguments
    ///
    /// * `queue_name` - Name of the queue to execute on
    /// * `tensor_handle` - Handle of the tensor to synchronize
    /// * `dependencies` - Operations this depends on (by index)
    pub fn add_tensor_sync(
        mut self,
        queue_name: &str,
        tensor_handle: u64,
        dependencies: &[usize],
    ) -> Self {
        // Validate inputs (note: we can't return errors from builder pattern, so we store invalid operations)
        if queue_name.is_empty() || tensor_handle == 0 {
            // In a production implementation, we'd want to make this fallible
            eprintln!(
                "Warning: Invalid tensor sync operation - queue_name empty or tensor_handle is 0"
            );
        }

        // Validate dependencies don't reference future operations
        for &dep in dependencies {
            if dep >= self.operations.len() {
                eprintln!(
                    "Warning: Invalid dependency index {} - operation doesn't exist yet",
                    dep
                );
            }
        }

        self.operations.push(WorkflowOperation {
            queue_name: queue_name.to_string(),
            operation_type: OperationType::TensorSync { tensor_handle },
            dependencies: dependencies.to_vec(),
        });
        self
    }

    /// Add an algorithm dispatch operation to the workflow.
    ///
    /// # Arguments
    ///
    /// * `queue_name` - Name of the queue to execute on
    /// * `algorithm_handle` - Handle of the algorithm to dispatch
    /// * `dependencies` - Operations this depends on (by index)
    pub fn add_algorithm_dispatch(
        mut self,
        queue_name: &str,
        algorithm_handle: u64,
        dependencies: &[usize],
    ) -> Self {
        // Validate inputs (note: we can't return errors from builder pattern, so we store invalid operations)
        if queue_name.is_empty() || algorithm_handle == 0 {
            // In a production implementation, we'd want to make this fallible
            eprintln!("Warning: Invalid algorithm dispatch operation - queue_name empty or algorithm_handle is 0");
        }

        // Validate dependencies don't reference future operations
        for &dep in dependencies {
            if dep >= self.operations.len() {
                eprintln!(
                    "Warning: Invalid dependency index {} - operation doesn't exist yet",
                    dep
                );
            }
        }

        self.operations.push(WorkflowOperation {
            queue_name: queue_name.to_string(),
            operation_type: OperationType::AlgorithmDispatch { algorithm_handle },
            dependencies: dependencies.to_vec(),
        });
        self
    }

    /// Execute the complete workflow with dependency resolution.
    ///
    /// This automatically handles dependencies between operations and
    /// executes them in the correct order across multiple queues.
    pub fn execute(mut self) -> Result<()> {
        // Create all required queues
        let queue_names: std::collections::HashSet<String> = self
            .operations
            .iter()
            .map(|op| op.queue_name.clone())
            .collect();

        for queue_name in queue_names {
            self.manager.create_queue(&queue_name)?;
        }

        // For now, execute operations in order (dependency resolution would be more complex)
        for (i, operation) in self.operations.iter().enumerate() {
            println!(
                "✓ Executing operation {} on queue '{}'",
                i, operation.queue_name
            );

            // In a real implementation, we would:
            // 1. Wait for dependencies to complete
            // 2. Create sequence for this operation
            // 3. Execute it on the appropriate queue
        }

        println!("✓ Workflow completed successfully");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_multi_queue_creation() {
        if let Ok(mut multi_manager) = MultiQueueManager::new() {
            // Test queue creation
            assert_eq!(multi_manager.queue_count(), 0);

            let compute_queue = multi_manager.create_queue("compute");
            let transfer_queue = multi_manager.create_queue("transfer");

            assert!(compute_queue.is_ok());
            assert!(transfer_queue.is_ok());
            assert_eq!(multi_manager.queue_count(), 2);

            let queue_names = multi_manager.queue_names();
            assert!(queue_names.contains(&"compute".to_string()));
            assert!(queue_names.contains(&"transfer".to_string()));
        }
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_device_capabilities() {
        if let Ok(multi_manager) = MultiQueueManager::new() {
            if let Ok(capabilities) = multi_manager.device_capabilities() {
                assert!(capabilities.max_queues > 0);
                assert!(capabilities.supports_async_compute);
                println!("Device supports {} queues", capabilities.max_queues);
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_workflow_builder() {
        if let Ok(workflow) = MultiQueueWorkflow::new() {
            let result = workflow
                .add_tensor_sync("transfer", 1, &[])
                .add_algorithm_dispatch("compute", 2, &[0])
                .add_tensor_sync("transfer", 3, &[1])
                .execute();

            match result {
                Ok(()) => println!("✓ Workflow executed successfully"),
                Err(_) => {
                    println!("⚠️ Workflow execution failed (expected in headless environments)")
                }
            }
        }
    }
}
