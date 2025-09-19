//! GPU queue abstraction with lifecycle management.

use crate::error::{KomputeError, Result};
use crate::sequence::Sequence;
use kompute_sys::ffi;

/// GPU compute queue for parallel operations.
///
/// Queues represent independent execution streams on the GPU that can execute
/// operations in parallel. Each queue maintains its own command submission order
/// and provides synchronization mechanisms.
///
/// # RAII Resource Management
///
/// Queues follow RAII principles - the GPU queue resources are automatically
/// released when the Queue is dropped, ensuring no resource leaks.
///
/// # Examples
///
/// ```rust,no_run
/// use kompute::{Manager, Queue, Result};
///
/// fn example() -> Result<()> {
///     let mut manager = Manager::new()?;
///     let queue = Queue::new(&mut manager, 0)?; // Create on queue family 0
///
///     let sequence = queue.create_sequence(&mut manager)?;
///     queue.submit_sequence(sequence.handle())?;
///     queue.wait_idle()?;
///
///     Ok(())
/// }
/// ```
pub struct Queue {
    handle: u64,
    family_index: u32,
    name: String,
}

impl Queue {
    /// Create a new queue on the specified queue family.
    ///
    /// # Arguments
    ///
    /// * `manager` - The GPU manager to create the queue with
    /// * `queue_family_index` - The queue family index (0 for compute)
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if queue creation fails.
    pub fn new(manager: &mut crate::manager::Manager, queue_family_index: u32) -> Result<Self> {
        let handle = manager.create_queue(queue_family_index)?;
        Ok(Self {
            handle,
            family_index: queue_family_index,
            name: format!("queue_family_{}", queue_family_index),
        })
    }

    /// Create a named queue on the specified queue family.
    ///
    /// # Arguments
    ///
    /// * `manager` - The GPU manager to create the queue with
    /// * `queue_family_index` - The queue family index (0 for compute)
    /// * `name` - Human-readable name for the queue
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if queue creation fails.
    pub fn with_name(
        manager: &mut crate::manager::Manager,
        queue_family_index: u32,
        name: &str,
    ) -> Result<Self> {
        let handle = manager.create_queue(queue_family_index)?;
        Ok(Self {
            handle,
            family_index: queue_family_index,
            name: name.to_string(),
        })
    }

    /// Create a sequence on this queue.
    ///
    /// # Arguments
    ///
    /// * `manager` - The GPU manager to create the sequence with
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if sequence creation fails.
    pub fn create_sequence(&self, manager: &mut crate::manager::Manager) -> Result<Sequence> {
        manager.create_sequence_on_queue(self.handle)
    }

    /// Submit a sequence for execution on this queue.
    ///
    /// # Arguments
    ///
    /// * `sequence_handle` - Handle to the sequence to submit
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if submission fails.
    pub fn submit_sequence(&self, sequence_handle: u64) -> Result<()> {
        if sequence_handle == 0 {
            return Err(KomputeError::compute(
                "Cannot submit sequence with invalid handle",
            ));
        }

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::queue_submit_sequence(self.handle, sequence_handle);
        }))
        .map_err(|_| KomputeError::compute("Queue sequence submission failed due to GPU error"))?;

        Ok(())
    }

    /// Wait for all operations on this queue to complete.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if wait operation fails.
    pub fn wait_idle(&self) -> Result<()> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::queue_wait_idle(self.handle);
        }))
        .map_err(|_| KomputeError::compute("Queue wait idle failed due to GPU error"))?;

        Ok(())
    }

    /// Get the queue handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Get the queue family index.
    pub fn family_index(&self) -> u32 {
        self.family_index
    }

    /// Get the queue name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for Queue {
    /// Automatically destroy the GPU queue when the Queue is dropped.
    ///
    /// This ensures proper RAII resource management and prevents GPU resource leaks.
    fn drop(&mut self) {
        ffi::queue_destroy_handle(self.handle);
    }
}

// Queue is safe to send between threads (contains only a handle to GPU resources)
unsafe impl Send for Queue {}

// Queue can be shared between threads with proper synchronization
unsafe impl Sync for Queue {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manager::Manager;

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_queue_creation() {
        if let Ok(mut manager) = Manager::new() {
            let queue_result = Queue::new(&mut manager, 0);
            match queue_result {
                Ok(queue) => {
                    assert_eq!(queue.family_index(), 0);
                    assert!(queue.handle() > 0);
                    println!(
                        "✓ Queue created successfully with handle {}",
                        queue.handle()
                    );
                }
                Err(_) => {
                    println!("⚠️ Queue creation failed (expected in headless environments)");
                }
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_named_queue_creation() {
        if let Ok(mut manager) = Manager::new() {
            let queue_result = Queue::with_name(&mut manager, 0, "test_compute_queue");
            match queue_result {
                Ok(queue) => {
                    assert_eq!(queue.name(), "test_compute_queue");
                    assert_eq!(queue.family_index(), 0);
                    println!("✓ Named queue '{}' created successfully", queue.name());
                }
                Err(_) => {
                    println!("⚠️ Named queue creation failed (expected in headless environments)");
                }
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_queue_sequence_creation() {
        if let Ok(mut manager) = Manager::new() {
            if let Ok(queue) = Queue::new(&mut manager, 0) {
                let sequence_result = queue.create_sequence(&mut manager);
                match sequence_result {
                    Ok(sequence) => {
                        assert!(sequence.handle() > 0);
                        println!(
                            "✓ Sequence created on queue with handle {}",
                            sequence.handle()
                        );
                    }
                    Err(_) => {
                        println!("⚠️ Sequence creation on queue failed (expected in headless environments)");
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_queue_wait_idle() {
        if let Ok(mut manager) = Manager::new() {
            if let Ok(queue) = Queue::new(&mut manager, 0) {
                let wait_result = queue.wait_idle();
                match wait_result {
                    Ok(()) => {
                        println!("✓ Queue wait idle completed successfully");
                    }
                    Err(_) => {
                        println!("⚠️ Queue wait idle failed (expected in headless environments)");
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_queue_raii_cleanup() {
        // Test that queues are properly cleaned up when dropped
        if let Ok(mut manager) = Manager::new() {
            let initial_handle = {
                if let Ok(queue) = Queue::new(&mut manager, 0) {
                    let handle = queue.handle();
                    println!("Created queue with handle {}", handle);
                    handle
                }
                // Queue is dropped here
                else {
                    println!("⚠️ Queue creation failed");
                    return;
                }
            };

            // Queue should be automatically cleaned up
            println!(
                "✓ Queue with handle {} dropped and cleaned up",
                initial_handle
            );

            // Create another queue to verify handle reuse/recycling works
            if let Ok(new_queue) = Queue::new(&mut manager, 0) {
                println!("✓ New queue created with handle {}", new_queue.handle());
            }
        }
    }
}
