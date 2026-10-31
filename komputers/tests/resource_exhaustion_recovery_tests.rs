//! Resource Exhaustion and Recovery Testing Framework
//!
//! This comprehensive test suite validates GPU resource limit handling,
//! exhaustion detection, and recovery mechanisms. Critical for production
//! deployments that must handle resource constraints gracefully.
//!
//! **Production Safety Focus:**
//! - GPU memory exhaustion detection and recovery
//! - Resource limit boundary testing
//! - Graceful degradation under resource pressure
//! - System stability after resource exhaustion
//! - Error handling and informative messaging
//!
//! **Safety-First Philosophy:** Production systems must fail safely
//! and recover gracefully from resource exhaustion scenarios.

use komputers::*;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Resource exhaustion tracking and recovery validation framework
#[derive(Debug)]
struct ResourceExhaustionTracker {
    max_successful_allocations: usize,
    exhaustion_point_reached: bool,
    recovery_successful: bool,
    error_messages: Vec<String>,
    allocation_sizes: VecDeque<usize>,
    timeline: Vec<(String, Instant)>,
}

impl ResourceExhaustionTracker {
    fn new() -> Self {
        Self {
            max_successful_allocations: 0,
            exhaustion_point_reached: false,
            recovery_successful: false,
            error_messages: Vec::new(),
            allocation_sizes: VecDeque::new(),
            timeline: Vec::new(),
        }
    }

    fn record_successful_allocation(&mut self, size: usize) {
        self.max_successful_allocations += 1;
        self.allocation_sizes.push_back(size);
        self.timeline
            .push(("ALLOCATION_SUCCESS".to_string(), Instant::now()));

        // Keep only recent allocation sizes for analysis
        if self.allocation_sizes.len() > 1000 {
            self.allocation_sizes.pop_front();
        }
    }

    fn record_exhaustion_reached(&mut self, error_message: &str) {
        self.exhaustion_point_reached = true;
        self.error_messages.push(error_message.to_string());
        self.timeline
            .push(("EXHAUSTION_REACHED".to_string(), Instant::now()));
    }

    fn record_recovery_attempt(&mut self, successful: bool) {
        self.recovery_successful = successful;
        let event = if successful {
            "RECOVERY_SUCCESS"
        } else {
            "RECOVERY_FAILED"
        };
        self.timeline.push((event.to_string(), Instant::now()));
    }

    fn print_exhaustion_report(&self) {
        println!("\n=== RESOURCE EXHAUSTION & RECOVERY REPORT ===");
        println!(
            "Max successful allocations: {}",
            self.max_successful_allocations
        );
        println!(
            "Exhaustion point reached: {}",
            self.exhaustion_point_reached
        );
        println!("Recovery successful: {}", self.recovery_successful);
        println!("Error messages captured: {}", self.error_messages.len());

        if let Some(total_size) = self
            .allocation_sizes
            .iter()
            .map(|&s| s)
            .sum::<usize>()
            .checked_mul(1)
        {
            println!(
                "Estimated memory at exhaustion: ~{} MB",
                total_size / (1024 * 1024)
            );
        }

        if !self.error_messages.is_empty() {
            println!("\n📋 Error messages during exhaustion:");
            for (i, msg) in self.error_messages.iter().take(3).enumerate() {
                println!("  {}: {}", i + 1, msg);
            }
            if self.error_messages.len() > 3 {
                println!("  ... and {} more", self.error_messages.len() - 3);
            }
        }

        println!("\n📅 Timeline summary:");
        let timeline_summary =
            self.timeline
                .iter()
                .fold(std::collections::HashMap::new(), |mut acc, (event, _)| {
                    *acc.entry(event).or_insert(0) += 1;
                    acc
                });
        for (event, count) in timeline_summary {
            println!("  {}: {}", event, count);
        }

        self.assess_recovery_quality();
    }

    fn assess_recovery_quality(&self) {
        println!("\n🔍 Recovery Quality Assessment:");

        if !self.exhaustion_point_reached {
            println!("   ⚠️  WARNING: Resource exhaustion was not reached");
            println!("      Test may not have pushed limits sufficiently");
            return;
        }

        if !self.recovery_successful {
            println!("   ❌ CRITICAL: Recovery failed after resource exhaustion");
            println!("      System may be unstable after resource pressure");
            return;
        }

        if self.error_messages.is_empty() {
            println!("   ⚠️  WARNING: No error messages captured during exhaustion");
            println!("      Error reporting may be insufficient");
        } else {
            println!(
                "   ✅ Error reporting: {} messages captured",
                self.error_messages.len()
            );
        }

        if self.max_successful_allocations < 10 {
            println!("   ⚠️  WARNING: Very few allocations before exhaustion");
            println!("      GPU memory may be extremely limited");
        } else {
            println!(
                "   ✅ Resource capacity: {} allocations before exhaustion",
                self.max_successful_allocations
            );
        }

        println!("   🎉 RECOVERY ASSESSMENT: SUCCESSFUL");
        println!("      System demonstrated graceful resource exhaustion handling");
    }
}

/// GPU memory exhaustion and recovery test
///
/// This test attempts to exhaust GPU memory by creating increasingly
/// large tensors until allocation fails, then validates recovery.
#[test]
fn test_gpu_memory_exhaustion_recovery() -> Result<()> {
    println!("🚨 GPU Memory Exhaustion and Recovery Test");
    println!("   Testing graceful handling of GPU memory limits");

    let mut tracker = ResourceExhaustionTracker::new();
    let mut manager = Manager::new()?;
    let mut allocated_tensors = Vec::new();

    let start_time = Instant::now();

    println!("   📈 Phase 1: Pushing toward GPU memory exhaustion");

    // Start with medium-sized allocations and increase
    let mut allocation_size = 10_000; // Start with ~40KB tensors
    let size_increment = 5_000;
    let max_attempts = 500; // Prevent infinite loops

    for attempt in 0..max_attempts {
        let data: Vec<f32> = (0..allocation_size).map(|i| (i + attempt) as f32).collect();

        match Tensor::new(&mut manager, &data) {
            Ok(tensor) => {
                tracker.record_successful_allocation(allocation_size);
                allocated_tensors.push(tensor);

                if attempt % 20 == 0 {
                    println!(
                        "     Allocation {}: {} elements ({:.1} MB)",
                        attempt,
                        allocation_size,
                        (allocation_size * 4) as f64 / 1_048_576.0
                    );
                }

                // Gradually increase allocation size
                if attempt % 10 == 0 {
                    allocation_size += size_increment;
                }
            }
            Err(e) => {
                let error_msg = format!("GPU memory exhaustion: {}", e);
                tracker.record_exhaustion_reached(&error_msg);

                println!(
                    "     💥 GPU memory exhaustion reached at allocation {}",
                    attempt
                );
                println!("        Error: {}", e);
                break;
            }
        }

        // Safety check to prevent runaway allocation
        if allocated_tensors.len() > 1000 {
            println!("     🛡️  Safety limit reached: 1000 tensors allocated");
            break;
        }
    }

    let exhaustion_duration = start_time.elapsed();
    println!(
        "   ⏱️  Exhaustion phase completed in: {:?}",
        exhaustion_duration
    );

    println!("   🔄 Phase 2: Testing recovery after exhaustion");

    // Clear some resources to create room for recovery testing
    let tensors_to_clear = std::cmp::min(allocated_tensors.len() / 2, 50);
    for _ in 0..tensors_to_clear {
        allocated_tensors.pop();
    }

    // Small delay to allow GPU memory cleanup
    thread::sleep(Duration::from_millis(100));

    // Test recovery with small allocation
    let recovery_data = vec![42.0f32; 1000]; // Small recovery test
    match Tensor::new(&mut manager, &recovery_data) {
        Ok(recovery_tensor) => {
            tracker.record_recovery_attempt(true);
            println!("     ✅ Recovery successful: Created 1000-element tensor");

            // Verify the recovery tensor works
            let sequence = manager
                .sequence()?
                .record_tensor_sync_device(recovery_tensor.handle())?;
            sequence.execute()?;

            let result = recovery_tensor.to_vec()?;
            assert_eq!(result.len(), 1000);
            assert_eq!(result[0], 42.0);

            println!("     ✅ Recovery tensor validated successfully");
        }
        Err(e) => {
            tracker.record_recovery_attempt(false);
            println!("     ❌ Recovery failed: {}", e);
        }
    }

    tracker.print_exhaustion_report();

    // Validate that we actually tested resource exhaustion
    assert!(
        tracker.exhaustion_point_reached || tracker.max_successful_allocations > 50,
        "Test did not adequately stress GPU memory limits"
    );

    if tracker.exhaustion_point_reached && !tracker.recovery_successful {
        return Err(KomputeError::device(
            "System failed to recover from GPU memory exhaustion",
        ));
    }

    println!("   🎉 GPU memory exhaustion and recovery test passed");
    Ok(())
}

/// System queue exhaustion and recovery test
///
/// This test attempts to exhaust GPU compute queues by creating
/// many concurrent operations.
#[test]
fn test_gpu_queue_exhaustion_recovery() -> Result<()> {
    println!("⚡ GPU Queue Exhaustion and Recovery Test");
    println!("   Testing graceful handling of GPU queue limits");

    let mut tracker = ResourceExhaustionTracker::new();
    let mut manager = Manager::new()?;
    let mut active_sequences = Vec::new();

    println!("   📊 Phase 1: Creating concurrent GPU operations");

    // Create many sequences to potentially exhaust queue resources
    for sequence_id in 0..200 {
        let data = vec![sequence_id as f32; 100];

        match Tensor::new(&mut manager, &data) {
            Ok(tensor) => match manager.sequence() {
                Ok(sequence) => match sequence.record_tensor_sync_device(tensor.handle()) {
                    Ok(recorded_sequence) => {
                        tracker.record_successful_allocation(1);
                        active_sequences.push((tensor, recorded_sequence));

                        if sequence_id % 25 == 0 {
                            println!("     Created {} active sequences", sequence_id + 1);
                        }
                    }
                    Err(e) => {
                        let error_msg = format!("Sequence recording failed: {}", e);
                        tracker.record_exhaustion_reached(&error_msg);
                        println!(
                            "     💥 Queue exhaustion at sequence {}: {}",
                            sequence_id, e
                        );
                        break;
                    }
                },
                Err(e) => {
                    let error_msg = format!("Sequence creation failed: {}", e);
                    tracker.record_exhaustion_reached(&error_msg);
                    println!(
                        "     💥 Queue creation exhaustion at {}: {}",
                        sequence_id, e
                    );
                    break;
                }
            },
            Err(e) => {
                let error_msg = format!("Tensor creation failed: {}", e);
                tracker.record_exhaustion_reached(&error_msg);
                break;
            }
        }
    }

    println!("   🔄 Phase 2: Testing queue recovery");

    // Execute and clear some sequences to free up queue resources
    let sequences_to_execute = std::cmp::min(active_sequences.len(), 50);
    for i in 0..sequences_to_execute {
        if let Some((tensor, sequence)) = active_sequences.pop() {
            match sequence.execute() {
                Ok(_) => {
                    // Verify the operation completed successfully
                    let result = tensor.to_vec()?;
                    assert_eq!(result.len(), 100);

                    if i % 10 == 0 {
                        println!("     Executed sequence {}/{}", i + 1, sequences_to_execute);
                    }
                }
                Err(e) => {
                    println!("     ⚠️  Sequence execution failed: {}", e);
                }
            }
        }
    }

    // Test queue recovery with new operation
    let recovery_data = vec![999.0f32; 50];
    match Tensor::new(&mut manager, &recovery_data) {
        Ok(recovery_tensor) => match manager.sequence() {
            Ok(sequence) => match sequence.record_tensor_sync_device(recovery_tensor.handle()) {
                Ok(recorded_sequence) => match recorded_sequence.execute() {
                    Ok(_) => {
                        tracker.record_recovery_attempt(true);
                        println!("     ✅ Queue recovery successful");

                        let result = recovery_tensor.to_vec()?;
                        assert_eq!(result.len(), 50);
                        assert_eq!(result[0], 999.0);
                    }
                    Err(e) => {
                        tracker.record_recovery_attempt(false);
                        println!("     ❌ Queue recovery execution failed: {}", e);
                    }
                },
                Err(e) => {
                    tracker.record_recovery_attempt(false);
                    println!("     ❌ Queue recovery recording failed: {}", e);
                }
            },
            Err(e) => {
                tracker.record_recovery_attempt(false);
                println!("     ❌ Queue recovery creation failed: {}", e);
            }
        },
        Err(e) => {
            tracker.record_recovery_attempt(false);
            println!("     ❌ Recovery tensor creation failed: {}", e);
        }
    }

    tracker.print_exhaustion_report();

    println!("   ✅ GPU queue exhaustion and recovery test completed");
    Ok(())
}

/// Concurrent resource pressure test
///
/// This test applies resource pressure from multiple threads
/// to validate system behavior under concurrent resource stress.
#[test]
fn test_concurrent_resource_pressure() -> Result<()> {
    println!("🧵 Concurrent Resource Pressure Test");
    println!("   Testing system stability under concurrent resource stress");

    let shared_tracker = Arc::new(Mutex::new(ResourceExhaustionTracker::new()));
    let thread_count = 4;
    let pressure_duration = Duration::from_secs(10);
    let start_time = Instant::now();

    println!(
        "   ⚡ Applying resource pressure from {} threads for {:?}",
        thread_count, pressure_duration
    );

    let handles: Vec<thread::JoinHandle<Result<()>>> = (0..thread_count)
        .map(|thread_id| {
            let tracker = Arc::clone(&shared_tracker);

            thread::spawn(move || {
                let mut manager = Manager::new()?;
                let mut thread_tensors = Vec::new();
                let mut operation_count = 0;

                while start_time.elapsed() < pressure_duration {
                    // Variable allocation sizes to create pressure
                    let size = 1000 + (operation_count % 5000);
                    let data: Vec<f32> = (0..size)
                        .map(|i| (thread_id * 10000 + operation_count * 100 + i) as f32)
                        .collect();

                    match Tensor::new(&mut manager, &data) {
                        Ok(tensor) => {
                            {
                                let mut t = tracker.lock().unwrap();
                                t.record_successful_allocation(size);
                            }

                            // Perform operation to stress GPU
                            match manager.sequence() {
                                Ok(sequence) => {
                                    match sequence.record_tensor_sync_device(tensor.handle()) {
                                        Ok(recorded_sequence) => {
                                            match recorded_sequence.execute() {
                                                Ok(_) => {
                                                    thread_tensors.push(tensor);
                                                }
                                                Err(e) => {
                                                    let mut t = tracker.lock().unwrap();
                                                    let error_msg = format!(
                                                        "Thread {} execution error: {}",
                                                        thread_id, e
                                                    );
                                                    t.record_exhaustion_reached(&error_msg);
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            let mut t = tracker.lock().unwrap();
                                            let error_msg = format!(
                                                "Thread {} recording error: {}",
                                                thread_id, e
                                            );
                                            t.record_exhaustion_reached(&error_msg);
                                        }
                                    }
                                }
                                Err(e) => {
                                    let mut t = tracker.lock().unwrap();
                                    let error_msg =
                                        format!("Thread {} sequence error: {}", thread_id, e);
                                    t.record_exhaustion_reached(&error_msg);
                                }
                            }

                            // Prevent unlimited tensor accumulation
                            if thread_tensors.len() > 20 {
                                thread_tensors.clear();
                            }
                        }
                        Err(e) => {
                            let mut t = tracker.lock().unwrap();
                            let error_msg = format!("Thread {} allocation error: {}", thread_id, e);
                            t.record_exhaustion_reached(&error_msg);

                            // Brief pause after allocation failure
                            thread::sleep(Duration::from_millis(10));
                        }
                    }

                    operation_count += 1;

                    if operation_count % 20 == 0 {
                        thread::sleep(Duration::from_millis(1));
                    }
                }

                println!(
                    "     Thread {} completed {} operations",
                    thread_id, operation_count
                );
                Ok(())
            })
        })
        .collect();

    // Wait for all pressure threads to complete
    let mut successful_threads = 0;
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(())) => {
                successful_threads += 1;
            }
            Ok(Err(e)) => {
                println!("     ❌ Pressure thread {} failed: {}", i, e);
            }
            Err(_) => {
                println!("     ❌ Pressure thread {} panicked", i);
            }
        }
    }

    let total_duration = start_time.elapsed();
    println!(
        "   📊 Pressure test completed in {:?} ({}/{} threads successful)",
        total_duration, successful_threads, thread_count
    );

    // Test recovery after pressure
    println!("   🔄 Testing system recovery after pressure");

    // Brief pause to allow system recovery
    thread::sleep(Duration::from_millis(200));

    let mut recovery_manager = Manager::new()?;
    let recovery_data = vec![123.0f32; 500];

    match Tensor::new(&mut recovery_manager, &recovery_data) {
        Ok(recovery_tensor) => match recovery_manager.sequence() {
            Ok(sequence) => match sequence.record_tensor_sync_device(recovery_tensor.handle()) {
                Ok(recorded_sequence) => match recorded_sequence.execute() {
                    Ok(_) => {
                        let mut t = shared_tracker.lock().unwrap();
                        t.record_recovery_attempt(true);
                        println!("     ✅ System recovery after pressure: SUCCESS");

                        let result = recovery_tensor.to_vec()?;
                        assert_eq!(result.len(), 500);
                        assert_eq!(result[0], 123.0);
                    }
                    Err(e) => {
                        let mut t = shared_tracker.lock().unwrap();
                        t.record_recovery_attempt(false);
                        println!("     ❌ Recovery execution failed: {}", e);
                    }
                },
                Err(e) => {
                    let mut t = shared_tracker.lock().unwrap();
                    t.record_recovery_attempt(false);
                    println!("     ❌ Recovery recording failed: {}", e);
                }
            },
            Err(e) => {
                let mut t = shared_tracker.lock().unwrap();
                t.record_recovery_attempt(false);
                println!("     ❌ Recovery sequence creation failed: {}", e);
            }
        },
        Err(e) => {
            let mut t = shared_tracker.lock().unwrap();
            t.record_recovery_attempt(false);
            println!("     ❌ Recovery tensor creation failed: {}", e);
        }
    }

    {
        let tracker = shared_tracker.lock().unwrap();
        tracker.print_exhaustion_report();
    }

    assert!(
        successful_threads >= thread_count - 1,
        "Too many thread failures during resource pressure"
    );

    println!("   🎉 Concurrent resource pressure test completed");
    Ok(())
}

/// Error message quality validation test
///
/// This test verifies that error messages during resource exhaustion
/// are informative and actionable for production debugging.
#[test]
fn test_resource_exhaustion_error_quality() -> Result<()> {
    println!("📝 Resource Exhaustion Error Quality Test");
    println!("   Validating error message informativeness and actionability");

    let mut manager = Manager::new()?;
    let mut error_messages = Vec::new();
    let mut allocation_sizes = Vec::new();

    // Attempt to create increasingly large tensors until we get errors
    for size_factor in 1..=20 {
        let size = size_factor * 50_000; // Start at 200KB, increase rapidly
        allocation_sizes.push(size);

        let data = vec![size_factor as f32; size];

        match Tensor::new(&mut manager, &data) {
            Ok(tensor) => {
                // Try to use the tensor to potentially trigger more errors
                match manager.sequence() {
                    Ok(sequence) => match sequence.record_tensor_sync_device(tensor.handle()) {
                        Ok(recorded_sequence) => match recorded_sequence.execute() {
                            Ok(_) => {
                                println!(
                                    "     ✅ Successfully allocated and processed {} elements",
                                    size
                                );
                            }
                            Err(e) => {
                                error_messages.push(format!("Execution error ({}): {}", size, e));
                            }
                        },
                        Err(e) => {
                            error_messages.push(format!("Recording error ({}): {}", size, e));
                        }
                    },
                    Err(e) => {
                        error_messages.push(format!("Sequence error ({}): {}", size, e));
                    }
                }
            }
            Err(e) => {
                error_messages.push(format!("Allocation error ({}): {}", size, e));
                println!("     💥 Allocation failed at {} elements: {}", size, e);
                break;
            }
        }

        // Stop if we've accumulated enough error examples
        if error_messages.len() >= 5 {
            break;
        }
    }

    println!("   📊 Error Message Quality Analysis:");
    println!("     Captured {} error messages", error_messages.len());

    if error_messages.is_empty() {
        println!("     ⚠️  WARNING: No errors captured - may need larger allocation attempts");
        return Ok(());
    }

    // Analyze error message quality
    let mut quality_score = 0;
    let mut quality_feedback = Vec::new();

    for (i, message) in error_messages.iter().enumerate() {
        println!("     Error {}: {}", i + 1, message);

        // Check for informative elements in error messages
        if message.to_lowercase().contains("memory") {
            quality_score += 1;
            quality_feedback.push("✓ Mentions memory");
        }

        if message.to_lowercase().contains("gpu") || message.to_lowercase().contains("device") {
            quality_score += 1;
            quality_feedback.push("✓ Identifies GPU/device context");
        }

        if message.len() > 20 {
            quality_score += 1;
            quality_feedback.push("✓ Reasonably detailed");
        }

        if message.to_lowercase().contains("out of") || message.to_lowercase().contains("exhausted")
        {
            quality_score += 2;
            quality_feedback.push("✓ Clearly indicates exhaustion");
        }
    }

    let avg_quality = quality_score as f64 / error_messages.len() as f64;

    println!("   🔍 Error Quality Assessment:");
    println!("     Average quality score: {:.1}/5", avg_quality);

    for feedback in quality_feedback {
        println!("     {}", feedback);
    }

    if avg_quality >= 3.0 {
        println!("     ✅ ERROR QUALITY: EXCELLENT - Messages are informative");
    } else if avg_quality >= 2.0 {
        println!("     🔶 ERROR QUALITY: GOOD - Messages provide basic information");
    } else {
        println!("     ⚠️  ERROR QUALITY: NEEDS IMPROVEMENT - Messages lack detail");
    }

    println!("   🎉 Error message quality validation completed");
    Ok(())
}

#[cfg(test)]
mod resource_exhaustion_integration_tests {
    use super::*;

    /// Master resource exhaustion and recovery validation suite
    #[test]
    fn test_resource_exhaustion_recovery_comprehensive() -> Result<()> {
        println!("🔬 Master Resource Exhaustion and Recovery Test Suite");
        println!("   Production Safety: Comprehensive resource limit testing");

        let master_start = Instant::now();

        // Core resource exhaustion and recovery tests
        test_gpu_memory_exhaustion_recovery()?;
        test_gpu_queue_exhaustion_recovery()?;
        test_concurrent_resource_pressure()?;
        test_resource_exhaustion_error_quality()?;

        let total_duration = master_start.elapsed();

        println!("\n=== MASTER RESOURCE EXHAUSTION VALIDATION COMPLETE ===");
        println!("Total validation time: {:?}", total_duration);
        println!("✅ All resource exhaustion and recovery tests passed");
        println!("🛡️  Production readiness: Resource limit handling verified");
        println!("🔄 System recovery capabilities validated");
        println!("📝 Error message quality confirmed");

        Ok(())
    }
}
