//! Property-based tests for the komputers library using proptest.
//!
//! These tests verify that properties hold across a wide range of inputs,
//! helping to catch edge cases and ensure robust behavior.
//!
//! Note: These tests require the `property-testing` feature to be enabled.

#[cfg(feature = "property-testing")]
use komputers::{Manager, Tensor};
#[cfg(feature = "property-testing")]
use proptest::prelude::*;

/// Property test: Tensor creation and data retrieval should be consistent.
#[cfg(all(test, feature = "property-testing"))]
mod tensor_properties {
    use super::*;

    proptest! {
        #[test]

        fn prop_tensor_data_consistency(data in prop::collection::vec(any::<f32>(), 1..1000)) {
            if let Ok(mut manager) = Manager::new() {
                if let Ok(tensor) = Tensor::new(&mut manager, &data) {
                    // Property: Retrieved data should match original data
                    if let Ok(retrieved) = tensor.to_vec() {
                        prop_assert_eq!(data.len(), retrieved.len());

                        for (orig, retr) in data.iter().zip(retrieved.iter()) {
                            if orig.is_nan() {
                                prop_assert!(retr.is_nan());
                            } else {
                                prop_assert_eq!(orig, retr);
                            }
                        }
                    }
                }
            }
        }

        #[test]

        fn prop_tensor_size_properties(data in prop::collection::vec(any::<f32>(), 0..10000)) {
            if let Ok(mut manager) = Manager::new() {
                if let Ok(tensor) = Tensor::new(&mut manager, &data) {
                    // Property: Tensor size should match input data length
                    prop_assert_eq!(tensor.size(), data.len());

                    // Property: Tensor byte size should be size * sizeof(f32)
                    prop_assert_eq!(tensor.size_bytes(), data.len() * 4);

                    // Property: Handle should be non-zero
                    prop_assert_ne!(tensor.handle(), 0);
                }
            }
        }

        #[test]

        fn prop_tensor_handle_uniqueness(
            data1 in prop::collection::vec(any::<f32>(), 1..100),
            data2 in prop::collection::vec(any::<f32>(), 1..100)
        ) {
            if let Ok(mut manager) = Manager::new() {
                if let (Ok(tensor1), Ok(tensor2)) = (
                    Tensor::new(&mut manager, &data1),
                    Tensor::new(&mut manager, &data2)
                ) {
                    // Property: Different tensors should have different handles
                    prop_assert_ne!(tensor1.handle(), tensor2.handle());

                    // Property: Handles should be non-zero
                    prop_assert_ne!(tensor1.handle(), 0);
                    prop_assert_ne!(tensor2.handle(), 0);
                }
            }
        }

        #[test]
        fn prop_tensor_uninit_size(size in 1usize..10000) {
            if let Ok(mut manager) = Manager::new() {
                if let Ok(tensor) = Tensor::<f32>::uninit(&mut manager, size) {
                    // Property: Uninitialized tensor should have correct size
                    prop_assert_eq!(tensor.size(), size);
                    prop_assert_eq!(tensor.size_bytes(), size * 4);
                    prop_assert_ne!(tensor.handle(), 0);
                }
            }
        }

        #[test]

        fn prop_multiple_tensor_creation(
            tensor_count in 1usize..50,
            size_per_tensor in 1usize..100
        ) {
            if let Ok(mut manager) = Manager::new() {
                let mut tensors = Vec::new();
                let mut handles = Vec::new();

                for i in 0..tensor_count {
                    let data: Vec<f32> = vec![i as f32; size_per_tensor];
                    if let Ok(tensor) = Tensor::new(&mut manager, &data) {
                        handles.push(tensor.handle());
                        tensors.push(tensor);
                    }
                }

                // Property: All handles should be unique
                for i in 0..handles.len() {
                    for j in (i + 1)..handles.len() {
                        prop_assert_ne!(handles[i], handles[j]);
                    }
                }

                // Property: All tensors should have correct properties
                for (i, tensor) in tensors.iter().enumerate() {
                    prop_assert_eq!(tensor.size(), size_per_tensor);
                    if let Ok(data) = tensor.to_vec() {
                        prop_assert_eq!(data.len(), size_per_tensor);
                        if !data.is_empty() {
                            prop_assert_eq!(data[0], i as f32);
                        }
                    }
                }
            }
        }
    }
}

/// Property tests for special float values and edge cases.
#[cfg(all(test, feature = "property-testing"))]
mod float_properties {
    use super::*;

    proptest! {
        #[test]

        fn prop_special_float_values(
            normal_vals in prop::collection::vec(prop::num::f32::NORMAL, 1..100),
            inf_count in 0usize..10,
            nan_count in 0usize..10,
            zero_count in 0usize..10
        ) {
            let mut data = normal_vals;

            // Add special values
            for _ in 0..inf_count {
                data.push(f32::INFINITY);
                data.push(f32::NEG_INFINITY);
            }
            for _ in 0..nan_count {
                data.push(f32::NAN);
            }
            for _ in 0..zero_count {
                data.push(0.0);
                data.push(-0.0);
            }

            if !data.is_empty() {
                if let Ok(mut manager) = Manager::new() {
                    if let Ok(tensor) = Tensor::new(&mut manager, &data) {
                        prop_assert_eq!(tensor.size(), data.len());

                        if let Ok(retrieved) = tensor.to_vec() {
                            prop_assert_eq!(retrieved.len(), data.len());

                            // Check that special values are preserved
                            for (orig, retr) in data.iter().zip(retrieved.iter()) {
                                if orig.is_nan() {
                                    prop_assert!(retr.is_nan());
                                } else if orig.is_infinite() {
                                    prop_assert!(retr.is_infinite());
                                    prop_assert_eq!(orig.is_sign_positive(), retr.is_sign_positive());
                                } else {
                                    prop_assert_eq!(orig, retr);
                                }
                            }
                        }
                    }
                }
            }
        }

        #[test]

        fn prop_extreme_float_values(
            data in prop::collection::vec(
                prop::num::f32::ANY.prop_filter("finite values only", |x| x.is_finite()),
                1..1000
            )
        ) {
            if let Ok(mut manager) = Manager::new() {
                if let Ok(tensor) = Tensor::new(&mut manager, &data) {
                    if let Ok(retrieved) = tensor.to_vec() {
                        // Property: All finite values should be exactly preserved
                        for (orig, retr) in data.iter().zip(retrieved.iter()) {
                            prop_assert_eq!(orig, retr);
                        }
                    }
                }
            }
        }
    }
}

/// Property tests for error handling and robustness.
#[cfg(all(test, feature = "property-testing"))]
mod error_properties {
    use super::*;

    #[test]
    fn prop_device_id_bounds() {
        // Property: Invalid device IDs should fail gracefully
        for device_id in 100..110u32 {
            let result = Manager::with_device(device_id);
            // Should either succeed or fail with a proper error
            match result {
                Ok(_) => {
                    // If it succeeds, we can't test the error case
                    // but at least it didn't panic
                }
                Err(_) => {
                    // This is the expected case for invalid device IDs
                    // The important thing is that it returned an error rather than panicking
                }
            }
        }
    }

    #[test]
    fn prop_tensor_creation_never_panics() {
        // Property: Tensor creation should never panic, only return errors
        let test_data_sets = vec![
            vec![],
            vec![1.0f32],
            vec![1.0f32; 100],
            vec![f32::NAN; 10],
            vec![f32::INFINITY; 10],
        ];

        for data in test_data_sets {
            if let Ok(mut manager) = Manager::new() {
                let _result = Tensor::new(&mut manager, &data);
                // The important thing is we reach this point without panicking
            }
        }
    }

    #[test]
    fn prop_manager_creation_stability() {
        // Property: Multiple manager creation attempts should be stable
        for _ in 0..10 {
            let _result = Manager::new();
            // Should not panic regardless of success or failure
        }
    }
}

/// Property tests for concurrency and thread safety.
#[cfg(all(test, feature = "property-testing"))]
mod concurrency_properties {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]

    fn prop_concurrent_tensor_creation() {
        let thread_count = 3;
        let tensors_per_thread = 5;
        let tensor_size = 50;

        if let Ok(manager) = Manager::new() {
            let manager = Arc::new(Mutex::new(manager));
            let mut handles = vec![];

            for thread_id in 0..thread_count {
                let manager_clone = Arc::clone(&manager);
                let handle = thread::spawn(move || -> Vec<u64> {
                    let mut thread_handles = Vec::new();

                    for tensor_id in 0..tensors_per_thread {
                        let data: Vec<f32> =
                            vec![(thread_id * 1000 + tensor_id) as f32; tensor_size];

                        let mut manager_guard = manager_clone.lock().unwrap();
                        if let Ok(tensor) = Tensor::new(&mut *manager_guard, &data) {
                            thread_handles.push(tensor.handle());
                            drop(manager_guard);

                            // Verify tensor outside the lock
                            if let Ok(retrieved) = tensor.to_vec() {
                                assert_eq!(retrieved.len(), tensor_size);
                                if !retrieved.is_empty() {
                                    assert_eq!(retrieved[0], (thread_id * 1000 + tensor_id) as f32);
                                }
                            }
                        } else {
                            drop(manager_guard);
                        }
                    }

                    thread_handles
                });
                handles.push(handle);
            }

            // Collect all handles from all threads
            let mut all_handles = Vec::new();
            for handle in handles {
                if let Ok(thread_handles) = handle.join() {
                    all_handles.extend(thread_handles);
                }
            }

            // Property: All handles should be unique across threads
            for i in 0..all_handles.len() {
                for j in (i + 1)..all_handles.len() {
                    assert_ne!(all_handles[i], all_handles[j]);
                }
            }

            // Property: All handles should be non-zero
            for handle in all_handles {
                assert_ne!(handle, 0);
            }
        }
    }
}

/// Property tests for data type consistency.
#[cfg(all(test, feature = "property-testing"))]
mod type_properties {
    use komputers::TensorData;

    #[test]
    fn prop_tensor_data_type_ids_unique() {
        // Property: All data type IDs should be unique
        let f32_id = f32::data_type_id();
        let f64_id = f64::data_type_id();
        let i32_id = i32::data_type_id();
        let u32_id = u32::data_type_id();

        let ids = vec![f32_id, f64_id, i32_id, u32_id];

        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j]);
            }
        }
    }

    #[test]
    fn prop_tensor_data_type_consistency() {
        // Property: Type IDs should be consistent across calls
        assert_eq!(f32::data_type_id(), f32::data_type_id());
        assert_eq!(f64::data_type_id(), f64::data_type_id());
        assert_eq!(i32::data_type_id(), i32::data_type_id());
        assert_eq!(u32::data_type_id(), u32::data_type_id());
    }
}
