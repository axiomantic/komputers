
#ifndef KOMPUTE_RS_WRAPPER_H
#define KOMPUTE_RS_WRAPPER_H

// Pure C wrapper functions for critical operations - no C++ headers needed
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Opaque types
typedef void kp_Manager;
typedef void kp_Tensor;
typedef void kp_Algorithm;
typedef void kp_Sequence;

// Manager operations
kp_Manager* kp_manager_new(uint32_t device_id);
void kp_manager_destroy(kp_Manager* manager);
int kp_manager_get_device_count(void);

// Tensor operations
kp_Tensor* kp_tensor_new(kp_Manager* manager, const void* data, uint64_t size, uint32_t data_type);
void kp_tensor_destroy(kp_Tensor* tensor);
int kp_tensor_map_to_host(kp_Tensor* tensor);
int kp_tensor_map_to_device(kp_Tensor* tensor);

// Algorithm operations
kp_Algorithm* kp_algorithm_new(kp_Manager* manager, const uint32_t* spirv_data, uint64_t spirv_size);
void kp_algorithm_destroy(kp_Algorithm* algorithm);

// Sequence operations
kp_Sequence* kp_sequence_new(kp_Manager* manager);
void kp_sequence_destroy(kp_Sequence* sequence);
int kp_sequence_record_tensor_sync_device(kp_Sequence* sequence, kp_Tensor* tensor);
int kp_sequence_record_tensor_sync_local(kp_Sequence* sequence, kp_Tensor* tensor);
int kp_sequence_record_algo_dispatch(kp_Sequence* sequence, kp_Algorithm* algorithm);
int kp_sequence_eval(kp_Sequence* sequence);

#ifdef __cplusplus
}
#endif

#endif // KOMPUTE_RS_WRAPPER_H
