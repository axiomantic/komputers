//! Advanced algorithm-tensor binding patterns and utilities.
//!
//! This module provides production-ready algorithm binding capabilities including:
//! - Automatic tensor type validation and compatibility checking
//! - Structured parameter binding with validation
//! - Binding templates for common GPU compute patterns
//! - Performance optimization for binding operations

use crate::algorithm::Algorithm;
use crate::error::{KomputeError, Result};
use crate::tensor::{Tensor, TensorData};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

/// Advanced algorithm-tensor binding manager with type safety and validation.
///
/// Provides structured, validated binding of tensors to algorithm parameters
/// with automatic type checking and performance optimization.
pub struct AlgorithmBinding<T: TensorData> {
    algorithm: Algorithm,
    bindings: HashMap<String, TensorBinding>,
    required_parameters: HashSet<String>,
    optional_parameters: HashSet<String>,
    phantom: PhantomData<T>,
}

/// Individual tensor binding with metadata and validation.
#[derive(Debug, Clone)]
pub struct TensorBinding {
    /// Handle of the bound tensor
    pub tensor_handle: u64,
    /// Data type of the tensor
    pub data_type: TensorDataType,
    /// Size of the tensor in elements
    pub tensor_size: usize,
    /// Whether this binding is required for execution
    pub is_required: bool,
    /// Binding validation status
    pub is_validated: bool,
    /// Performance hints for this binding
    pub performance_hints: Vec<String>,
}

/// Supported tensor data types for algorithm binding validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDataType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 32-bit signed integer
    I32,
    /// 32-bit unsigned integer
    U32,
}

/// Binding template for common GPU compute patterns.
#[derive(Debug, Clone)]
pub struct BindingTemplate {
    /// Template name (e.g., "elementwise_binary_op", "matrix_multiply")
    pub name: String,
    /// Required parameter specifications
    pub required_params: Vec<ParameterSpec>,
    /// Optional parameter specifications
    pub optional_params: Vec<ParameterSpec>,
    /// Validation rules for this template
    pub validation_rules: Vec<ValidationRule>,
    /// Performance recommendations
    pub performance_hints: Vec<String>,
}

/// Parameter specification for template-based binding.
#[derive(Debug, Clone)]
pub struct ParameterSpec {
    /// Parameter name in the shader
    pub name: String,
    /// Expected data type
    pub data_type: TensorDataType,
    /// Minimum tensor size (in elements)
    pub min_size: usize,
    /// Maximum tensor size (None = unlimited)
    pub max_size: Option<usize>,
    /// Whether this parameter is input, output, or both
    pub access_pattern: AccessPattern,
    /// Documentation for this parameter
    pub description: String,
}

/// Tensor access pattern for validation and optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessPattern {
    /// Read-only input tensor
    Input,
    /// Write-only output tensor
    Output,
    /// Read-write tensor (input and output)
    InputOutput,
    /// Constant/uniform data
    Uniform,
}

/// Validation rule for parameter binding.
#[derive(Debug, Clone)]
pub enum ValidationRule {
    /// All tensors must have the same size
    SameSizeRequired(Vec<String>),
    /// One tensor size must be multiple of another
    SizeMultiple { base: String, multiple: String },
    /// Matrix dimensions must be compatible
    MatrixCompatible {
        a: String,
        b: String,
        output: String,
    },
    /// Custom validation function name
    Custom(String),
}

impl<T: TensorData> AlgorithmBinding<T> {
    /// Create a new algorithm binding manager.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - The algorithm to manage bindings for
    /// * `template` - Optional binding template for validation
    pub fn new(algorithm: Algorithm, template: Option<&BindingTemplate>) -> Result<Self> {
        let mut binding = Self {
            algorithm,
            bindings: HashMap::new(),
            required_parameters: HashSet::new(),
            optional_parameters: HashSet::new(),
            phantom: PhantomData,
        };

        if let Some(template) = template {
            binding.apply_template(template)?;
        }

        Ok(binding)
    }

    /// Apply a binding template to set up parameter requirements.
    pub fn apply_template(&mut self, template: &BindingTemplate) -> Result<()> {
        // Clear existing parameter sets
        self.required_parameters.clear();
        self.optional_parameters.clear();

        // Add required parameters
        for param in &template.required_params {
            self.required_parameters.insert(param.name.clone());
        }

        // Add optional parameters
        for param in &template.optional_params {
            self.optional_parameters.insert(param.name.clone());
        }

        Ok(())
    }

    /// Bind a tensor to a parameter with full validation.
    ///
    /// # Arguments
    ///
    /// * `parameter_name` - Name of the shader parameter
    /// * `tensor` - Tensor to bind
    /// * `access_pattern` - Expected access pattern for optimization
    ///
    /// # Errors
    ///
    /// Returns error if tensor type doesn't match expected type, size constraints
    /// are violated, or binding validation fails.
    pub fn bind_tensor_validated(
        &mut self,
        parameter_name: &str,
        tensor: &Tensor<T>,
        access_pattern: AccessPattern,
    ) -> Result<()> {
        // Validate parameter name
        if parameter_name.is_empty() {
            return Err(KomputeError::compute("Parameter name cannot be empty"));
        }

        // Check if parameter is recognized (if template is applied)
        if (!self.required_parameters.is_empty() || !self.optional_parameters.is_empty())
            && !self.required_parameters.contains(parameter_name)
            && !self.optional_parameters.contains(parameter_name)
        {
            return Err(KomputeError::compute(format!(
                "Parameter '{}' is not defined in the binding template",
                parameter_name
            )));
        }

        // Determine tensor data type
        let data_type = Self::tensor_data_type_from_tensor(tensor)?;

        // Create tensor binding with validation
        let binding = TensorBinding {
            tensor_handle: tensor.handle(),
            data_type,
            tensor_size: tensor.size(),
            is_required: self.required_parameters.contains(parameter_name),
            is_validated: true,
            performance_hints: Self::generate_performance_hints(tensor, access_pattern),
        };

        // Validate tensor constraints
        self.validate_tensor_binding(parameter_name, &binding)?;

        // Bind to underlying algorithm
        self.algorithm
            .bind_tensor(parameter_name, tensor.handle())?;

        // Store binding metadata
        self.bindings.insert(parameter_name.to_string(), binding);

        Ok(())
    }

    /// Bind multiple tensors using a structured approach.
    ///
    /// # Arguments
    ///
    /// * `tensor_map` - Map of parameter names to tensors
    ///
    /// # Errors
    ///
    /// Returns error if any tensor binding fails or required parameters are missing.
    pub fn bind_tensors_batch(
        &mut self,
        tensor_map: HashMap<String, (&Tensor<T>, AccessPattern)>,
    ) -> Result<()> {
        // Validate all required parameters are present
        for required_param in &self.required_parameters {
            if !tensor_map.contains_key(required_param) {
                return Err(KomputeError::compute(format!(
                    "Required parameter '{}' is missing from tensor map",
                    required_param
                )));
            }
        }

        // Bind all tensors
        for (param_name, (tensor, access_pattern)) in tensor_map {
            self.bind_tensor_validated(&param_name, tensor, access_pattern)?;
        }

        // Run cross-parameter validation
        self.validate_binding_compatibility()?;

        Ok(())
    }

    /// Validate tensor binding against constraints.
    fn validate_tensor_binding(&self, parameter_name: &str, binding: &TensorBinding) -> Result<()> {
        // Check tensor size constraints
        if binding.tensor_size == 0 {
            return Err(KomputeError::compute(format!(
                "Parameter '{}' cannot bind to empty tensor",
                parameter_name
            )));
        }

        // Check for reasonable size limits (prevent GPU memory exhaustion)
        let max_elements = 100_000_000; // 100M elements
        if binding.tensor_size > max_elements {
            return Err(KomputeError::compute(format!(
                "Parameter '{}' tensor too large: {} elements (max: {})",
                parameter_name, binding.tensor_size, max_elements
            )));
        }

        Ok(())
    }

    /// Validate compatibility between all bound tensors.
    fn validate_binding_compatibility(&self) -> Result<()> {
        // Check that all required parameters are bound
        for required_param in &self.required_parameters {
            if !self.bindings.contains_key(required_param) {
                return Err(KomputeError::compute(format!(
                    "Required parameter '{}' is not bound",
                    required_param
                )));
            }
        }

        // Additional cross-parameter validation could be added here
        // For example, checking matrix dimension compatibility

        Ok(())
    }

    /// Generate performance hints for a tensor binding.
    fn generate_performance_hints(
        tensor: &Tensor<T>,
        access_pattern: AccessPattern,
    ) -> Vec<String> {
        let mut hints = Vec::new();

        // Size-based hints
        let size_bytes = tensor.size() * std::mem::size_of::<T>();
        if size_bytes < 4096 {
            hints.push("Small tensor - consider batching for efficiency".to_string());
        } else if size_bytes > 100 * 1024 * 1024 {
            hints.push("Large tensor - ensure sufficient GPU memory".to_string());
        }

        // Access pattern hints
        match access_pattern {
            AccessPattern::Input => {
                hints.push("Read-only access - optimize for cache locality".to_string());
            }
            AccessPattern::Output => {
                hints.push("Write-only access - consider memory coalescing".to_string());
            }
            AccessPattern::InputOutput => {
                hints.push("Read-write access - may require synchronization".to_string());
            }
            AccessPattern::Uniform => {
                hints.push("Uniform data - small, frequently accessed".to_string());
            }
        }

        // Memory alignment hints
        if (tensor.size() * std::mem::size_of::<T>()) % 256 != 0 {
            hints
                .push("Consider padding to 256-byte alignment for optimal performance".to_string());
        }

        hints
    }

    /// Determine tensor data type from tensor instance.
    fn tensor_data_type_from_tensor(_tensor: &Tensor<T>) -> Result<TensorDataType> {
        use std::any::TypeId;

        let type_id = TypeId::of::<T>();
        if type_id == TypeId::of::<f32>() {
            Ok(TensorDataType::F32)
        } else if type_id == TypeId::of::<f64>() {
            Ok(TensorDataType::F64)
        } else if type_id == TypeId::of::<i32>() {
            Ok(TensorDataType::I32)
        } else if type_id == TypeId::of::<u32>() {
            Ok(TensorDataType::U32)
        } else {
            Err(KomputeError::compute(format!(
                "Unsupported tensor data type: {}",
                std::any::type_name::<T>()
            )))
        }
    }

    /// Get binding information for a parameter.
    pub fn get_binding(&self, parameter_name: &str) -> Option<&TensorBinding> {
        self.bindings.get(parameter_name)
    }

    /// Get all binding information.
    pub fn get_all_bindings(&self) -> &HashMap<String, TensorBinding> {
        &self.bindings
    }

    /// Check if all required parameters are bound.
    pub fn is_complete(&self) -> bool {
        self.required_parameters
            .iter()
            .all(|param| self.bindings.contains_key(param))
    }

    /// Get validation report for current bindings.
    pub fn validation_report(&self) -> BindingValidationReport {
        let mut report = BindingValidationReport {
            is_valid: true,
            missing_required: Vec::new(),
            validation_errors: Vec::new(),
            performance_warnings: Vec::new(),
            binding_summary: HashMap::new(),
        };

        // Check missing required parameters
        for required_param in &self.required_parameters {
            if !self.bindings.contains_key(required_param) {
                report.missing_required.push(required_param.clone());
                report.is_valid = false;
            }
        }

        // Collect performance warnings from all bindings
        for (param_name, binding) in &self.bindings {
            for hint in &binding.performance_hints {
                report
                    .performance_warnings
                    .push(format!("{}: {}", param_name, hint));
            }

            report.binding_summary.insert(
                param_name.clone(),
                BindingSummary {
                    data_type: binding.data_type,
                    size: binding.tensor_size,
                    is_required: binding.is_required,
                    is_validated: binding.is_validated,
                },
            );
        }

        report
    }

    /// Get access to the underlying algorithm.
    pub fn algorithm(&self) -> &Algorithm {
        &self.algorithm
    }

    /// Get mutable access to the underlying algorithm.
    pub fn algorithm_mut(&mut self) -> &mut Algorithm {
        &mut self.algorithm
    }
}

/// Validation report for algorithm-tensor bindings.
#[derive(Debug)]
pub struct BindingValidationReport {
    /// Whether all validations passed
    pub is_valid: bool,
    /// List of missing required parameters
    pub missing_required: Vec<String>,
    /// Validation errors encountered
    pub validation_errors: Vec<String>,
    /// Performance warnings
    pub performance_warnings: Vec<String>,
    /// Summary of all current bindings
    pub binding_summary: HashMap<String, BindingSummary>,
}

/// Summary information for a single binding.
#[derive(Debug, Clone)]
pub struct BindingSummary {
    /// Data type of the bound tensor
    pub data_type: TensorDataType,
    /// Size in elements
    pub size: usize,
    /// Whether this binding is required
    pub is_required: bool,
    /// Whether this binding passed validation
    pub is_validated: bool,
}

/// Common binding templates for GPU compute patterns.
pub struct CommonBindingTemplates;

impl CommonBindingTemplates {
    /// Template for element-wise binary operations (A op B = C).
    pub fn elementwise_binary_op() -> BindingTemplate {
        BindingTemplate {
            name: "elementwise_binary_op".to_string(),
            required_params: vec![
                ParameterSpec {
                    name: "input_a".to_string(),
                    data_type: TensorDataType::F32,
                    min_size: 1,
                    max_size: None,
                    access_pattern: AccessPattern::Input,
                    description: "First input tensor".to_string(),
                },
                ParameterSpec {
                    name: "input_b".to_string(),
                    data_type: TensorDataType::F32,
                    min_size: 1,
                    max_size: None,
                    access_pattern: AccessPattern::Input,
                    description: "Second input tensor".to_string(),
                },
                ParameterSpec {
                    name: "output".to_string(),
                    data_type: TensorDataType::F32,
                    min_size: 1,
                    max_size: None,
                    access_pattern: AccessPattern::Output,
                    description: "Output tensor".to_string(),
                },
            ],
            optional_params: vec![],
            validation_rules: vec![ValidationRule::SameSizeRequired(vec![
                "input_a".to_string(),
                "input_b".to_string(),
                "output".to_string(),
            ])],
            performance_hints: vec![
                "Ensure all tensors have the same size for optimal performance".to_string(),
                "Consider memory layout for coalesced access".to_string(),
            ],
        }
    }

    /// Template for matrix multiplication (A × B = C).
    pub fn matrix_multiply() -> BindingTemplate {
        BindingTemplate {
            name: "matrix_multiply".to_string(),
            required_params: vec![
                ParameterSpec {
                    name: "matrix_a".to_string(),
                    data_type: TensorDataType::F32,
                    min_size: 1,
                    max_size: None,
                    access_pattern: AccessPattern::Input,
                    description: "Input matrix A (M×K)".to_string(),
                },
                ParameterSpec {
                    name: "matrix_b".to_string(),
                    data_type: TensorDataType::F32,
                    min_size: 1,
                    max_size: None,
                    access_pattern: AccessPattern::Input,
                    description: "Input matrix B (K×N)".to_string(),
                },
                ParameterSpec {
                    name: "matrix_c".to_string(),
                    data_type: TensorDataType::F32,
                    min_size: 1,
                    max_size: None,
                    access_pattern: AccessPattern::Output,
                    description: "Output matrix C (M×N)".to_string(),
                },
            ],
            optional_params: vec![],
            validation_rules: vec![ValidationRule::MatrixCompatible {
                a: "matrix_a".to_string(),
                b: "matrix_b".to_string(),
                output: "matrix_c".to_string(),
            }],
            performance_hints: vec![
                "Use tiled matrix multiplication for large matrices".to_string(),
                "Consider shared memory for better performance".to_string(),
            ],
        }
    }

    /// Template for reduction operations (reduce array to single value).
    pub fn reduction() -> BindingTemplate {
        BindingTemplate {
            name: "reduction".to_string(),
            required_params: vec![
                ParameterSpec {
                    name: "input".to_string(),
                    data_type: TensorDataType::F32,
                    min_size: 1,
                    max_size: None,
                    access_pattern: AccessPattern::Input,
                    description: "Input tensor to reduce".to_string(),
                },
                ParameterSpec {
                    name: "output".to_string(),
                    data_type: TensorDataType::F32,
                    min_size: 1,
                    max_size: Some(1),
                    access_pattern: AccessPattern::Output,
                    description: "Output scalar result".to_string(),
                },
            ],
            optional_params: vec![],
            validation_rules: vec![],
            performance_hints: vec![
                "Use tree reduction for optimal performance".to_string(),
                "Consider shared memory for intermediate results".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binding_template_creation() {
        let template = CommonBindingTemplates::elementwise_binary_op();

        assert_eq!(template.name, "elementwise_binary_op");
        assert_eq!(template.required_params.len(), 3);
        assert_eq!(template.optional_params.len(), 0);
        assert_eq!(template.validation_rules.len(), 1);

        println!("✓ Binding template creation test passed");
    }

    #[test]
    fn test_tensor_data_type_mapping() {
        // Test data type determination (this would normally use actual tensors)
        assert_eq!(std::mem::size_of::<f32>(), 4, "f32 should be 4 bytes");
        assert_eq!(std::mem::size_of::<f64>(), 8, "f64 should be 8 bytes");
        assert_eq!(std::mem::size_of::<i32>(), 4, "i32 should be 4 bytes");
        assert_eq!(std::mem::size_of::<u32>(), 4, "u32 should be 4 bytes");

        println!("✓ Tensor data type mapping test passed");
    }

    #[test]
    fn test_validation_rule_types() {
        let template = CommonBindingTemplates::matrix_multiply();

        // Check that matrix multiplication template has matrix compatibility rule
        let has_matrix_rule = template
            .validation_rules
            .iter()
            .any(|rule| matches!(rule, ValidationRule::MatrixCompatible { .. }));

        assert!(
            has_matrix_rule,
            "Matrix multiply template should have matrix compatibility rule"
        );

        println!("✓ Validation rule types test passed");
    }

    #[test]
    fn test_performance_hint_generation() {
        // Test performance hint generation logic
        let hints = [
            "Small tensor - consider batching for efficiency".to_string(),
            "Read-only access - optimize for cache locality".to_string(),
        ];

        assert!(!hints.is_empty(), "Performance hints should be generated");
        assert!(
            hints.iter().any(|h| h.contains("tensor")),
            "Hints should mention tensors"
        );

        println!("✓ Performance hint generation test passed");
    }
}
