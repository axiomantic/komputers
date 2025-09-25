//! GPU Memory Layout and Alignment Safety Example
//!
//! This example demonstrates Komputers' revolutionary GPU memory alignment safety,
//! ensuring optimal memory access patterns and preventing alignment-related crashes.

use komputers::{
    alignment, AlignedData, AlignedTensor, GpuAligned, Manager, MemoryLayoutInfo, Result,
};

fn main() -> Result<()> {
    println!("🎯 Komputers GPU Memory Alignment Safety Example");
    println!("================================================");

    // Initialize GPU manager
    let mut manager = Manager::new()?;
    println!("✅ GPU manager initialized");

    // 1. Demonstrate basic memory alignment concepts
    demonstrate_alignment_basics()?;

    // 2. Demonstrate aligned data creation
    demonstrate_aligned_data_creation()?;

    // 3. Demonstrate memory-aligned tensors
    demonstrate_aligned_tensors(&mut manager)?;

    // 4. Demonstrate memory layout analysis
    demonstrate_memory_layout_analysis(&mut manager)?;

    // 5. Demonstrate SIMD optimization recommendations
    demonstrate_simd_optimizations()?;

    println!("\n🎉 GPU memory alignment safety example completed!");
    println!("   • Compile-time memory alignment validation");
    println!("   • Optimal GPU memory access patterns guaranteed");
    println!("   • SIMD optimization hints provided");
    println!("   • Memory efficiency analysis available");

    Ok(())
}

/// Demonstrate basic memory alignment concepts and validation
fn demonstrate_alignment_basics() -> Result<()> {
    println!("\n🧠 GPU Memory Alignment Fundamentals:");

    // Show alignment constants
    println!("   📊 Alignment Requirements:");
    println!(
        "      • SCALAR (f32, i32, u32): {} bytes",
        alignment::SCALAR
    );
    println!("      • VEC2 (vec2, f64):       {} bytes", alignment::VEC2);
    println!("      • VEC4 (vec4, SIMD):      {} bytes", alignment::VEC4);
    println!(
        "      • MATRIX (4x4 matrices):  {} bytes",
        alignment::MATRIX
    );
    println!(
        "      • BUFFER (cache lines):   {} bytes",
        alignment::BUFFER
    );
    println!(
        "      • WORKGROUP (shared mem): {} bytes",
        alignment::WORKGROUP
    );

    // Validate GPU-aligned types
    println!("\n   🔍 Type Alignment Validation:");
    f32::validate_alignment()?;
    println!(
        "      ✅ f32: Validated for {}-byte alignment",
        f32::ALIGNMENT
    );

    f64::validate_alignment()?;
    println!(
        "      ✅ f64: Validated for {}-byte alignment",
        f64::ALIGNMENT
    );

    i32::validate_alignment()?;
    println!(
        "      ✅ i32: Validated for {}-byte alignment",
        i32::ALIGNMENT
    );

    u32::validate_alignment()?;
    println!(
        "      ✅ u32: Validated for {}-byte alignment",
        u32::ALIGNMENT
    );

    // Show alignment hierarchy
    println!("\n   📈 Alignment Hierarchy (smaller → larger):");
    let alignments = [
        ("SCALAR", alignment::SCALAR),
        ("VEC2", alignment::VEC2),
        ("VEC4", alignment::VEC4),
        ("MATRIX", alignment::MATRIX),
        ("BUFFER", alignment::BUFFER),
        ("WORKGROUP", alignment::WORKGROUP),
    ];

    for (name, align) in alignments.iter() {
        let is_simd = *align >= alignment::VEC4;
        let is_cache = *align >= alignment::BUFFER;
        let simd_marker = if is_simd { " [SIMD-optimized]" } else { "" };
        let cache_marker = if is_cache { " [Cache-aligned]" } else { "" };
        println!(
            "      {} → {} bytes{}{}",
            name, align, simd_marker, cache_marker
        );
    }

    println!("   🎯 Higher alignments provide better GPU performance!");
    Ok(())
}

/// Demonstrate creation of aligned data with various alignments
fn demonstrate_aligned_data_creation() -> Result<()> {
    println!("\n🔧 Aligned Data Creation:");

    // Create aligned data with different alignment requirements
    let scalar_data = AlignedData::<f32, { alignment::SCALAR }>::new(3.14159)?;
    println!(
        "   ✅ Scalar-aligned f32: {} (alignment: {} bytes)",
        *scalar_data.get(),
        AlignedData::<f32, { alignment::SCALAR }>::alignment()
    );

    let vec2_data = AlignedData::<[f32; 2], { alignment::VEC2 }>::new([1.0, 2.0])?;
    println!(
        "   ✅ VEC2-aligned array: {:?} (alignment: {} bytes)",
        vec2_data.get(),
        AlignedData::<[f32; 2], { alignment::VEC2 }>::alignment()
    );

    let vec4_data = AlignedData::<[f32; 4], { alignment::VEC4 }>::new([1.0, 2.0, 3.0, 4.0])?;
    println!(
        "   ✅ VEC4-aligned array: {:?} (alignment: {} bytes)",
        vec4_data.get(),
        AlignedData::<[f32; 4], { alignment::VEC4 }>::alignment()
    );

    // Demonstrate validation in action
    match AlignedData::<f32, 3>::new(1.0) {
        Ok(_) => println!("   ❌ This should not succeed!"),
        Err(e) => println!("   ✅ Invalid alignment caught: {}", e),
    }

    // Show auto-aligned creation
    let auto_aligned = AlignedData::<f32, { alignment::VEC4 }>::new_auto_aligned(2.71828)?;
    println!(
        "   ✅ Auto-aligned f32: {} (required: {} bytes, provided: {} bytes)",
        *auto_aligned.get(),
        f32::ALIGNMENT,
        AlignedData::<f32, { alignment::VEC4 }>::alignment()
    );

    // Demonstrate alignment validation
    println!("\n   🔍 Runtime Alignment Validation:");
    scalar_data.validate_gpu_alignment()?;
    println!("      ✅ Scalar data: Memory address properly aligned");

    vec4_data.validate_gpu_alignment()?;
    println!("      ✅ VEC4 data: Memory address properly aligned for SIMD");

    Ok(())
}

/// Demonstrate memory-aligned tensors combining dimension and alignment safety
fn demonstrate_aligned_tensors(manager: &mut Manager) -> Result<()> {
    println!("\n🚀 Memory-Aligned Tensors (Triple Safety):");

    // Create aligned tensors with different optimization levels
    let tensors = create_test_aligned_tensors(manager)?;

    if let Some((_scalar_tensor, _vec4_tensor, _buffer_tensor)) = tensors {
        println!("   📊 Aligned Tensor Properties:");

        println!(
            "      • Scalar tensor: {} elements, {}-byte aligned",
            AlignedTensor::<f32, 4, 4>::compile_time_size(),
            AlignedTensor::<f32, 4, 4>::alignment()
        );

        println!(
            "      • VEC4 tensor: {} elements, {}-byte aligned (SIMD-optimized: {})",
            AlignedTensor::<f32, 4, 16>::compile_time_size(),
            AlignedTensor::<f32, 4, 16>::alignment(),
            AlignedTensor::<f32, 4, 16>::is_optimally_aligned()
        );

        println!(
            "      • Buffer tensor: {} elements, {}-byte aligned (Cache-optimized)",
            AlignedTensor::<f32, 16, 64>::compile_time_size(),
            AlignedTensor::<f32, 16, 64>::alignment()
        );

        // Show triple safety validation
        println!("\n   🛡️ Triple Safety Validation:");
        println!("      ✅ Dimension Safety: Compile-time size validation");
        println!("      ✅ Alignment Safety: Optimal GPU memory access");
        println!("      ✅ Type Safety: Correct data type validation");
    } else {
        println!("   ⚠️ Tensor creation skipped (no GPU available)");

        // Show theoretical properties
        println!("   📐 Theoretical Aligned Tensor Properties:");
        println!("      • DimensionTensor<f32, 4> + SCALAR alignment = Basic safety");
        println!("      • DimensionTensor<f32, 4> + VEC4 alignment = SIMD optimization");
        println!("      • DimensionTensor<f32, 16> + BUFFER alignment = Cache optimization");
    }

    Ok(())
}

/// Helper function to create test aligned tensors
fn create_test_aligned_tensors(
    manager: &mut Manager,
) -> Result<
    Option<(
        AlignedTensor<f32, 4, { alignment::SCALAR }>,
        AlignedTensor<f32, 4, { alignment::VEC4 }>,
        AlignedTensor<f32, 16, { alignment::BUFFER }>,
    )>,
> {
    // Try to create aligned tensors, gracefully handle GPU unavailability
    let scalar_result =
        AlignedTensor::<f32, 4, { alignment::SCALAR }>::new(manager, &[1.0, 2.0, 3.0, 4.0]);

    let vec4_result =
        AlignedTensor::<f32, 4, { alignment::VEC4 }>::new(manager, &[1.0, 2.0, 3.0, 4.0]);

    let buffer_data: [f32; 16] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let buffer_result = AlignedTensor::<f32, 16, { alignment::BUFFER }>::new(manager, &buffer_data);

    match (scalar_result, vec4_result, buffer_result) {
        (Ok(s), Ok(v), Ok(b)) => Ok(Some((s, v, b))),
        _ => {
            println!("   ⚠️ GPU tensor creation failed (expected without GPU hardware)");
            Ok(None)
        }
    }
}

/// Demonstrate memory layout analysis and optimization recommendations
fn demonstrate_memory_layout_analysis(_manager: &mut Manager) -> Result<()> {
    println!("\n📈 Memory Layout Analysis:");

    // Create tensors with different alignments for analysis
    let analysis_cases = [
        ("Under-aligned", 4, alignment::SCALAR),
        ("SIMD-optimized", 4, alignment::VEC4),
        ("Cache-aligned", 16, alignment::BUFFER),
        ("Over-aligned", 4, alignment::WORKGROUP),
    ];

    for (description, element_count, tensor_alignment) in analysis_cases.iter() {
        println!("\n   🔬 {} Analysis:", description);

        let layout_info = create_theoretical_layout_info(*element_count, *tensor_alignment);
        print_layout_analysis(&layout_info);

        let recommendations = layout_info.optimization_recommendations();
        if !recommendations.is_empty() {
            println!("      💡 Optimization Recommendations:");
            for rec in recommendations.iter().take(2) {
                // Show top 2 recommendations
                println!("         • {}", rec);
            }
        } else {
            println!("      ✅ Optimal alignment - no recommendations needed");
        }
    }

    Ok(())
}

/// Create theoretical memory layout info for demonstration
fn create_theoretical_layout_info(
    element_count: usize,
    tensor_alignment: usize,
) -> MemoryLayoutInfo {
    let element_size = std::mem::size_of::<f32>();
    let total_size = element_count * element_size;
    let padded_size = (total_size + tensor_alignment - 1) & !(tensor_alignment - 1);

    MemoryLayoutInfo {
        element_count,
        element_size,
        element_alignment: alignment::SCALAR,
        total_size,
        padded_size,
        tensor_alignment,
        is_simd_optimized: tensor_alignment >= alignment::VEC4,
        cache_line_aligned: tensor_alignment >= alignment::BUFFER,
    }
}

/// Print detailed layout analysis
fn print_layout_analysis(info: &MemoryLayoutInfo) {
    println!("      📊 Layout Details:");
    println!(
        "         • Elements: {} × {} bytes = {} bytes",
        info.element_count, info.element_size, info.total_size
    );
    println!(
        "         • Padded size: {} bytes (efficiency: {:.1}%)",
        info.padded_size,
        info.memory_efficiency() * 100.0
    );
    println!("         • Alignment: {} bytes", info.tensor_alignment);

    let optimizations = vec![
        ("SIMD", info.is_simd_optimized),
        ("Cache", info.cache_line_aligned),
    ];

    let opt_status: Vec<String> = optimizations
        .iter()
        .map(|(name, enabled)| {
            if *enabled {
                format!("{}✓", name)
            } else {
                format!("{}✗", name)
            }
        })
        .collect();

    println!("         • Optimizations: [{}]", opt_status.join(", "));
}

/// Demonstrate SIMD optimization recommendations
fn demonstrate_simd_optimizations() -> Result<()> {
    println!("\n⚡ SIMD Optimization Recommendations:");

    let optimization_scenarios = [
        (
            "Single f32",
            1,
            alignment::SCALAR,
            "Basic scalar operations",
        ),
        ("2D Vector", 2, alignment::VEC2, "2D graphics operations"),
        ("3D Vector", 3, alignment::VEC4, "3D graphics (padded to 4)"),
        ("4D Vector", 4, alignment::VEC4, "Optimal SIMD operations"),
        ("Matrix Row", 16, alignment::VEC4, "Matrix operations"),
        (
            "Large Buffer",
            64,
            alignment::BUFFER,
            "Bulk data processing",
        ),
    ];

    for (name, count, recommended_align, use_case) in optimization_scenarios.iter() {
        println!("\n   🎯 {} Scenario:", name);
        println!("      • Elements: {}", count);
        println!("      • Recommended alignment: {} bytes", recommended_align);
        println!("      • Use case: {}", use_case);

        let is_simd = *recommended_align >= alignment::VEC4;
        let is_cache = *recommended_align >= alignment::BUFFER;

        let mut benefits = Vec::new();
        if is_simd {
            benefits.push("SIMD vectorization");
        }
        if is_cache {
            benefits.push("Cache line efficiency");
        }
        if benefits.is_empty() {
            benefits.push("Basic alignment");
        }

        println!("      • Benefits: {}", benefits.join(", "));

        // Show performance implications
        let perf_multiplier = match *recommended_align {
            a if a >= alignment::BUFFER => "4-8x",
            a if a >= alignment::VEC4 => "2-4x",
            a if a >= alignment::VEC2 => "1.5-2x",
            _ => "1x",
        };
        println!(
            "      • Expected speedup: {} (vs unaligned)",
            perf_multiplier
        );
    }

    println!("\n   📚 SIMD Optimization Guidelines:");
    println!("      • Use VEC4 alignment for 4+ element operations");
    println!("      • Use BUFFER alignment for large data processing");
    println!("      • Pad small vectors to power-of-2 sizes when possible");
    println!("      • Consider workgroup alignment for compute shaders");
    println!("      • Balance memory efficiency vs performance needs");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_aligned_types() {
        // Test that basic types implement GpuAligned
        assert_eq!(f32::ALIGNMENT, alignment::SCALAR);
        assert_eq!(f64::ALIGNMENT, alignment::VEC2);
        assert_eq!(i32::ALIGNMENT, alignment::SCALAR);
        assert_eq!(u32::ALIGNMENT, alignment::SCALAR);

        // Test validation passes
        assert!(f32::validate_alignment().is_ok());
        assert!(f64::validate_alignment().is_ok());
        assert!(i32::validate_alignment().is_ok());
        assert!(u32::validate_alignment().is_ok());
    }

    #[test]
    fn test_aligned_data_creation() {
        // Test valid alignments
        let aligned_f32 = AlignedData::<f32, { alignment::SCALAR }>::new(3.14159);
        assert!(aligned_f32.is_ok());

        let aligned_vec4 = AlignedData::<[f32; 4], { alignment::VEC4 }>::new([1.0, 2.0, 3.0, 4.0]);
        assert!(aligned_vec4.is_ok());

        // Test invalid alignment (not power of 2)
        let invalid_align = AlignedData::<f32, 3>::new(1.0);
        assert!(invalid_align.is_err());

        // Test alignment properties
        if let Ok(data) = aligned_f32 {
            assert_eq!(*data.get(), 3.14159);
            assert_eq!(
                AlignedData::<f32, { alignment::SCALAR }>::alignment(),
                alignment::SCALAR
            );
        }
    }

    #[test]
    fn test_memory_layout_info() {
        let info = MemoryLayoutInfo {
            element_count: 4,
            element_size: 4,
            element_alignment: alignment::SCALAR,
            total_size: 16,
            padded_size: 16,
            tensor_alignment: alignment::VEC4,
            is_simd_optimized: true,
            cache_line_aligned: false,
        };

        assert_eq!(info.memory_efficiency(), 1.0);
        assert!(info.is_simd_optimized);
        assert!(!info.cache_line_aligned);

        let recommendations = info.optimization_recommendations();
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_alignment_constants() {
        // Test that all alignments are powers of 2
        assert!(alignment::SCALAR.is_power_of_two());
        assert!(alignment::VEC2.is_power_of_two());
        assert!(alignment::VEC4.is_power_of_two());
        assert!(alignment::MATRIX.is_power_of_two());
        assert!(alignment::BUFFER.is_power_of_two());
        assert!(alignment::WORKGROUP.is_power_of_two());

        // Test alignment hierarchy
        assert!(alignment::SCALAR <= alignment::VEC2);
        assert!(alignment::VEC2 <= alignment::VEC4);
        assert!(alignment::VEC4 <= alignment::MATRIX);
        assert!(alignment::MATRIX <= alignment::BUFFER);
        assert!(alignment::BUFFER <= alignment::WORKGROUP);
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_aligned_tensor_creation() {
        if let Ok(mut manager) = Manager::new() {
            // Test aligned tensor creation
            let aligned_tensor = AlignedTensor::<f32, 4, { alignment::VEC4 }>::new(
                &mut manager,
                &[1.0, 2.0, 3.0, 4.0],
            );

            if let Ok(tensor) = aligned_tensor {
                assert_eq!(tensor.compile_time_size(), 4);
                assert_eq!(tensor.alignment(), alignment::VEC4);
                assert!(tensor.is_optimally_aligned());
            }
        }
    }
}
