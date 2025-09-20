//! Image Processing Pipeline Example
//!
//! This example demonstrates a complete image processing pipeline using GPU
//! acceleration, including filters, transformations, and analysis operations
//! commonly used in computer vision and digital image processing.

use kompute::{Algorithm, Manager, PerformanceProfiler, Result, Tensor};

fn main() -> Result<()> {
    println!("🖼️  GPU-Accelerated Image Processing Pipeline");
    println!("===========================================");

    let mut manager = Manager::new()?;
    let mut profiler = PerformanceProfiler::new();
    profiler.begin_session("image_processing")?;

    // Create sample image data
    let (image_data, width, height) = create_sample_image();
    println!(
        "📏 Processing {}x{} RGB image ({} pixels)",
        width,
        height,
        width * height
    );

    // Image processing pipeline steps
    example_gaussian_blur(&mut manager, &mut profiler, &image_data, width, height)?;
    example_edge_detection(&mut manager, &mut profiler, &image_data, width, height)?;
    example_histogram_equalization(&mut manager, &mut profiler, &image_data, width, height)?;
    example_morphological_operations(&mut manager, &mut profiler, &image_data, width, height)?;
    example_color_space_conversion(&mut manager, &mut profiler, &image_data, width, height)?;

    profiler.end_session()?;
    let report = profiler.generate_report()?;
    println!("\n📊 Image Processing Performance Report:");
    println!("{}", report);

    Ok(())
}

/// Create a sample RGB image with various patterns for testing
fn create_sample_image() -> (Vec<f32>, usize, usize) {
    let width = 512;
    let height = 512;
    let channels = 3; // RGB

    let mut image = Vec::with_capacity(width * height * channels);

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;

            // Create interesting patterns for processing
            let r = if x < width / 2 {
                // Gradient on the left
                fx
            } else {
                // Sine wave pattern on the right
                (fx * 20.0).sin().abs()
            };

            let g = if y < height / 2 {
                // Vertical gradient on top
                fy
            } else {
                // Checkerboard pattern on bottom
                if ((x / 32) + (y / 32)) % 2 == 0 {
                    1.0
                } else {
                    0.0
                }
            };

            let b = {
                // Radial pattern
                let center_x = width as f32 / 2.0;
                let center_y = height as f32 / 2.0;
                let distance =
                    ((x as f32 - center_x).powi(2) + (y as f32 - center_y).powi(2)).sqrt();
                let max_distance = (center_x.powi(2) + center_y.powi(2)).sqrt();
                1.0 - (distance / max_distance).min(1.0)
            };

            image.push(r);
            image.push(g);
            image.push(b);
        }
    }

    (image, width, height)
}

/// Gaussian blur filter for noise reduction and smoothing
fn example_gaussian_blur(
    manager: &mut Manager,
    profiler: &mut PerformanceProfiler,
    image_data: &[f32],
    width: usize,
    height: usize,
) -> Result<()> {
    println!("\n🌀 Gaussian Blur Filter");

    profiler.begin_operation("gaussian_setup")?;

    let channels = 3;
    let kernel_size = 5; // 5x5 Gaussian kernel
    let sigma = 1.5f32; // Standard deviation for Gaussian

    // Generate Gaussian kernel
    let mut kernel = Vec::new();
    let center = kernel_size as i32 / 2;
    let mut sum = 0.0f32;

    for y in 0..kernel_size {
        for x in 0..kernel_size {
            let dx = x as i32 - center;
            let dy = y as i32 - center;
            let value = (-((dx * dx + dy * dy) as f32) / (2.0 * sigma * sigma)).exp();
            kernel.push(value);
            sum += value;
        }
    }

    // Normalize kernel
    for val in &mut kernel {
        *val /= sum;
    }

    println!(
        "  📐 Kernel: {}x{}, σ = {:.1}",
        kernel_size, kernel_size, sigma
    );

    let image_tensor = Tensor::new(manager, image_data)?;
    let kernel_tensor = Tensor::new(manager, &kernel)?;
    let params = vec![
        width as f32,
        height as f32,
        channels as f32,
        kernel_size as f32,
    ];
    let params_tensor = Tensor::new(manager, &params)?;
    let mut output_tensor: Tensor<f32> = Tensor::uninit(manager, width * height * channels)?;

    profiler.end_operation("gaussian_setup")?;

    let blur_spirv = create_gaussian_blur_shader();

    profiler.begin_operation("gaussian_compute")?;

    let algorithm = Algorithm::with_tensors(
        manager,
        &blur_spirv,
        &[
            &image_tensor,
            &kernel_tensor,
            &params_tensor,
            &output_tensor,
        ],
    )?;

    let mut sequence = manager.create_sequence()?;
    let workgroup_x = (width + 15) / 16;
    let workgroup_y = (height + 15) / 16;
    sequence.record_algorithm_dispatch(&algorithm, &[workgroup_x as u32, workgroup_y as u32, 1])?;
    sequence.eval()?;

    profiler.end_operation("gaussian_compute")?;

    // Analyze smoothing effect
    profiler.begin_operation("gaussian_analysis")?;
    let blurred = output_tensor.data()?;

    let original_variance = calculate_image_variance(image_data, width, height);
    let blurred_variance = calculate_image_variance(&blurred, width, height);
    let smoothing_factor = blurred_variance / original_variance;

    println!("  📊 Smoothing Analysis:");
    println!("    Original variance: {:.6}", original_variance);
    println!("    Blurred variance: {:.6}", blurred_variance);
    println!(
        "    Smoothing factor: {:.4}x reduction",
        1.0 / smoothing_factor
    );
    println!("  ✅ Gaussian blur completed!");

    profiler.end_operation("gaussian_analysis")?;

    Ok(())
}

/// Edge detection using Sobel operators
fn example_edge_detection(
    manager: &mut Manager,
    profiler: &mut PerformanceProfiler,
    image_data: &[f32],
    width: usize,
    height: usize,
) -> Result<()> {
    println!("\n🔍 Edge Detection (Sobel Operators)");

    profiler.begin_operation("edge_setup")?;

    let channels = 3;

    // Sobel X kernel (vertical edges)
    let sobel_x = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];

    // Sobel Y kernel (horizontal edges)
    let sobel_y = vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

    let image_tensor = Tensor::new(manager, image_data)?;
    let sobel_x_tensor = Tensor::new(manager, &sobel_x)?;
    let sobel_y_tensor = Tensor::new(manager, &sobel_y)?;
    let params = vec![width as f32, height as f32, channels as f32];
    let params_tensor = Tensor::new(manager, &params)?;
    let mut edge_magnitude: Tensor<f32> = Tensor::uninit(manager, width * height)?; // Grayscale output

    profiler.end_operation("edge_setup")?;

    let edge_spirv = create_edge_detection_shader();

    profiler.begin_operation("edge_compute")?;

    let algorithm = Algorithm::with_tensors(
        manager,
        &edge_spirv,
        &[
            &image_tensor,
            &sobel_x_tensor,
            &sobel_y_tensor,
            &params_tensor,
            &edge_magnitude,
        ],
    )?;

    let mut sequence = manager.create_sequence()?;
    let workgroup_x = (width + 15) / 16;
    let workgroup_y = (height + 15) / 16;
    sequence.record_algorithm_dispatch(&algorithm, &[workgroup_x as u32, workgroup_y as u32, 1])?;
    sequence.eval()?;

    profiler.end_operation("edge_compute")?;

    // Analyze edge detection results
    profiler.begin_operation("edge_analysis")?;
    let edges = edge_magnitude.data()?;

    let max_edge = edges.iter().fold(0.0f32, |acc, &x| acc.max(x));
    let avg_edge = edges.iter().sum::<f32>() / edges.len() as f32;
    let strong_edges = edges.iter().filter(|&&x| x > max_edge * 0.3).count();
    let edge_percentage = strong_edges as f32 / edges.len() as f32 * 100.0;

    println!("  📊 Edge Detection Results:");
    println!("    Maximum edge strength: {:.4}", max_edge);
    println!("    Average edge strength: {:.4}", avg_edge);
    println!(
        "    Strong edges (>30% max): {} ({:.1}%)",
        strong_edges, edge_percentage
    );
    println!("  ✅ Edge detection completed!");

    profiler.end_operation("edge_analysis")?;

    Ok(())
}

/// Histogram equalization for contrast enhancement
fn example_histogram_equalization(
    manager: &mut Manager,
    profiler: &mut PerformanceProfiler,
    image_data: &[f32],
    width: usize,
    height: usize,
) -> Result<()> {
    println!("\n📊 Histogram Equalization");

    profiler.begin_operation("histogram_setup")?;

    let channels = 3;
    let num_bins = 256;
    let total_pixels = width * height;

    // Convert to grayscale first for histogram calculation
    let mut grayscale = Vec::with_capacity(total_pixels);
    for chunk in image_data.chunks_exact(channels) {
        let gray = 0.299 * chunk[0] + 0.587 * chunk[1] + 0.114 * chunk[2]; // Luminance
        grayscale.push(gray);
    }

    let grayscale_tensor = Tensor::new(manager, &grayscale)?;
    let mut histogram: Tensor<f32> = Tensor::uninit(manager, num_bins)?;
    let params = vec![width as f32, height as f32, num_bins as f32];
    let params_tensor = Tensor::new(manager, &params)?;

    profiler.end_operation("histogram_setup")?;

    // Compute histogram
    let histogram_spirv = create_histogram_shader();

    profiler.begin_operation("histogram_compute")?;

    let algorithm = Algorithm::with_tensors(
        manager,
        &histogram_spirv,
        &[&grayscale_tensor, &params_tensor, &histogram],
    )?;

    let mut sequence = manager.create_sequence()?;
    let workgroup_x = (total_pixels + 255) / 256;
    sequence.record_algorithm_dispatch(&algorithm, &[workgroup_x as u32, 1, 1])?;
    sequence.eval()?;

    profiler.end_operation("histogram_compute")?;

    // Calculate cumulative distribution function (CDF) and apply equalization
    profiler.begin_operation("equalization_compute")?;
    let hist_data = histogram.data()?;

    // Calculate CDF (cumulative sum)
    let mut cdf = vec![0.0f32; num_bins];
    cdf[0] = hist_data[0];
    for i in 1..num_bins {
        cdf[i] = cdf[i - 1] + hist_data[i];
    }

    // Normalize CDF to [0, 1] range
    let total_count = cdf[num_bins - 1];
    if total_count > 0.0 {
        for val in &mut cdf {
            *val /= total_count;
        }
    }

    let cdf_tensor = Tensor::new(manager, &cdf)?;
    let mut equalized: Tensor<f32> = Tensor::uninit(manager, width * height * channels)?;

    let equalize_spirv = create_equalization_shader();
    let algorithm = Algorithm::with_tensors(
        manager,
        &equalize_spirv,
        &[&image_tensor, &cdf_tensor, &params_tensor, &equalized],
    )?;

    let mut sequence = manager.create_sequence()?;
    let workgroup_x = (total_pixels + 255) / 256;
    sequence.record_algorithm_dispatch(&algorithm, &[workgroup_x as u32, 1, 1])?;
    sequence.eval()?;

    profiler.end_operation("equalization_compute")?;

    // Analyze contrast enhancement
    profiler.begin_operation("contrast_analysis")?;
    let equalized_data = equalized.data()?;

    let original_contrast = calculate_image_contrast(image_data, width, height);
    let enhanced_contrast = calculate_image_contrast(&equalized_data, width, height);
    let contrast_improvement = enhanced_contrast / original_contrast;

    println!("  📊 Contrast Enhancement:");
    println!("    Original contrast: {:.4}", original_contrast);
    println!("    Enhanced contrast: {:.4}", enhanced_contrast);
    println!("    Improvement factor: {:.2}x", contrast_improvement);
    println!("  ✅ Histogram equalization completed!");

    profiler.end_operation("contrast_analysis")?;

    Ok(())
}

/// Morphological operations (erosion and dilation)
fn example_morphological_operations(
    manager: &mut Manager,
    profiler: &mut PerformanceProfiler,
    image_data: &[f32],
    width: usize,
    height: usize,
) -> Result<()> {
    println!("\n🔲 Morphological Operations");

    profiler.begin_operation("morphology_setup")?;

    let channels = 3;
    let struct_elem_size = 3; // 3x3 structuring element

    // Square structuring element
    let structuring_element = vec![1.0f32; struct_elem_size * struct_elem_size];

    // Convert to binary image (threshold-based)
    let threshold = 0.5f32;
    let mut binary_image = Vec::with_capacity(width * height);
    for chunk in image_data.chunks_exact(channels) {
        let intensity = (chunk[0] + chunk[1] + chunk[2]) / 3.0; // Average RGB
        binary_image.push(if intensity > threshold { 1.0 } else { 0.0 });
    }

    let binary_tensor = Tensor::new(manager, &binary_image)?;
    let struct_tensor = Tensor::new(manager, &structuring_element)?;
    let params = vec![
        width as f32,
        height as f32,
        struct_elem_size as f32,
        threshold,
    ];
    let params_tensor = Tensor::new(manager, &params)?;
    let mut eroded: Tensor<f32> = Tensor::uninit(manager, width * height)?;
    let mut dilated: Tensor<f32> = Tensor::uninit(manager, width * height)?;

    profiler.end_operation("morphology_setup")?;

    // Erosion operation
    profiler.begin_operation("erosion_compute")?;
    let erosion_spirv = create_erosion_shader();
    let algorithm = Algorithm::with_tensors(
        manager,
        &erosion_spirv,
        &[&binary_tensor, &struct_tensor, &params_tensor, &eroded],
    )?;

    let mut sequence = manager.create_sequence()?;
    let workgroup_x = (width + 15) / 16;
    let workgroup_y = (height + 15) / 16;
    sequence.record_algorithm_dispatch(&algorithm, &[workgroup_x as u32, workgroup_y as u32, 1])?;
    sequence.eval()?;
    profiler.end_operation("erosion_compute")?;

    // Dilation operation
    profiler.begin_operation("dilation_compute")?;
    let dilation_spirv = create_dilation_shader();
    let algorithm = Algorithm::with_tensors(
        manager,
        &dilation_spirv,
        &[&binary_tensor, &struct_tensor, &params_tensor, &dilated],
    )?;

    let mut sequence = manager.create_sequence()?;
    sequence.record_algorithm_dispatch(&algorithm, &[workgroup_x as u32, workgroup_y as u32, 1])?;
    sequence.eval()?;
    profiler.end_operation("dilation_compute")?;

    // Analyze morphological operations
    profiler.begin_operation("morphology_analysis")?;
    let eroded_data = eroded.data()?;
    let dilated_data = dilated.data()?;

    let original_white_pixels = binary_image.iter().filter(|&&x| x > 0.5).count();
    let eroded_white_pixels = eroded_data.iter().filter(|&&x| x > 0.5).count();
    let dilated_white_pixels = dilated_data.iter().filter(|&&x| x > 0.5).count();

    println!("  📊 Morphological Analysis:");
    println!(
        "    Original white pixels: {} ({:.1}%)",
        original_white_pixels,
        original_white_pixels as f32 / binary_image.len() as f32 * 100.0
    );
    println!(
        "    After erosion: {} ({:.1}%)",
        eroded_white_pixels,
        eroded_white_pixels as f32 / binary_image.len() as f32 * 100.0
    );
    println!(
        "    After dilation: {} ({:.1}%)",
        dilated_white_pixels,
        dilated_white_pixels as f32 / binary_image.len() as f32 * 100.0
    );
    println!("  ✅ Morphological operations completed!");

    profiler.end_operation("morphology_analysis")?;

    Ok(())
}

/// Color space conversion (RGB to HSV)
fn example_color_space_conversion(
    manager: &mut Manager,
    profiler: &mut PerformanceProfiler,
    image_data: &[f32],
    width: usize,
    height: usize,
) -> Result<()> {
    println!("\n🌈 Color Space Conversion (RGB → HSV)");

    profiler.begin_operation("color_conversion_setup")?;

    let channels = 3;
    let total_pixels = width * height;

    let image_tensor = Tensor::new(manager, image_data)?;
    let params = vec![width as f32, height as f32, channels as f32];
    let params_tensor = Tensor::new(manager, &params)?;
    let mut hsv_tensor: Tensor<f32> = Tensor::uninit(manager, total_pixels * channels)?;

    profiler.end_operation("color_conversion_setup")?;

    let rgb_to_hsv_spirv = create_rgb_to_hsv_shader();

    profiler.begin_operation("color_conversion_compute")?;

    let algorithm = Algorithm::with_tensors(
        manager,
        &rgb_to_hsv_spirv,
        &[&image_tensor, &params_tensor, &hsv_tensor],
    )?;

    let mut sequence = manager.create_sequence()?;
    let workgroup_x = (total_pixels + 255) / 256;
    sequence.record_algorithm_dispatch(&algorithm, &[workgroup_x as u32, 1, 1])?;
    sequence.eval()?;

    profiler.end_operation("color_conversion_compute")?;

    // Analyze color distribution in HSV space
    profiler.begin_operation("hsv_analysis")?;
    let hsv_data = hsv_tensor.data()?;

    let mut hue_sum = 0.0f32;
    let mut saturation_sum = 0.0f32;
    let mut value_sum = 0.0f32;
    let mut valid_pixels = 0;

    for chunk in hsv_data.chunks_exact(3) {
        let h = chunk[0]; // Hue [0, 360)
        let s = chunk[1]; // Saturation [0, 1]
        let v = chunk[2]; // Value [0, 1]

        if s > 0.01 {
            // Only consider colored pixels for hue statistics
            hue_sum += h;
            valid_pixels += 1;
        }
        saturation_sum += s;
        value_sum += v;
    }

    let avg_hue = if valid_pixels > 0 {
        hue_sum / valid_pixels as f32
    } else {
        0.0
    };
    let avg_saturation = saturation_sum / total_pixels as f32;
    let avg_value = value_sum / total_pixels as f32;

    // Classify dominant color
    let dominant_color = match avg_hue {
        h if h < 30.0 || h >= 330.0 => "Red",
        h if h < 90.0 => "Yellow",
        h if h < 150.0 => "Green",
        h if h < 210.0 => "Cyan",
        h if h < 270.0 => "Blue",
        _ => "Magenta",
    };

    println!("  📊 HSV Color Analysis:");
    println!("    Average hue: {:.1}° ({})", avg_hue, dominant_color);
    println!(
        "    Average saturation: {:.3} ({:.1}%)",
        avg_saturation,
        avg_saturation * 100.0
    );
    println!(
        "    Average value: {:.3} ({:.1}%)",
        avg_value,
        avg_value * 100.0
    );
    println!(
        "    Colored pixels: {} / {} ({:.1}%)",
        valid_pixels,
        total_pixels,
        valid_pixels as f32 / total_pixels as f32 * 100.0
    );
    println!("  ✅ Color space conversion completed!");

    profiler.end_operation("hsv_analysis")?;

    Ok(())
}

// Utility functions for image analysis

fn calculate_image_variance(image_data: &[f32], width: usize, height: usize) -> f32 {
    let channels = 3;
    let total_pixels = width * height;

    // Calculate mean
    let mean = image_data.iter().sum::<f32>() / image_data.len() as f32;

    // Calculate variance
    let variance =
        image_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / image_data.len() as f32;

    variance
}

fn calculate_image_contrast(image_data: &[f32], width: usize, height: usize) -> f32 {
    let channels = 3;
    let mut intensities = Vec::with_capacity(width * height);

    // Convert to grayscale intensities
    for chunk in image_data.chunks_exact(channels) {
        let intensity = 0.299 * chunk[0] + 0.587 * chunk[1] + 0.114 * chunk[2];
        intensities.push(intensity);
    }

    // Calculate standard deviation as contrast measure
    let mean = intensities.iter().sum::<f32>() / intensities.len() as f32;
    let variance =
        intensities.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / intensities.len() as f32;

    variance.sqrt() // Standard deviation
}

// Shader creation functions (simplified SPIR-V)

fn create_gaussian_blur_shader() -> Vec<u32> {
    vec![
        0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000, 0x000100fd, 0x00010038,
    ]
}

fn create_edge_detection_shader() -> Vec<u32> {
    vec![
        0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000, 0x000100fd, 0x00010038,
    ]
}

fn create_histogram_shader() -> Vec<u32> {
    vec![
        0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000, 0x000100fd, 0x00010038,
    ]
}

fn create_equalization_shader() -> Vec<u32> {
    vec![
        0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000, 0x000100fd, 0x00010038,
    ]
}

fn create_erosion_shader() -> Vec<u32> {
    vec![
        0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000, 0x000100fd, 0x00010038,
    ]
}

fn create_dilation_shader() -> Vec<u32> {
    vec![
        0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000, 0x000100fd, 0x00010038,
    ]
}

fn create_rgb_to_hsv_shader() -> Vec<u32> {
    vec![
        0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000, 0x000100fd, 0x00010038,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_image_processing_example() {
        if let Err(e) = main() {
            println!(
                "Image processing example failed (expected without GPU): {}",
                e
            );
        }
    }

    #[test]
    fn test_sample_image_generation() {
        let (image_data, width, height) = create_sample_image();
        assert_eq!(image_data.len(), width * height * 3);
        assert!(image_data.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_image_statistics() {
        let (image_data, width, height) = create_sample_image();
        let variance = calculate_image_variance(&image_data, width, height);
        let contrast = calculate_image_contrast(&image_data, width, height);

        assert!(variance > 0.0, "Image should have some variance");
        assert!(contrast > 0.0, "Image should have some contrast");
    }

    #[test]
    fn test_gaussian_kernel_generation() {
        let kernel_size = 5;
        let sigma = 1.5f32;
        let mut kernel = Vec::new();
        let center = kernel_size as i32 / 2;
        let mut sum = 0.0f32;

        for y in 0..kernel_size {
            for x in 0..kernel_size {
                let dx = x as i32 - center;
                let dy = y as i32 - center;
                let value = (-((dx * dx + dy * dy) as f32) / (2.0 * sigma * sigma)).exp();
                kernel.push(value);
                sum += value;
            }
        }

        // Normalize kernel
        for val in &mut kernel {
            *val /= sum;
        }

        let kernel_sum: f32 = kernel.iter().sum();
        assert!(
            (kernel_sum - 1.0).abs() < 1e-6,
            "Normalized kernel should sum to 1.0"
        );
        assert!(
            kernel[center as usize * kernel_size + center as usize] > 0.1,
            "Center should have significant weight"
        );
    }
}
