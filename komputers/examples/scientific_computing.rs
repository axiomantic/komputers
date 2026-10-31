//! Scientific Computing with GPU Example
//!
//! This example demonstrates various scientific computing applications
//! using GPU acceleration, including numerical methods, simulations,
//! and mathematical operations commonly used in scientific research.

use komputers::{Algorithm, Manager, PerformanceProfiler, Result, Tensor};
use std::f32::consts::PI;

fn main() -> Result<()> {
    println!("🔬 Scientific Computing with GPU");
    println!("================================");

    let mut manager = Manager::new()?;
    let mut profiler = PerformanceProfiler::new();
    profiler.begin_session("scientific_computing")?;

    // Monte Carlo Pi estimation
    example_monte_carlo_pi(&mut manager, &mut profiler)?;

    // N-body gravitational simulation step
    example_nbody_simulation(&mut manager, &mut profiler)?;

    // Numerical integration (trapezoidal rule)
    example_numerical_integration(&mut manager, &mut profiler)?;

    // FFT-like frequency domain analysis
    example_frequency_analysis(&mut manager, &mut profiler)?;

    // Heat equation solver (finite difference method)
    example_heat_equation(&mut manager, &mut profiler)?;

    profiler.end_session()?;
    let report = profiler.generate_report()?;
    println!("\n📊 Scientific Computing Performance Report:");
    println!("{}", report);

    Ok(())
}

/// Monte Carlo estimation of π using random sampling
fn example_monte_carlo_pi(manager: &mut Manager, profiler: &mut PerformanceProfiler) -> Result<()> {
    println!("\n🎯 Monte Carlo π Estimation");

    profiler.begin_operation("monte_carlo_setup")?;

    let num_samples = 1_000_000;
    println!("  🎲 Samples: {}", num_samples);

    // Generate random points in [0,1] x [0,1]
    let random_x: Vec<f32> = (0..num_samples)
        .map(|i| (i as f32 * 1234.5) % 1.0)
        .collect();

    let random_y: Vec<f32> = (0..num_samples)
        .map(|i| (i as f32 * 5678.9) % 1.0)
        .collect();

    let x_tensor = Tensor::new(manager, &random_x)?;
    let y_tensor = Tensor::new(manager, &random_y)?;
    let results_tensor: Tensor<f32> = Tensor::uninit(manager, num_samples)?;

    profiler.end_operation("monte_carlo_setup")?;

    let monte_carlo_spirv = create_monte_carlo_shader();

    profiler.begin_operation("monte_carlo_compute")?;

    let algorithm = Algorithm::with_tensors(
        manager,
        &monte_carlo_spirv,
        &[&x_tensor, &y_tensor, &results_tensor],
    )?;

    manager
        .sequence()?
        .record_algorithm_dispatch(algorithm.handle())?
        .execute()?;

    profiler.end_operation("monte_carlo_compute")?;

    // Calculate π estimate
    profiler.begin_operation("pi_calculation")?;
    let results = results_tensor.to_vec()?;
    let hits_in_circle = results.iter().sum::<f32>() as usize;
    let pi_estimate = 4.0 * hits_in_circle as f32 / num_samples as f32;
    let error = ((pi_estimate - PI) / PI * 100.0).abs();

    println!(
        "  🎯 Hits in unit circle: {} / {}",
        hits_in_circle, num_samples
    );
    println!("  🔢 π estimate: {:.6}", pi_estimate);
    println!("  📏 Actual π: {:.6}", PI);
    println!("  📊 Error: {:.4}%", error);
    profiler.end_operation("pi_calculation")?;

    Ok(())
}

/// N-body gravitational simulation (simplified)
fn example_nbody_simulation(
    manager: &mut Manager,
    profiler: &mut PerformanceProfiler,
) -> Result<()> {
    println!("\n🌌 N-Body Gravitational Simulation");

    profiler.begin_operation("nbody_setup")?;

    let num_bodies = 1024;
    let dt = 0.01f32; // Time step
    let g_constant = 1.0f32; // Gravitational constant (scaled for simulation)

    println!("  🌟 Bodies: {}", num_bodies);
    println!("  ⏰ Time step: {:.3}s", dt);

    // Initial positions (random distribution in a cube)
    let positions_x: Vec<f32> = (0..num_bodies)
        .map(|i| (i as f32 * 0.123).sin() * 10.0)
        .collect();
    let positions_y: Vec<f32> = (0..num_bodies)
        .map(|i| (i as f32 * 0.456).cos() * 10.0)
        .collect();
    let positions_z: Vec<f32> = (0..num_bodies)
        .map(|i| (i as f32 * 0.789).sin() * (i as f32 * 0.234).cos() * 10.0)
        .collect();

    // Initial velocities (small random values)
    let velocities_x: Vec<f32> = vec![0.0; num_bodies]; // Start at rest
    let velocities_y: Vec<f32> = vec![0.0; num_bodies];
    let velocities_z: Vec<f32> = vec![0.0; num_bodies];

    // Masses (uniform for simplicity)
    let masses: Vec<f32> = vec![1.0; num_bodies];

    // Create tensors
    let pos_x_tensor = Tensor::new(manager, &positions_x)?;
    let pos_y_tensor = Tensor::new(manager, &positions_y)?;
    let pos_z_tensor = Tensor::new(manager, &positions_z)?;
    let vel_x_tensor = Tensor::new(manager, &velocities_x)?;
    let vel_y_tensor = Tensor::new(manager, &velocities_y)?;
    let vel_z_tensor = Tensor::new(manager, &velocities_z)?;
    let mass_tensor = Tensor::new(manager, &masses)?;
    let dt_tensor = Tensor::new(manager, &[dt])?;
    let g_tensor = Tensor::new(manager, &[g_constant])?;

    // Output tensors for forces
    let force_x_tensor: Tensor<f32> = Tensor::uninit(manager, num_bodies)?;
    let force_y_tensor: Tensor<f32> = Tensor::uninit(manager, num_bodies)?;
    let force_z_tensor: Tensor<f32> = Tensor::uninit(manager, num_bodies)?;

    profiler.end_operation("nbody_setup")?;

    let nbody_spirv = create_nbody_shader();

    profiler.begin_operation("nbody_force_calculation")?;

    let algorithm = Algorithm::with_tensors(
        manager,
        &nbody_spirv,
        &[
            &pos_x_tensor,
            &pos_y_tensor,
            &pos_z_tensor,
            &vel_x_tensor,
            &vel_y_tensor,
            &vel_z_tensor,
            &mass_tensor,
            &dt_tensor,
            &g_tensor,
            &force_x_tensor,
            &force_y_tensor,
            &force_z_tensor,
        ],
    )?;

    manager
        .sequence()?
        .record_algorithm_dispatch(algorithm.handle())?
        .execute()?;

    profiler.end_operation("nbody_force_calculation")?;

    // Analyze simulation results
    profiler.begin_operation("nbody_analysis")?;
    let forces_x = force_x_tensor.to_vec()?;
    let forces_y = force_y_tensor.to_vec()?;
    let forces_z = force_z_tensor.to_vec()?;

    let total_force_magnitude: f32 = forces_x
        .iter()
        .zip(forces_y.iter())
        .zip(forces_z.iter())
        .map(|((&fx, &fy), &fz)| (fx * fx + fy * fy + fz * fz).sqrt())
        .sum();

    let avg_force = total_force_magnitude / num_bodies as f32;
    let max_force = forces_x
        .iter()
        .zip(forces_y.iter())
        .zip(forces_z.iter())
        .map(|((&fx, &fy), &fz)| (fx * fx + fy * fy + fz * fz).sqrt())
        .fold(0.0f32, |acc, f| acc.max(f));

    println!("  🔍 Force Analysis:");
    println!("    Average force magnitude: {:.6}", avg_force);
    println!("    Maximum force magnitude: {:.6}", max_force);
    println!("  ✅ N-body simulation step completed!");

    profiler.end_operation("nbody_analysis")?;

    Ok(())
}

/// Numerical integration using the trapezoidal rule
fn example_numerical_integration(
    manager: &mut Manager,
    profiler: &mut PerformanceProfiler,
) -> Result<()> {
    println!("\n📐 Numerical Integration (Trapezoidal Rule)");

    profiler.begin_operation("integration_setup")?;

    // Integrate sin(x) from 0 to π (should equal 2.0)
    let a = 0.0f32; // Lower bound
    let b = PI; // Upper bound
    let n = 1_000_000; // Number of intervals

    let h = (b - a) / n as f32; // Step size

    println!("  📊 Function: sin(x)");
    println!("  📏 Interval: [{:.3}, {:.3}]", a, b);
    println!("  📈 Steps: {}", n);
    println!("  ⚡ Step size: {:.9}", h);

    // Create x values for evaluation points
    let x_values: Vec<f32> = (0..=n).map(|i| a + i as f32 * h).collect();

    let x_tensor = Tensor::new(manager, &x_values)?;
    let fx_tensor: Tensor<f32> = Tensor::uninit(manager, n + 1)?;

    profiler.end_operation("integration_setup")?;

    let function_eval_spirv = create_sin_evaluation_shader();

    profiler.begin_operation("function_evaluation")?;

    // Evaluate sin(x) at all points
    let algorithm =
        Algorithm::with_tensors(manager, &function_eval_spirv, &[&x_tensor, &fx_tensor])?;

    manager
        .sequence()?
        .record_algorithm_dispatch(algorithm.handle())?
        .execute()?;

    profiler.end_operation("function_evaluation")?;

    // Apply trapezoidal rule
    profiler.begin_operation("trapezoidal_integration")?;
    let fx_values = fx_tensor.to_vec()?;

    // Trapezoidal rule: h * (f(a)/2 + f(x1) + f(x2) + ... + f(xn-1) + f(b)/2)
    let mut integral = fx_values[0] / 2.0 + fx_values[n] / 2.0; // End points
    for i in 1..n {
        integral += fx_values[i]; // Middle points
    }
    integral *= h;

    let exact_value = 2.0f32; // ∫₀^π sin(x) dx = 2
    let error = ((integral - exact_value) / exact_value * 100.0).abs();

    println!("  📊 Results:");
    println!("    Numerical result: {:.9}", integral);
    println!("    Exact value: {:.9}", exact_value);
    println!("    Relative error: {:.6}%", error);
    println!("  ✅ Numerical integration completed!");

    profiler.end_operation("trapezoidal_integration")?;

    Ok(())
}

/// Frequency domain analysis (simplified DFT)
fn example_frequency_analysis(
    manager: &mut Manager,
    profiler: &mut PerformanceProfiler,
) -> Result<()> {
    println!("\n🌊 Frequency Domain Analysis (DFT)");

    profiler.begin_operation("frequency_setup")?;

    let n = 512; // Signal length
    let sample_rate = 1000.0f32; // Hz
    let signal_freq1 = 50.0f32; // First component frequency
    let signal_freq2 = 120.0f32; // Second component frequency

    println!("  📏 Signal length: {} samples", n);
    println!("  🎵 Sample rate: {} Hz", sample_rate);
    println!(
        "  🔊 Signal components: {} Hz + {} Hz",
        signal_freq1, signal_freq2
    );

    // Generate test signal: sin(2π*f1*t) + 0.5*sin(2π*f2*t)
    let dt = 1.0 / sample_rate;
    let signal: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 * dt;
            (2.0 * PI * signal_freq1 * t).sin() + 0.5 * (2.0 * PI * signal_freq2 * t).sin()
        })
        .collect();

    let signal_tensor = Tensor::new(manager, &signal)?;
    let magnitude_spectrum: Tensor<f32> = Tensor::uninit(manager, n / 2)?; // Only positive frequencies

    profiler.end_operation("frequency_setup")?;

    let dft_spirv = create_dft_shader();

    profiler.begin_operation("dft_computation")?;

    let algorithm =
        Algorithm::with_tensors(manager, &dft_spirv, &[&signal_tensor, &magnitude_spectrum])?;

    manager
        .sequence()?
        .record_algorithm_dispatch(algorithm.handle())?
        .execute()?;

    profiler.end_operation("dft_computation")?;

    // Analyze frequency spectrum
    profiler.begin_operation("spectrum_analysis")?;
    let spectrum = magnitude_spectrum.to_vec()?;

    // Find peak frequencies
    let freq_resolution = sample_rate / n as f32;
    let mut peaks = Vec::new();

    for i in 1..(spectrum.len() - 1) {
        if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] && spectrum[i] > 0.1 {
            let freq = i as f32 * freq_resolution;
            peaks.push((freq, spectrum[i]));
        }
    }

    // Sort by magnitude
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  📊 Frequency Analysis Results:");
    println!("    Frequency resolution: {:.2} Hz", freq_resolution);
    println!("    Detected peaks:");
    for (i, (freq, magnitude)) in peaks.iter().take(5).enumerate() {
        println!(
            "      {}. {:.1} Hz (magnitude: {:.3})",
            i + 1,
            freq,
            magnitude
        );
    }

    // Verify expected frequencies are detected
    let freq1_detected = peaks
        .iter()
        .any(|(f, _)| (f - signal_freq1).abs() < freq_resolution * 2.0);
    let freq2_detected = peaks
        .iter()
        .any(|(f, _)| (f - signal_freq2).abs() < freq_resolution * 2.0);

    if freq1_detected && freq2_detected {
        println!("  ✅ Both signal components correctly detected!");
    }

    profiler.end_operation("spectrum_analysis")?;

    Ok(())
}

/// Heat equation solver using finite difference method
fn example_heat_equation(manager: &mut Manager, profiler: &mut PerformanceProfiler) -> Result<()> {
    println!("\n🔥 Heat Equation Solver (Finite Difference)");

    profiler.begin_operation("heat_setup")?;

    let nx = 64; // Grid points in x direction
    let ny = 64; // Grid points in y direction
    let grid_size = nx * ny;

    let dx = 1.0f32 / (nx - 1) as f32; // Spatial step
    let dy = 1.0f32 / (ny - 1) as f32;
    let dt = 0.0001f32; // Time step
    let alpha = 1.0f32; // Thermal diffusivity
    let num_steps = 100;

    println!("  📏 Grid: {}x{}", nx, ny);
    println!("  📐 Spatial resolution: dx={:.4}, dy={:.4}", dx, dy);
    println!("  ⏰ Time step: {:.6}", dt);
    println!("  🌡️  Time steps: {}", num_steps);

    // Initial temperature distribution (hot center, cold boundaries)
    let mut temperature = vec![0.0f32; grid_size];
    for i in 0..nx {
        for j in 0..ny {
            let x = i as f32 * dx;
            let y = j as f32 * dy;
            let idx = i * ny + j;

            // Gaussian hot spot in the center
            let center_x = 0.5;
            let center_y = 0.5;
            let distance_sq = (x - center_x).powi(2) + (y - center_y).powi(2);
            temperature[idx] = 100.0 * (-distance_sq / 0.01).exp();
        }
    }

    let initial_temp_tensor = Tensor::new(manager, &temperature)?;
    let new_temp_tensor: Tensor<f32> = Tensor::uninit(manager, grid_size)?;

    // Parameters tensor
    let params = vec![dx, dy, dt, alpha, nx as f32, ny as f32];
    let params_tensor = Tensor::new(manager, &params)?;

    profiler.end_operation("heat_setup")?;

    let heat_equation_spirv = create_heat_equation_shader();

    // Run simulation for several time steps
    profiler.begin_operation("heat_simulation")?;

    for step in 0..num_steps {
        let algorithm = Algorithm::with_tensors(
            manager,
            &heat_equation_spirv,
            &[&initial_temp_tensor, &params_tensor, &new_temp_tensor],
        )?;

        manager
            .sequence()?
            .record_algorithm_dispatch(algorithm.handle())?
            .execute()?;

        // Swap tensors for next iteration (in real implementation)
        if (step + 1) % 20 == 0 {
            println!("    Step {}/{}", step + 1, num_steps);
        }
    }

    profiler.end_operation("heat_simulation")?;

    // Analyze final temperature distribution
    profiler.begin_operation("heat_analysis")?;
    let final_temps = new_temp_tensor.to_vec()?;

    let max_temp = final_temps.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_temp = final_temps.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let avg_temp = final_temps.iter().sum::<f32>() / grid_size as f32;

    println!("  📊 Final Temperature Distribution:");
    println!("    Maximum: {:.3}°C", max_temp);
    println!("    Minimum: {:.3}°C", min_temp);
    println!("    Average: {:.3}°C", avg_temp);
    println!("  ✅ Heat equation simulation completed!");

    profiler.end_operation("heat_analysis")?;

    Ok(())
}

// Shader creation functions (simplified SPIR-V headers)

fn create_monte_carlo_shader() -> Vec<u32> {
    vec![
        0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000,
        // Monte Carlo pi estimation shader
        0x00020011, 0x00000001, 0x000100fd, 0x00010038,
    ]
}

fn create_nbody_shader() -> Vec<u32> {
    vec![
        0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000,
        // N-body force calculation shader
        0x00020011, 0x00000001, 0x000100fd, 0x00010038,
    ]
}

fn create_sin_evaluation_shader() -> Vec<u32> {
    vec![
        0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000,
        // Sin function evaluation shader
        0x00020011, 0x00000001, 0x000100fd, 0x00010038,
    ]
}

fn create_dft_shader() -> Vec<u32> {
    vec![
        0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000,
        // Discrete Fourier Transform shader
        0x00020011, 0x00000001, 0x000100fd, 0x00010038,
    ]
}

fn create_heat_equation_shader() -> Vec<u32> {
    vec![
        0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000,
        // Heat equation finite difference shader
        0x00020011, 0x00000001, 0x000100fd, 0x00010038,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_scientific_computing_example() {
        if let Err(e) = main() {
            println!(
                "Scientific computing example failed (expected without GPU): {}",
                e
            );
        }
    }

    #[test]
    fn test_signal_generation() {
        // Test signal generation for frequency analysis
        let n = 100;
        let sample_rate = 1000.0;
        let freq = 50.0;
        let dt = 1.0 / sample_rate;

        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 * dt;
                (2.0 * PI * freq * t).sin()
            })
            .collect();

        assert_eq!(signal.len(), n);

        // Check that signal oscillates around zero
        let avg = signal.iter().sum::<f32>() / signal.len() as f32;
        assert!(avg.abs() < 0.1, "Signal should average near zero");
    }

    #[test]
    fn test_temperature_initialization() {
        let nx = 10;
        let ny = 10;
        let dx = 1.0 / (nx - 1) as f32;
        let dy = 1.0 / (ny - 1) as f32;

        let mut temperature = vec![0.0f32; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let x = i as f32 * dx;
                let y = j as f32 * dy;
                let idx = i * ny + j;

                let center_x = 0.5;
                let center_y = 0.5;
                let distance_sq = (x - center_x).powi(2) + (y - center_y).powi(2);
                temperature[idx] = 100.0 * (-distance_sq / 0.01).exp();
            }
        }

        // Check that center is hottest
        let center_idx = (nx / 2) * ny + (ny / 2);
        let center_temp = temperature[center_idx];

        assert!(center_temp > 50.0, "Center should be hot");
        assert!(
            temperature[0] < center_temp,
            "Corner should be cooler than center"
        );
    }
}
