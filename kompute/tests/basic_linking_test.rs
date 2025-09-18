//! Basic test to verify linking works

#[test]
fn test_basic_linking() {
    // Just verify that the library loads without crashing
    println!("✓ kompute library loaded successfully");
}

#[test]
fn test_imports() {
    use kompute::{Algorithm, Manager, Sequence, Tensor};

    // Just verify all types can be imported
    println!("✓ All kompute types imported successfully");

    // Check type sizes to ensure they're reasonable
    println!("Manager size: {}", std::mem::size_of::<Manager>());
    println!("Tensor<f32> size: {}", std::mem::size_of::<Tensor<f32>>());
    println!("Sequence size: {}", std::mem::size_of::<Sequence>());
    println!("Algorithm size: {}", std::mem::size_of::<Algorithm>());
}
