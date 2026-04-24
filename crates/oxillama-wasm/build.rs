fn main() {
    // Inform the build system about WASM SIMD128 feature availability.
    // When building for wasm32, RUSTFLAGS should include "+simd128" from .cargo/config.toml
    // to enable SIMD-accelerated dequantization in browsers.
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-changed=.cargo/config.toml");
}
