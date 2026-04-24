//! Integration tests: verify that error types satisfy Send + Sync bounds and
//! can be used across thread boundaries.

fn assert_send_sync<T: Send + Sync + 'static>() {}

#[test]
fn test_gguf_error_is_send_sync() {
    assert_send_sync::<oxillama_gguf::GgufError>();
}

#[test]
fn test_quant_error_is_send_sync() {
    assert_send_sync::<oxillama_quant::QuantError>();
}

#[test]
fn test_arch_error_is_send_sync() {
    assert_send_sync::<oxillama_arch::ArchError>();
}

#[test]
fn test_runtime_error_is_send_sync() {
    assert_send_sync::<oxillama_runtime::RuntimeError>();
}

#[test]
fn test_gguf_result_type_alias() {
    // Ensure the Result alias resolves correctly.
    fn _needs_result(_: oxillama_gguf::GgufResult<()>) {}
}

#[test]
fn test_quant_result_type_alias() {
    fn _needs_result(_: oxillama_quant::QuantResult<()>) {}
}

#[test]
fn test_arch_result_type_alias() {
    fn _needs_result(_: oxillama_arch::ArchResult<()>) {}
}

#[test]
fn test_runtime_result_type_alias() {
    fn _needs_result(_: oxillama_runtime::RuntimeResult<()>) {}
}
