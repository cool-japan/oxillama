//! DLPack v0.8 capsule producer/consumer for f32 CPU tensors.
//!
//! DLPack is the standard zero-copy tensor interchange protocol used by
//! PyTorch, TensorFlow, JAX, CuPy, and every other major ML framework.
//! This module implements the producer side (`Vec<f32>` → PyCapsule) and the
//! consumer side (PyCapsule → `Vec<f32>`) for 1-D and N-D CPU tensors of dtype
//! float32.
//!
//! # Wire format
//!
//! The `"dltensor"` PyCapsule contains a heap-allocated `DLManagedTensor`.
//! The tensor's `manager_ctx` field points back to the allocating box so that
//! the `deleter` callback can free both the shape slice and the data buffer in
//! one shot.
//!
//! # Safety contract
//!
//! * `DLManagedTensor` is heap-allocated via `Box::into_raw`.
//! * The `deleter` callback reconstructs the box and drops it, freeing both
//!   the data and the shape vector.
//! * The capsule destructor calls `deleter` when the capsule is garbage-
//!   collected.  This is the only path that frees memory — we never double-free
//!   because the capsule takes exclusive ownership.
//! * All raw pointer operations are isolated inside this module and carefully
//!   annotated.

use std::ffi::c_void;
use std::ptr::NonNull;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

// ---------------------------------------------------------------------------
// DLPack v0.8 C structs (reproduced inline; no C dependency)
// ---------------------------------------------------------------------------

/// Device type codes (only kCPU = 1 is used here).
const DEVICE_CPU: i32 = 1;

/// Data type codes.
const DTYPE_FLOAT: u8 = 2;

/// f32 has 32 bits.
const F32_BITS: u8 = 32;

/// Standard dense tensor has lanes = 1.
const LANES_DENSE: u16 = 1;

/// The DLPack capsule name as required by the protocol (NUL-terminated).
/// Must be `'static` and valid for the lifetime of any capsule we produce.
static CAPSULE_NAME: &std::ffi::CStr = c"dltensor";

/// A DLPack device descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDevice {
    /// Device type: 1 = kCPU.
    device_type: i32,
    /// Device ordinal (0 for the only CPU).
    device_id: i32,
}

/// Data type descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDataType {
    /// Type code: 2 = float.
    code: u8,
    /// Number of bits: 32 for f32.
    bits: u8,
    /// Number of lanes (SIMD width); always 1 for scalar tensors.
    lanes: u16,
}

/// The actual tensor descriptor.
#[repr(C)]
pub struct DLTensor {
    /// Pointer to the data buffer (unowned by this struct).
    data: *mut c_void,
    /// Device the tensor lives on.
    device: DLDevice,
    /// Number of dimensions.
    ndim: i32,
    /// Element type.
    dtype: DLDataType,
    /// Shape array (`ndim` elements, heap-allocated, owned by the manager).
    shape: *mut i64,
    /// Strides array, or NULL for C-contiguous (row-major) layout.
    strides: *mut i64,
    /// Byte offset into `data`.  Always 0 for freshly created tensors.
    byte_offset: u64,
}

/// The managed tensor wrapper that owns the memory.
#[repr(C)]
pub struct DLManagedTensor {
    /// The tensor descriptor.
    dl_tensor: DLTensor,
    /// Opaque context pointer passed to `deleter`.  We store a
    /// `*mut ManagedTensorState` here.
    manager_ctx: *mut c_void,
    /// Called by the consumer (or capsule destructor) to free the tensor.
    /// Must accept `NULL` gracefully.
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

// ---------------------------------------------------------------------------
// Owned state bundled with the managed tensor
// ---------------------------------------------------------------------------

/// All heap allocations owned by one `DLManagedTensor` instance.
///
/// Placed behind a raw pointer stored in `manager_ctx`.  The `deleter`
/// reconstructs the `Box` and drops it, releasing all memory.
struct ManagedTensorState {
    /// The original data buffer.
    data: Vec<f32>,
    /// The shape array.
    shape: Vec<i64>,
}

/// Deleter callback registered on every `DLManagedTensor` we produce.
///
/// # Safety
///
/// `managed` must be a valid `*mut DLManagedTensor` produced by
/// `Box::into_raw(Box::new(DLManagedTensor {...}))`.  The function is called
/// exactly once — either by the consumer (after copying the data) or by the
/// `PyCapsule` destructor when the capsule is garbage-collected.
unsafe extern "C" fn managed_tensor_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    // SAFETY: managed is non-null and was produced by Box::into_raw.
    let managed_ref = unsafe { &*managed };
    if !managed_ref.manager_ctx.is_null() {
        // SAFETY: manager_ctx points to a Box<ManagedTensorState> we allocated.
        drop(unsafe { Box::from_raw(managed_ref.manager_ctx as *mut ManagedTensorState) });
    }
    // Drop the DLManagedTensor box itself.
    drop(unsafe { Box::from_raw(managed) });
}

/// PyCapsule destructor that delegates to `managed_tensor_deleter`.
///
/// Called by CPython when the capsule object is garbage-collected.
///
/// # Safety
///
/// `capsule` must be a valid `*mut PyObject` whose pointer was set to a
/// `*mut DLManagedTensor` created by `vec_to_dlpack`.
unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    if capsule.is_null() {
        return;
    }
    // SAFETY: `capsule` is a valid PyObject wrapping our DLManagedTensor pointer.
    let ptr = unsafe { ffi::PyCapsule_GetPointer(capsule, CAPSULE_NAME.as_ptr()) };
    if !ptr.is_null() {
        // SAFETY: ptr was set to a Box<DLManagedTensor> in vec_to_dlpack.
        let managed = ptr as *mut DLManagedTensor;
        unsafe { managed_tensor_deleter(managed) };
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Convert a `Vec<f32>` plus a shape into a Python DLPack capsule.
///
/// The returned object is a `PyCapsule` with name `"dltensor"` containing a
/// heap-allocated `DLManagedTensor`.  The capsule takes ownership of `data`
/// and `shape`; all memory is freed when the capsule is garbage-collected (or
/// when the consumer calls the `deleter`).
///
/// # Arguments
///
/// * `py`    – Active Python interpreter token.
/// * `data`  – The float32 data buffer.  Length must equal `shape.iter().product()`.
/// * `shape` – Tensor dimensions (row-major).  May be empty for scalars (ndim=0).
///
/// # Errors
///
/// Returns `PyErr` if the data length does not match the product of the shape,
/// or if the capsule cannot be created.
pub fn vec_to_dlpack(py: Python<'_>, data: Vec<f32>, shape: Vec<i64>) -> PyResult<Py<PyCapsule>> {
    // Validate that data length matches the shape.
    let expected_len: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().map(|&d| d as usize).product()
    };
    if data.len() != expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "data length {} does not match shape product {} (shape={:?})",
            data.len(),
            expected_len,
            shape
        )));
    }

    // Move both data and shape into a state struct so that `deleter` can free them.
    let mut state = Box::new(ManagedTensorState { data, shape });

    // Build the DLTensor pointing into the state's Vecs.
    let data_ptr = state.data.as_mut_ptr() as *mut c_void;
    let shape_ptr = state.shape.as_mut_ptr();
    let ndim = state.shape.len() as i32;

    // Leak the state box; it will be reclaimed by `managed_tensor_deleter`.
    let state_raw = Box::into_raw(state);

    let dl_tensor = DLTensor {
        data: data_ptr,
        device: DLDevice {
            device_type: DEVICE_CPU,
            device_id: 0,
        },
        ndim,
        dtype: DLDataType {
            code: DTYPE_FLOAT,
            bits: F32_BITS,
            lanes: LANES_DENSE,
        },
        shape: shape_ptr,
        strides: std::ptr::null_mut(), // C-contiguous; strides are implicit
        byte_offset: 0,
    };

    let managed = Box::new(DLManagedTensor {
        dl_tensor,
        manager_ctx: state_raw as *mut c_void,
        deleter: Some(managed_tensor_deleter),
    });

    // Leak the managed tensor; the capsule destructor owns it.
    let managed_raw = Box::into_raw(managed);

    // SAFETY:
    // - `managed_raw` is a valid non-null pointer to a heap-allocated `DLManagedTensor`.
    // - `CAPSULE_NAME` is a valid `'static` NUL-terminated C string.
    // - `capsule_destructor` is a safe-to-call-from-any-thread `extern "C"` function.
    let non_null_ptr = NonNull::new(managed_raw as *mut c_void)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("null DLManagedTensor pointer"))?;

    let capsule = unsafe {
        PyCapsule::new_with_pointer_and_destructor(
            py,
            non_null_ptr,
            CAPSULE_NAME,
            Some(capsule_destructor),
        )
    }?;

    Ok(capsule.unbind())
}

/// Extract a `Vec<f32>` from a DLPack capsule.
///
/// The function validates that:
/// - The device type is kCPU (1).
/// - The dtype is float32 (code=2, bits=32, lanes=1).
///
/// The data is **copied** out of the capsule (the capsule retains ownership).
///
/// # Errors
///
/// Returns `PyErr` if any validation check fails.
pub fn dlpack_to_vec(_py: Python<'_>, capsule: &Bound<'_, PyCapsule>) -> PyResult<Vec<f32>> {
    // SAFETY: we borrow the pointer from the capsule; the capsule remains
    // alive for the duration of this function.
    let managed_ptr = unsafe { ffi::PyCapsule_GetPointer(capsule.as_ptr(), CAPSULE_NAME.as_ptr()) }
        as *const DLManagedTensor;
    if managed_ptr.is_null() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "DLPack capsule contains a null pointer",
        ));
    }

    // SAFETY: pointer is non-null and points to a DLManagedTensor we (or a
    // compatible producer) created following the DLPack v0.8 protocol.
    let managed = unsafe { &*managed_ptr };
    let tensor = &managed.dl_tensor;

    // Validate device.
    if tensor.device.device_type != DEVICE_CPU {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "DLPack tensor is on device type {} (expected CPU=1)",
            tensor.device.device_type
        )));
    }

    // Validate dtype.
    if tensor.dtype.code != DTYPE_FLOAT
        || tensor.dtype.bits != F32_BITS
        || tensor.dtype.lanes != LANES_DENSE
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "DLPack tensor dtype is not float32 (got code={}, bits={}, lanes={})",
            tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes
        )));
    }

    // Compute total number of elements from shape.
    let ndim = tensor.ndim as usize;
    let total_elements: usize = if ndim == 0 {
        1
    } else {
        // SAFETY: shape array has `ndim` elements.
        let shape_slice = unsafe { std::slice::from_raw_parts(tensor.shape, ndim) };
        shape_slice.iter().map(|&d| d as usize).product()
    };

    // SAFETY: data pointer is valid and points to `total_elements` f32 values.
    let data_ptr =
        (tensor.data as *const u8).wrapping_add(tensor.byte_offset as usize) as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(data_ptr, total_elements) };

    Ok(slice.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that a `DLManagedTensor` built with `vec_to_dlpack` internals
    /// stores the correct shape values.
    ///
    /// We test the inner allocation directly (without a Python interpreter)
    /// so that this test runs in pure Rust via `cargo nextest`.
    #[test]
    fn dlpack_shape_matches_input() {
        // Simulate what `vec_to_dlpack` does internally.
        let shape: Vec<i64> = vec![3, 4];
        let data: Vec<f32> = vec![1.0_f32; 12]; // 3×4 = 12

        let mut state = Box::new(ManagedTensorState {
            data: data.clone(),
            shape: shape.clone(),
        });

        let shape_ptr = state.shape.as_mut_ptr();
        let ndim = state.shape.len() as i32;

        // Keep state alive while we read shape.
        let state_raw = Box::into_raw(state);

        // Verify shape values via pointer arithmetic (mirrors DLTensor access).
        // SAFETY: state_raw is valid and shape_ptr points into its shape Vec.
        assert_eq!(ndim, 2, "ndim must be 2");
        unsafe {
            assert_eq!(*shape_ptr, 3, "shape[0] must be 3");
            assert_eq!(*shape_ptr.add(1), 4, "shape[1] must be 4");
        }

        // Clean up.
        // SAFETY: state_raw was produced by Box::into_raw above.
        let _ = unsafe { Box::from_raw(state_raw) };
        let _ = data; // silence unused variable
    }

    /// Verify that `DLDataType` for f32 has the correct field values.
    #[test]
    fn dlpack_dtype_is_f32() {
        let dtype = DLDataType {
            code: DTYPE_FLOAT,
            bits: F32_BITS,
            lanes: LANES_DENSE,
        };
        assert_eq!(dtype.code, 2, "dtype code must be 2 (float)");
        assert_eq!(dtype.bits, 32, "dtype bits must be 32 (f32)");
        assert_eq!(dtype.lanes, 1, "dtype lanes must be 1 (scalar)");
    }

    /// Verify that `DLDevice` for CPU has the correct field values.
    #[test]
    fn dlpack_device_is_cpu() {
        let device = DLDevice {
            device_type: DEVICE_CPU,
            device_id: 0,
        };
        assert_eq!(device.device_type, 1, "device_type must be 1 (kCPU)");
        assert_eq!(device.device_id, 0, "device_id must be 0 for single CPU");
    }

    /// Verify the capsule name is exactly "dltensor".
    #[test]
    fn dlpack_capsule_name_is_dltensor() {
        assert_eq!(
            CAPSULE_NAME
                .to_str()
                .expect("capsule name must be valid UTF-8"),
            "dltensor"
        );
    }

    /// Verify that `managed_tensor_deleter` does not panic on a null pointer.
    #[test]
    fn dlpack_deleter_null_is_safe() {
        // SAFETY: passing null must not dereference or panic.
        unsafe { managed_tensor_deleter(std::ptr::null_mut()) };
    }
}
