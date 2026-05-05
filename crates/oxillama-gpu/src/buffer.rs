//! GPU buffer helpers — upload and download f32 arrays.
//!
//! These helpers abstract the common pattern of:
//! 1. Uploading a `&[f32]` to a wgpu storage buffer (for shader reads/writes).
//! 2. Downloading f32 data from the GPU back to a `Vec<f32>` via a staging
//!    buffer + map_async.
//!
//! All functions are gated behind `#[cfg(feature = "gpu")]`.  The module
//! itself is always compiled so that call-sites remain syntactically valid.

#[cfg(feature = "gpu")]
use crate::error::{GpuError, GpuResult};

/// Upload a `&[f32]` slice to a GPU storage buffer (STORAGE | COPY_SRC).
///
/// The buffer is suitable for use as a read-only shader storage binding.
#[cfg(feature = "gpu")]
pub(crate) fn upload_f32(device: &wgpu::Device, label: &str, data: &[f32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

/// Create an empty, writable GPU storage buffer of `len` f32 elements.
#[cfg(feature = "gpu")]
pub(crate) fn create_output_f32(device: &wgpu::Device, label: &str, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Create a uniform buffer from a `bytemuck::Pod` value.
#[cfg(feature = "gpu")]
pub(crate) fn upload_uniform<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    value: &T,
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Read back `len` f32 values from `src_buf` on the GPU.
///
/// Blocks until the GPU work submitted prior to this call completes.
/// Returns a `Vec<f32>` with the results.
#[cfg(feature = "gpu")]
pub(crate) fn download_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_buf: &wgpu::Buffer,
    len: usize,
) -> GpuResult<Vec<f32>> {
    let byte_len = (len * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-staging-readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(src_buf, 0, &staging, 0, byte_len);
    queue.submit([encoder.finish()]);

    // Map the staging buffer and wait for the GPU to finish.
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        // Ignore send errors — receiver may have already dropped if GPU failed.
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| GpuError::BufferMap {
            detail: format!("{e:?}"),
        })?;

    rx.recv()
        .map_err(|_| GpuError::BufferMap {
            detail: "channel closed before GPU mapped buffer".to_owned(),
        })?
        .map_err(|e| GpuError::BufferMap {
            detail: format!("{e:?}"),
        })?;

    let view = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();

    Ok(result)
}

/// Upload a `&[u32]` slice to a GPU storage buffer (STORAGE | COPY_SRC).
///
/// Suitable for read-only shader storage bindings of `array<u32>`.
#[cfg(feature = "gpu")]
pub(crate) fn upload_u32(device: &wgpu::Device, label: &str, data: &[u32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

/// Create an empty, writable GPU storage buffer of `len` u32 elements.
#[cfg(feature = "gpu")]
pub(crate) fn create_output_u32(device: &wgpu::Device, label: &str, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Read back `len` u32 values from `src_buf` on the GPU.
///
/// Blocks until the GPU work submitted prior to this call completes.
#[cfg(feature = "gpu")]
pub(crate) fn download_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_buf: &wgpu::Buffer,
    len: usize,
) -> GpuResult<Vec<u32>> {
    let byte_len = (len * std::mem::size_of::<u32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-staging-readback-u32"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback-u32"),
    });
    encoder.copy_buffer_to_buffer(src_buf, 0, &staging, 0, byte_len);
    queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| GpuError::BufferMap {
            detail: format!("{e:?}"),
        })?;

    rx.recv()
        .map_err(|_| GpuError::BufferMap {
            detail: "channel closed before GPU mapped u32 buffer".to_owned(),
        })?
        .map_err(|e| GpuError::BufferMap {
            detail: format!("{e:?}"),
        })?;

    let view = slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();

    Ok(result)
}

// ─── stub when the gpu feature is absent ─────────────────────────────────────
//
// No stubs needed here; the `buffer` module functions are only called from
// kernel code that is itself gated with `#[cfg(feature = "gpu")]`.
