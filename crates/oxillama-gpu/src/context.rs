//! GPU device and queue initialisation.
//!
//! [`GpuContext`] holds an initialised wgpu device+queue pair.  When the
//! `gpu` feature is disabled the struct is zero-size and `try_init` always
//! returns `None`, so all call-sites compile without GPU hardware or the
//! feature flag.

/// Information about an available GPU device.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Human-readable device name.
    pub name: String,
    /// Backend type (Vulkan, Metal, DX12, etc.)
    pub backend: String,
    /// Device type (discrete, integrated, software, etc.)
    pub device_type: String,
}

/// An initialised GPU device and queue.
///
/// Construct via [`GpuContext::try_init`].  Returns `None` if no compatible
/// adapter is available (headless CI, no GPU hardware, feature disabled).
///
/// The `_private` field is always present (unconditionally) so that external
/// code cannot construct a `GpuContext` with struct-literal syntax even when
/// the `gpu` feature is disabled (which would otherwise leave an empty struct
/// that can be trivially constructed).
pub struct GpuContext {
    #[cfg(feature = "gpu")]
    pub(crate) device: wgpu::Device,
    #[cfg(feature = "gpu")]
    pub(crate) queue: wgpu::Queue,
    /// Prevents external struct-literal construction.
    _private: (),
}

impl GpuContext {
    /// Try to initialise a GPU context.
    ///
    /// Returns `None` when:
    /// - The `gpu` feature is not enabled.
    /// - No compatible wgpu adapter exists on the current host.
    /// - The device-request step fails (e.g. out-of-resources).
    pub fn try_init() -> Option<Self> {
        #[cfg(feature = "gpu")]
        {
            pollster::block_on(Self::try_init_async())
        }
        #[cfg(not(feature = "gpu"))]
        {
            None
        }
    }

    /// Async GPU initialisation used by `try_init`.
    #[cfg(feature = "gpu")]
    async fn try_init_async() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()?;

        Some(GpuContext {
            device,
            queue,
            _private: (),
        })
    }

    /// Enumerate available GPU adapters and return info about each.
    pub fn enumerate_devices() -> Vec<GpuDeviceInfo> {
        #[cfg(feature = "gpu")]
        {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });

            pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
                .into_iter()
                .map(|adapter| {
                    let info = adapter.get_info();
                    GpuDeviceInfo {
                        name: info.name,
                        backend: format!("{:?}", info.backend),
                        device_type: format!("{:?}", info.device_type),
                    }
                })
                .collect()
        }
        #[cfg(not(feature = "gpu"))]
        {
            Vec::new()
        }
    }

    /// Try to initialise with a specific adapter selected by name substring
    /// match (case-insensitive).
    pub fn try_init_with_name(name_pattern: &str) -> Option<Self> {
        #[cfg(feature = "gpu")]
        {
            pollster::block_on(Self::try_init_with_name_async(name_pattern))
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = name_pattern;
            None
        }
    }

    /// Async helper for `try_init_with_name`.
    #[cfg(feature = "gpu")]
    async fn try_init_with_name_async(name_pattern: &str) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let pattern_lower = name_pattern.to_lowercase();
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .await
            .into_iter()
            .find(|a| a.get_info().name.to_lowercase().contains(&pattern_lower))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()?;

        Some(GpuContext {
            device,
            queue,
            _private: (),
        })
    }

    /// Try to initialise with a specific adapter by index.
    pub fn try_init_with_index(index: usize) -> Option<Self> {
        #[cfg(feature = "gpu")]
        {
            pollster::block_on(Self::try_init_with_index_async(index))
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = index;
            None
        }
    }

    /// Async helper for `try_init_with_index`.
    #[cfg(feature = "gpu")]
    async fn try_init_with_index_async(index: usize) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapters: Vec<_> = instance
            .enumerate_adapters(wgpu::Backends::all())
            .await
            .into_iter()
            .collect();

        let adapter = adapters.into_iter().nth(index)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()?;

        Some(GpuContext {
            device,
            queue,
            _private: (),
        })
    }
}
