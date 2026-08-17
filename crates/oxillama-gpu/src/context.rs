//! GPU device and queue initialisation.
//!
//! [`GpuContext`] holds an initialised wgpu device+queue pair.  When the
//! `gpu` feature is disabled the struct is zero-size and `try_init` always
//! returns `None`, so all call-sites compile without GPU hardware or the
//! feature flag.
//!
//! It also owns a per-context **compute pipeline cache**
//! ([`GpuContext::get_or_create_pipeline`]).  Every `gpu_gemv_*` function in
//! `kernels/*.rs` previously called `create_shader_module` →
//! `create_bind_group_layout` → `create_pipeline_layout` →
//! `create_compute_pipeline` on *every* invocation — i.e. on every token
//! during decode.  The cache makes pipeline construction pay-once-per-process
//! instead of once-per-token; see `kernels/q4_0_resident.rs` for the first
//! kernel wired through it end-to-end (device-resident quantised weights,
//! in-shader dequantisation, cached pipeline).

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

/// A cached compute pipeline plus the bind-group layout used to build its
/// bind groups (bind groups themselves are cheap and NOT cached — they
/// reference per-call buffers).
#[cfg(feature = "gpu")]
pub struct CachedPipeline {
    /// Bind-group layout matching this pipeline's shader bindings.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// The compiled compute pipeline.
    pub pipeline: wgpu::ComputePipeline,
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
    /// Compute-pipeline cache, keyed by a stable per-kernel-variant name.
    ///
    /// None of this crate's GEMV shaders bake tensor shape (rows/cols) into
    /// the compiled pipeline — shape is read at dispatch time from a uniform
    /// buffer — so the cache key is just the kernel's name.  A kernel whose
    /// shader *does* need shape-specialised code (e.g. a workgroup size
    /// chosen per `cols`) should fold that into the name, e.g.
    /// `format!("q4_0-dequant-wg{workgroup_size}")`, which is why the key
    /// type is a plain string rather than a fixed `(name, shape)` tuple.
    #[cfg(feature = "gpu")]
    pipeline_cache:
        std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<CachedPipeline>>>,
    /// Identity of the adapter this context is bound to, captured once at
    /// initialisation.  A context is permanently bound to one adapter, so this
    /// never changes for the lifetime of the context.
    #[cfg(feature = "gpu")]
    device_info: GpuDeviceInfo,
    /// Prevents external struct-literal construction.
    _private: (),
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Get the cached pipeline for `name`, building it with `build` on first
    /// use.  Subsequent calls with the same `name` on the same context are
    /// `O(1)` `HashMap` lookups and perform no shader compilation.
    ///
    /// `build` receives the device and must return the bind-group layout and
    /// compute pipeline for this kernel variant.
    pub fn get_or_create_pipeline(
        &self,
        name: &str,
        build: impl FnOnce(&wgpu::Device) -> CachedPipeline,
    ) -> std::sync::Arc<CachedPipeline> {
        if let Some(cached) = self
            .pipeline_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(name)
        {
            return std::sync::Arc::clone(cached);
        }
        let built = std::sync::Arc::new(build(&self.device));
        self.pipeline_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(name.to_owned(), std::sync::Arc::clone(&built));
        built
    }

    /// Number of distinct pipelines currently cached. Test/diagnostic use.
    pub fn cached_pipeline_count(&self) -> usize {
        self.pipeline_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Identity of the adapter this context is bound to.
    ///
    /// The returned info is captured at initialisation and is stable for the
    /// lifetime of the context; it describes the adapter actually bound, which
    /// for [`GpuContext::try_init_with_name`] / [`GpuContext::try_init_with_index`]
    /// is the selected one and not necessarily the host's default adapter.
    ///
    /// Gated on the `gpu` feature because without it no `GpuContext` value can
    /// exist (every constructor returns `None` and `_private` blocks struct
    /// literals), so an ungated accessor would be uncallable.
    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    /// Request a device+queue from `adapter` and wrap them in a context.
    ///
    /// Sole construction site for `GpuContext`; every init path funnels through
    /// here so that `device_info` can never disagree with the bound adapter.
    /// Returns `None` if the device request fails (e.g. out-of-resources).
    async fn from_adapter(adapter: wgpu::Adapter) -> Option<Self> {
        let info = adapter.get_info();
        let device_info = GpuDeviceInfo {
            name: info.name,
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()?;

        Some(GpuContext {
            device,
            queue,
            pipeline_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            device_info,
            _private: (),
        })
    }
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
                // Limit bucketing is a fingerprinting mitigation for untrusted
                // content; it only coarsens the reported adapter limits, so a
                // local inference engine wants the real (larger) limits.
                apply_limit_buckets: false,
            })
            .await
            .ok()?;

        Self::from_adapter(adapter).await
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

        Self::from_adapter(adapter).await
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

        Self::from_adapter(adapter).await
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::GpuContext;

    #[test]
    fn device_info_is_populated_when_context_initialises() {
        let Some(ctx) = GpuContext::try_init() else {
            return;
        };
        let info = ctx.device_info();
        assert!(!info.name.is_empty(), "adapter name must not be empty");
        assert!(!info.backend.is_empty(), "backend must not be empty");
        assert!(
            !info.device_type.is_empty(),
            "device type must not be empty"
        );
    }

    #[test]
    fn name_selected_context_reports_matching_adapter() {
        let devices = GpuContext::enumerate_devices();
        let Some(first) = devices.first() else {
            return;
        };
        // Use a prefix of a real adapter name so the pattern is guaranteed to
        // match at least one adapter; `find` may still legitimately settle on a
        // different adapter, hence the substring (not equality) assertion.
        let pattern: String = first.name.chars().take(4).collect();
        if pattern.is_empty() {
            return;
        }
        let Some(ctx) = GpuContext::try_init_with_name(&pattern) else {
            return;
        };
        assert!(
            ctx.device_info()
                .name
                .to_lowercase()
                .contains(&pattern.to_lowercase()),
            "selected adapter {:?} does not match pattern {:?}",
            ctx.device_info().name,
            pattern
        );
    }
}
