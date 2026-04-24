//! Pooled KV-cache page allocator.
//!
//! Provides a free-list based page pool so that KV cache memory can be
//! recycled across requests without returning it to the allocator.
//! Pages are fixed-size slabs of `f32` values.  The pool never frees pages
//! until it is itself dropped.

/// A pool of KV-cache pages.
///
/// Pages are fixed-size slabs of `f32` data backed by a single `Vec` per page.
/// The pool uses a simple free-list: allocated pages are handed out by
/// returning their index, and freed pages are pushed back onto the list.
///
/// The pool never shrinks — once a page is allocated it lives until the pool
/// is dropped.
pub struct KvCachePool {
    /// All allocated pages, indexed by page index.
    pages: Vec<Box<[f32]>>,
    /// Indices of pages currently not in use.
    free_list: Vec<usize>,
    /// Number of `f32` elements per page.
    page_size: usize,
}

impl KvCachePool {
    /// Create a new pool with `initial_pages` pre-allocated pages of
    /// `page_size` `f32` elements each.
    ///
    /// All pages start on the free list, ready for immediate allocation.
    pub fn new(page_size: usize, initial_pages: usize) -> Self {
        let mut pages = Vec::with_capacity(initial_pages);
        let mut free_list = Vec::with_capacity(initial_pages);

        for i in 0..initial_pages {
            pages.push(vec![0.0f32; page_size].into_boxed_slice());
            free_list.push(i);
        }

        Self {
            pages,
            free_list,
            page_size,
        }
    }

    /// Allocate a page from the free list.
    ///
    /// Returns `Some(page_idx)` on success, or `None` if the pool is
    /// exhausted.  The caller must pass the returned index to [`free`] when
    /// the page is no longer needed.
    ///
    /// [`free`]: KvCachePool::free
    pub fn alloc(&mut self) -> Option<usize> {
        self.free_list.pop()
    }

    /// Return page `page_idx` to the free list.
    ///
    /// The page data is **not** zeroed; callers should zero or overwrite the
    /// slice before treating it as a fresh allocation.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `page_idx >= total_pages()`.
    pub fn free(&mut self, page_idx: usize) {
        debug_assert!(
            page_idx < self.pages.len(),
            "KvCachePool::free: page_idx {page_idx} out of range (total {})",
            self.pages.len()
        );
        self.free_list.push(page_idx);
    }

    /// Get an immutable slice to page `page_idx`.
    ///
    /// # Panics
    ///
    /// Panics if `page_idx >= total_pages()`.
    pub fn page(&self, page_idx: usize) -> &[f32] {
        &self.pages[page_idx]
    }

    /// Get a mutable slice to page `page_idx`.
    ///
    /// # Panics
    ///
    /// Panics if `page_idx >= total_pages()`.
    pub fn page_mut(&mut self, page_idx: usize) -> &mut [f32] {
        &mut self.pages[page_idx]
    }

    /// Total number of pages ever allocated (including those on the free list).
    pub fn total_pages(&self) -> usize {
        self.pages.len()
    }

    /// Number of pages currently on the free list (available for allocation).
    pub fn free_pages(&self) -> usize {
        self.free_list.len()
    }

    /// Number of pages currently in use (allocated but not yet freed).
    pub fn used_pages(&self) -> usize {
        self.pages.len() - self.free_list.len()
    }

    /// The number of `f32` elements in each page.
    pub fn page_size(&self) -> usize {
        self.page_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_free() {
        let mut pool = KvCachePool::new(64, 4);
        assert_eq!(pool.total_pages(), 4);
        assert_eq!(pool.free_pages(), 4);
        assert_eq!(pool.used_pages(), 0);

        let idx0 = pool.alloc().expect("should allocate");
        assert_eq!(pool.used_pages(), 1);
        assert_eq!(pool.free_pages(), 3);

        let idx1 = pool.alloc().expect("should allocate");
        assert_ne!(idx0, idx1);

        pool.free(idx0);
        assert_eq!(pool.used_pages(), 1);
        assert_eq!(pool.free_pages(), 3);
    }

    #[test]
    fn test_pool_exhaustion_returns_none() {
        let mut pool = KvCachePool::new(16, 2);
        let _a = pool.alloc().expect("first alloc");
        let _b = pool.alloc().expect("second alloc");
        // Pool is now exhausted
        assert!(pool.alloc().is_none());
    }

    #[test]
    fn test_free_then_realloc() {
        let mut pool = KvCachePool::new(8, 1);
        let idx = pool.alloc().expect("alloc");
        pool.free(idx);
        // Page should be back
        let idx2 = pool.alloc().expect("re-alloc after free");
        assert_eq!(idx, idx2);
    }

    #[test]
    fn test_page_read_write() {
        let mut pool = KvCachePool::new(4, 2);
        let idx = pool.alloc().expect("alloc");
        {
            let page = pool.page_mut(idx);
            page[0] = 1.0;
            page[1] = 2.5;
        }
        let page = pool.page(idx);
        assert!((page[0] - 1.0).abs() < 1e-9);
        assert!((page[1] - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_page_size_accessor() {
        let pool = KvCachePool::new(128, 0);
        assert_eq!(pool.page_size(), 128);
        assert_eq!(pool.total_pages(), 0);
        assert_eq!(pool.free_pages(), 0);
    }
}
