//! Optional AIR-declared buffer reach and access, independent of native PSO reflection.
//!
//! Only `BufferExtent::Object` from the existing metadata parser may reduce
//! capture. Pointee sizes, native argument sizes and buffer names decide nothing.
//! Both successful reflection and failure are cached by exact MTLB bytes/stage;
//! a missing tool cannot launch another process on every draw.

use super::input::Class;
use crate::backend::blob::{BlobIdentity, BlobKey};
use crate::backend::hash::hash_u64;
use crate::model::content_cache::{CacheEntry, ContentCache};
use crate::observe::{Decline, Emit};
use metal2vulkan::reflect::{ShaderReflection, ShaderStage};
use parking_lot::Mutex;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    Vertex,
    Fragment,
}

impl Stage {
    fn tag(self) -> u64 {
        match self {
            Self::Vertex => 0,
            Self::Fragment => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }
}

/// A particular shader stage and slot's reach and access, never a native length.
#[derive(Clone, Copy)]
pub(crate) struct BufferRead {
    stage: Stage,
    index: u32,
    bytes: Option<u64>,
    read_only: bool,
}

/// Issued only for a binding whose reflected shader access cannot write.
/// The immutable shader/stage/slot association licenses immutable reuse.
/// Absent reach still captures the complete suffix.
#[derive(Clone, Copy)]
pub(crate) struct ReadOnlyCapture {
    bytes: Option<u64>,
}

impl ReadOnlyCapture {
    pub(crate) fn bytes(self) -> Option<u64> {
        self.bytes
    }
}

impl BufferRead {
    pub(crate) fn capture_for(
        self,
        class: Class,
        index: u32,
    ) -> Option<(Option<u64>, Option<ReadOnlyCapture>)> {
        self.matches(class, index).then_some(())?;
        let bytes = self.bytes;
        Some((bytes, self.read_only.then_some(ReadOnlyCapture { bytes })))
    }

    #[cfg(test)]
    pub(crate) fn bytes_for(self, class: Class, index: u32) -> Option<u64> {
        self.matches(class, index).then_some(self.bytes).flatten()
    }

    fn matches(self, class: Class, index: u32) -> bool {
        if self.index == index
            && matches!(
                (self.stage, class),
                (Stage::Vertex, Class::Vertex) | (Stage::Fragment, Class::Fragment)
            )
        {
            true
        } else {
            Emit::refusal(
                "metal_buffer_extent",
                &super::util::Status::args("metal_buffer_extent_binding_mismatch"),
            )
            .unwrap()
            .field("stage", self.stage.name())
            .field("class", format!("{class:?}"))
            .field("proof_index", self.index)
            .field("binding", index)
            .fail();
            false
        }
    }
}

pub(crate) struct BufferExtents {
    stage: Stage,
    reflection: Result<ShaderReflection, String>,
}

impl BufferExtents {
    pub(crate) fn bound(&self, index: u32) -> Option<BufferRead> {
        let Ok(reflection) = &self.reflection else {
            crate::runtime::drain::note_store_route("metal_buffer_extent_reflection_failed");
            return None;
        };
        let bytes = crate::runtime::spirv_bind::reflected_buffer_extent(reflection, index);
        // A zero-size declaration need not turn an otherwise valid full bind
        // into an empty guest read; retain the conservative existing path.
        if bytes == Some(0) {
            crate::runtime::drain::note_store_route("metal_buffer_extent_zero_object");
            return None;
        }
        // Missing slots must not acquire an Unused certificate by default.
        if !reflection.bindings.iter().any(|binding| {
            binding.kind == metal2vulkan::reflect::ResourceKind::Buffer
                && binding.metal_index == index
        }) {
            return None;
        }
        let read_only = matches!(
            crate::runtime::spirv_bind::reflected_buffer_access(reflection, index),
            crate::runtime::spirv_bind::ReflectedBufferAccess::ReadOnly
                | crate::runtime::spirv_bind::ReflectedBufferAccess::Unused
        );
        if bytes.is_none() && !read_only {
            return None;
        }
        Some(BufferRead {
            stage: self.stage,
            index,
            bytes,
            read_only,
        })
    }
}

struct Entry {
    blob: BlobIdentity,
    extents: Arc<BufferExtents>,
}

impl CacheEntry for Entry {
    type Key<'a> = (BlobKey<'a>, Stage);

    fn lookup_key(&self) -> Self::Key<'_> {
        (self.blob.as_key(), self.extents.stage)
    }

    fn matches(&self, key: &Self::Key<'_>) -> bool {
        self.extents.stage == key.1 && self.blob.is(&key.0)
    }

    fn bucket(key: &Self::Key<'_>) -> u64 {
        hash_u64(key.0.hash, key.1.tag())
    }
}

static CACHE: Mutex<ContentCache<Entry>> = Mutex::new(ContentCache::new());
static BUILD: Mutex<()> = Mutex::new(());

struct ReflectionFailure<'a>(&'a str);

impl Decline for ReflectionFailure<'_> {
    fn slug(&self) -> &'static str {
        "metal_buffer_extent_reflection_failed"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("detail", self.0.replace(char::is_whitespace, "_"))]
    }
}

fn cached_with(
    cache: &Mutex<ContentCache<Entry>>,
    key: BlobKey<'_>,
    stage: Stage,
    build: impl FnOnce() -> Result<ShaderReflection, String>,
) -> Arc<BufferExtents> {
    if let Some(entry) = cache.lock().find(&(key, stage)) {
        return entry.extents.clone();
    }
    let _build = BUILD.lock();
    if let Some(entry) = cache.lock().find(&(key, stage)) {
        return entry.extents.clone();
    }
    crate::runtime::drain::note_store_route("metal_buffer_extent_reflection_build");
    let reflection = build();
    if let Err(detail) = &reflection {
        Emit::decline("metal_buffer_extent", &ReflectionFailure(detail))
            .field("stage", stage.name())
            .field("hash", key.hash)
            .fail();
    }
    cache
        .lock()
        .insert_unique(Entry {
            blob: BlobIdentity::of(&key),
            extents: Arc::new(BufferExtents { stage, reflection }),
        })
        .extents
        .clone()
}

pub(crate) fn cached(key: BlobKey<'_>, stage: Stage) -> Arc<BufferExtents> {
    cached_with(&CACHE, key, stage, || {
        std::panic::catch_unwind(|| reflect_mtlb(key.bytes, stage))
            .unwrap_or_else(|_| Err("metadata parser panicked".into()))
    })
}

struct Scratch(PathBuf);

impl Scratch {
    fn write(bytes: &[u8], extension: &str) -> Result<Self, String> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        std::fs::create_dir_all(".cache").map_err(|error| error.to_string())?;
        let path = PathBuf::from(format!(
            ".cache/reims-metal-metadata-{}-{}.{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            extension,
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .map_err(|error| error.to_string())?;
        let scratch = Self(path);
        file.write_all(bytes).map_err(|error| error.to_string())?;
        Ok(scratch)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_file(&self.0) {
            if error.kind() != std::io::ErrorKind::NotFound {
                Emit::refusal(
                    "metal_buffer_extent",
                    &super::util::Status::execute("metal_buffer_extent_scratch_cleanup_failed"),
                )
                .unwrap()
                .field("path", self.0.display())
                .field("error", error)
                .fail();
            }
        }
    }
}

fn reflect_mtlb(mtlb: &[u8], stage: Stage) -> Result<ShaderReflection, String> {
    let air = crate::runtime::mtlb::extract_air(mtlb).map_err(|error| error.to_string())?;
    let scratch = Scratch::write(air, "air")?;
    // The existing bounded tool runner owns discovery, timeout and pipe drain.
    // Only disassembly occurs, once on a miss; no SPIR-V or executable lowering.
    let (ll, _) = metal2vulkan::tools::run_with_timeout(
        "llvm-dis",
        &[
            scratch.0.to_str().ok_or("non-UTF8 scratch path")?,
            "-o",
            "-",
        ],
        20,
    )?;
    let ll = String::from_utf8(ll).map_err(|error| error.to_string())?;
    reflect_text(&ll, stage)
}

fn reflect_text(ll: &str, stage: Stage) -> Result<ShaderReflection, String> {
    let (sanitized, _) = metal2vulkan::tools::sanitize_ll_text_with_datalayout(ll);
    // This public metadata facade constructs no SPIR-V. Reach is the AIR
    // declaration, not its optional access classification or an emitted footprint.
    let reflection = metal2vulkan::reflect_sanitized(
        &sanitized,
        match stage {
            Stage::Vertex => metal2vulkan::passes::Stage::Vertex,
            Stage::Fragment => metal2vulkan::passes::Stage::Fragment,
        },
        Default::default(),
    )?;
    let expected = match stage {
        Stage::Vertex => ShaderStage::Vertex,
        Stage::Fragment => ShaderStage::Fragment,
    };
    if reflection.stage != expected || reflection.entry_point.is_none() {
        return Err("requested AIR stage entry is absent".into());
    }
    reflection.validate_descriptor_abi()?;
    // The reflection owns metadata, not the executable IR or tool output.
    Ok(reflection)
}

#[cfg(test)]
pub(crate) mod tests;
