//! Bounded native compilation hints, shared by exact shader-program families.
//!
//! A few KosmicKrisp programs already exceed the old shared cache's 8 MiB
//! aggregate limit. Discarding that aggregate made every boot cold. Families
//! share compiled shader work across PSO variants; the runtime's complete PSO
//! key, not this hint, remains the authority for returning a pipeline.

use super::{pipeline_cache_blob_compatible, DeviceContext, PIPELINE_CACHE_MAX_WARM_BYTES};
use ash::vk;
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

const MAGIC: &[u8; 8] = b"RVKPC02\0";
const HEADER_BYTES: usize = 32;
const MAX_IDENTITY_BYTES: usize = 4 * 1024 * 1024;
const MAX_PROGRAM_BYTES: usize = 64 * 1024 * 1024;
pub(super) const MAX_ENTRY_BYTES: usize = HEADER_BYTES + MAX_IDENTITY_BYTES + MAX_PROGRAM_BYTES;
// Reserve the remaining 8 MiB for the context's shared utility-pipeline cache.
const MAX_NATIVE_BYTES: u64 = 512 * 1024 * 1024 - PIPELINE_CACHE_MAX_WARM_BYTES as u64;
const MAX_NATIVE_ENTRIES: usize = 2048;

#[derive(Default)]
struct Encoding(Vec<u8>);

macro_rules! encode_integers {
    ($($method:ident($ty:ty)),* $(,)?) => {
        $(fn $method(&mut self, value: $ty) {
            self.write(&value.to_le_bytes());
        })*
    };
}

impl Hasher for Encoding {
    fn finish(&self) -> u64 {
        xxhash_rust::xxh3::xxh3_64(&self.0)
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    encode_integers!(
        write_u8(u8),
        write_u16(u16),
        write_u32(u32),
        write_u64(u64),
        write_u128(u128),
        write_i8(i8),
        write_i16(i16),
        write_i32(i32),
        write_i64(i64),
        write_i128(i128),
    );

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    fn write_isize(&mut self, value: isize) {
        self.write_i64(value as i64);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Key {
    identity: Vec<u8>,
    bucket: u128,
}

impl Key {
    /// Called only after the immutable PSO cache misses. Exact program bytes
    /// are retained in the envelope and compared, not just their filename hash.
    pub(crate) fn new<T: Hash>(kind: u8, descriptor: &T, shaders: &[&[u32]]) -> Self {
        let mut encoding = Encoding::default();
        encoding.write_u32(2);
        encoding.write_u8(kind);
        descriptor.hash(&mut encoding);
        encoding.write_usize(shaders.len());
        for shader in shaders {
            encoding.write_usize(shader.len());
            for word in *shader {
                encoding.write_u32(*word);
            }
        }
        let bucket = xxhash_rust::xxh3::xxh3_128(&encoding.0);
        Self {
            identity: encoding.0,
            bucket,
        }
    }

    fn path(&self, directory: &Path) -> PathBuf {
        directory.join(format!("{:032x}.bin", self.bucket))
    }
}

#[derive(Debug)]
pub(super) enum Refusal {
    Io {
        stage: &'static str,
        kind: std::io::ErrorKind,
    },
    Envelope,
    Identity,
    Checksum,
    Device,
    Limit {
        bytes: usize,
        cap: usize,
    },
    Queue {
        jobs: usize,
        bytes: usize,
    },
    Driver(vk::Result),
    Data(vk::Result),
}

impl crate::observe::Decline for Refusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::Io { .. } => "vk_pipeline_cache_entry_io",
            Self::Envelope => "vk_pipeline_cache_entry_envelope",
            Self::Identity => "vk_pipeline_cache_entry_identity",
            Self::Checksum => "vk_pipeline_cache_entry_checksum",
            Self::Device => "vk_pipeline_cache_entry_device",
            Self::Limit { .. } => "vk_pipeline_cache_entry_limit",
            Self::Queue { .. } => "vk_pipeline_cache_entry_queue_full",
            Self::Driver(_) => "vk_pipeline_cache_entry_create",
            Self::Data(_) => "vk_pipeline_cache_entry_data",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Io { stage, kind } => {
                vec![("stage", (*stage).into()), ("io_kind", format!("{kind:?}"))]
            }
            Self::Limit { bytes, cap } => {
                vec![("bytes", bytes.to_string()), ("cap", cap.to_string())]
            }
            Self::Queue { jobs, bytes } => {
                vec![("jobs", jobs.to_string()), ("bytes", bytes.to_string())]
            }
            Self::Driver(result) | Self::Data(result) => vec![("vk_result", format!("{result:?}"))],
            _ => Vec::new(),
        }
    }
}

fn io(stage: &'static str, error: std::io::Error) -> Refusal {
    Refusal::Io {
        stage,
        kind: error.kind(),
    }
}

pub(super) fn report(refusal: &Refusal) {
    crate::observe::Emit::decline("vk_pipeline_cache_entry", refusal).fail_once(0);
}

pub(super) fn report_queue_full(jobs: usize, bytes: usize) {
    report(&Refusal::Queue { jobs, bytes });
}

struct Blob {
    bytes: Vec<u8>,
    native_start: usize,
    native_checksum: u128,
}

impl Blob {
    fn native(&self) -> &[u8] {
        &self.bytes[self.native_start..]
    }
}

fn encode(key: &Key, native: &[u8]) -> Result<Vec<u8>, Refusal> {
    for (bytes, cap) in [
        (key.identity.len(), MAX_IDENTITY_BYTES),
        (native.len(), MAX_PROGRAM_BYTES),
    ] {
        if bytes > cap {
            return Err(Refusal::Limit { bytes, cap });
        }
    }
    let mut bytes = Vec::with_capacity(HEADER_BYTES + key.identity.len() + native.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(key.identity.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(native.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&xxhash_rust::xxh3::xxh3_128(native).to_le_bytes());
    bytes.extend_from_slice(&key.identity);
    bytes.extend_from_slice(native);
    Ok(bytes)
}

fn decode(
    bytes: Vec<u8>,
    key: &Key,
    props: &vk::PhysicalDeviceProperties,
) -> Result<Blob, Refusal> {
    if bytes.len() < HEADER_BYTES || &bytes[..8] != MAGIC {
        return Err(Refusal::Envelope);
    }
    let identity_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let native_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    if identity_len > MAX_IDENTITY_BYTES || native_len > MAX_PROGRAM_BYTES {
        return Err(Refusal::Limit {
            bytes: bytes.len(),
            cap: MAX_ENTRY_BYTES,
        });
    }
    let native_start = HEADER_BYTES + identity_len;
    if bytes.len() != native_start + native_len {
        return Err(Refusal::Envelope);
    }
    if bytes[HEADER_BYTES..native_start] != key.identity {
        return Err(Refusal::Identity);
    }
    let checksum = u128::from_le_bytes(bytes[16..32].try_into().unwrap());
    let native = &bytes[native_start..];
    if checksum != xxhash_rust::xxh3::xxh3_128(native) {
        return Err(Refusal::Checksum);
    }
    if !pipeline_cache_blob_compatible(native, props) {
        return Err(Refusal::Device);
    }
    Ok(Blob { bytes, native_start, native_checksum: checksum })
}

fn load(
    path: &Path,
    key: &Key,
    props: &vk::PhysicalDeviceProperties,
) -> Result<Option<Blob>, Refusal> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io("read", error)),
    };
    let len = file
        .metadata()
        .map_err(|error| io("metadata", error))?
        .len();
    if len > MAX_ENTRY_BYTES as u64 {
        return Err(Refusal::Limit {
            bytes: len as usize,
            cap: MAX_ENTRY_BYTES,
        });
    }
    let mut bytes = Vec::with_capacity(len as usize);
    Read::by_ref(&mut file)
        .take(MAX_ENTRY_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| io("read", error))?;
    if bytes.len() > MAX_ENTRY_BYTES {
        return Err(Refusal::Limit {
            bytes: bytes.len(),
            cap: MAX_ENTRY_BYTES,
        });
    }
    decode(bytes, key, props).map(Some)
}

struct Retained<T> {
    key: Key,
    path: Option<PathBuf>,
    value: Arc<T>,
    bytes: usize,
}

struct RetainedPrograms<T> {
    entries: VecDeque<Retained<T>>,
    max_entries: usize,
    max_bytes: usize,
}

impl<T> RetainedPrograms<T> {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
            max_bytes,
        }
    }

    fn get(&mut self, key: &Key, path: &Option<PathBuf>) -> Option<Arc<T>> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.key == *key && entry.path == *path)?;
        let entry = self.entries.remove(index)?;
        let value = Arc::clone(&entry.value);
        self.entries.push_back(entry);
        Some(value)
    }

    fn insert(&mut self, key: Key, path: Option<PathBuf>, value: Arc<T>, bytes: usize) -> Arc<T> {
        if let Some(existing) = self.get(&key, &path) {
            return existing;
        }
        self.entries.push_back(Retained {
            key,
            path,
            value: Arc::clone(&value),
            bytes,
        });
        self.trim();
        value
    }

    fn size(&mut self, key: &Key, path: &Option<PathBuf>, bytes: usize) {
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.key == *key && entry.path == *path)
        {
            entry.bytes = bytes;
        }
        self.trim();
    }

    fn trim(&mut self) {
        let mut bytes: u128 = self.entries.iter().map(|entry| entry.bytes as u128).sum();
        while self.entries.len() > self.max_entries || bytes > self.max_bytes as u128 {
            let Some(entry) = self.entries.pop_front() else {
                break;
            };
            bytes -= entry.bytes as u128;
            crate::observe::off(format!(
                "vk_pipeline_cache_entry_release key={:032x} bytes={}",
                entry.key.bucket, entry.bytes,
            ));
        }
    }
}

struct Allocation {
    access: parking_lot::Mutex<()>,
    device: ash::Device,
    handle: vk::PipelineCache,
    owned: bool,
    initial_payload: Option<InitialPayload>,
}

fn with_cache_access<R>(
    access: &parking_lot::Mutex<()>, background: bool, handle: vk::PipelineCache,
    create: impl FnOnce(vk::PipelineCache) -> R,
) -> R {
    if background {
        let _access = access.lock();
        return create(handle);
    }
    match access.try_lock() {
        Some(_access) => create(handle),
        None => {
            // Cache hints are optional. A synchronous fallback must not wait
            // on a compiler-owned cache while retaining device service locks.
            crate::runtime::drain::note_store_route("native_cache_busy_uncached");
            create(vk::PipelineCache::null())
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InitialPayload {
    pub bytes: usize,
    pub checksum: u128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Origin {
    Resident,
    Disk,
    Empty,
    Rejected,
    SharedFallback,
    BusyUncached,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Diagnostic {
    pub program: u128,
    pub handle: u64,
    pub origin: Origin,
    pub initial_payload: Option<InitialPayload>,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if self.owned {
            unsafe { self.device.destroy_pipeline_cache(self.handle, None) };
        }
    }
}

pub(super) struct ResidentCaches {
    entries: parking_lot::Mutex<RetainedPrograms<Allocation>>,
    // Fields drop in declaration order: every cache allocation precedes the device.
    _device: Option<Arc<super::NativeDeviceOwner>>,
}

impl Default for ResidentCaches {
    fn default() -> Self {
        Self {
            entries: parking_lot::Mutex::new(RetainedPrograms::new(32, 128 * 1024 * 1024)),
            _device: None,
        }
    }
}

impl ResidentCaches {
    pub(super) fn with_owner(owner: Arc<super::NativeDeviceOwner>) -> Self {
        Self { _device: Some(owner), ..Self::default() }
    }

    pub(super) fn clear(&self) {
        self.entries.lock().entries.clear();
    }

    #[cfg(test)]
    pub(super) fn levels(&self) -> (usize, usize) {
        let cache = self.entries.lock();
        (cache.entries.len(), cache.entries.iter().map(|entry| entry.bytes).sum())
    }
}

pub(crate) struct CompileCache<'a> {
    context: &'a DeviceContext,
    allocation: Arc<Allocation>,
    path: Option<PathBuf>,
    key: Key,
    origin: Origin,
}

impl<'a> CompileCache<'a> {
    pub(crate) fn new(ctx: &'a DeviceContext, key: Key) -> Self {
        let directory = ctx.pipeline_cache_path.as_deref().and_then(Path::parent);
        Self::at(ctx, key, directory)
    }

    fn at(ctx: &'a DeviceContext, key: Key, directory: Option<&Path>) -> Self {
        if key.identity.len() > MAX_IDENTITY_BYTES {
            report(&Refusal::Limit {
                bytes: key.identity.len(),
                cap: MAX_IDENTITY_BYTES,
            });
            return Self {
                context: ctx,
                allocation: Arc::new(Allocation {
                    access: parking_lot::Mutex::new(()),
                    device: ctx.device.clone(),
                    handle: ctx.pipeline_cache,
                    owned: false,
                    initial_payload: None,
                }),
                path: None,
                key,
                origin: Origin::SharedFallback,
            };
        }
        let path = directory.map(|directory| key.path(directory));
        if let Some(allocation) = ctx.native_caches.entries.lock().get(&key, &path) {
            return Self {
                context: ctx,
                allocation,
                path,
                key,
                origin: Origin::Resident,
            };
        }
        let props = unsafe { ctx.instance.get_physical_device_properties(ctx.pd) };
        let mut rejected = false;
        let mut initial = match path.as_deref().map(|path| load(path, &key, &props)) {
            Some(Ok(blob)) => blob,
            Some(Err(refusal)) => {
                report(&refusal);
                rejected = true;
                None
            }
            None => None,
        };
        let info = vk::PipelineCacheCreateInfo::default()
            .initial_data(initial.as_ref().map_or(&[], Blob::native));
        let created = unsafe { ctx.device.create_pipeline_cache(&info, None) };
        let created = match created {
            Err(error) if initial.is_some() => {
                report(&Refusal::Driver(error));
                rejected = true;
                initial = None;
                unsafe {
                    ctx.device
                        .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                }
            }
            other => other,
        };
        let (handle, owned) = match created {
            Ok(handle) => (handle, true),
            Err(error) => {
                report(&Refusal::Driver(error));
                (ctx.pipeline_cache, false)
            }
        };
        crate::observe::off(format!(
            "vk_pipeline_cache_entry_load key={:032x} bytes={} owned={owned}",
            key.bucket,
            initial.as_ref().map_or(0, |blob| blob.native().len()),
        ));
        if owned && initial.is_some() {
            if let Some(path) = &path {
                super::persist::submit(super::persist::Job::Native(Save::Touch {
                    path: path.clone(),
                }));
            }
        }
        let allocation = Arc::new(Allocation {
            access: parking_lot::Mutex::new(()),
            device: ctx.device.clone(),
            handle,
            owned,
            initial_payload: initial.as_ref().map(|blob| InitialPayload {
                bytes: blob.native().len(),
                checksum: blob.native_checksum,
            }),
        });
        let origin = if !owned {
            Origin::SharedFallback
        } else if initial.is_some() {
            Origin::Disk
        } else if rejected {
            Origin::Rejected
        } else {
            Origin::Empty
        };
        let allocation = if owned {
            ctx.native_caches.entries.lock().insert(
                key.clone(),
                path.clone(),
                allocation,
                initial.as_ref().map_or(0, |blob| blob.native().len()),
            )
        } else {
            allocation
        };
        Self {
            context: ctx,
            allocation,
            path,
            key,
            origin,
        }
    }

    fn data(&self) -> Result<Vec<u8>, Refusal> {
        let result = Self::bounded_data(&self.allocation.device, self.handle(), MAX_PROGRAM_BYTES);
        let size = match &result {
            Ok(data) => Some(data.len()),
            Err(Refusal::Limit { bytes, .. }) => Some(*bytes),
            Err(_) => None,
        };
        if let Some(size) = size {
            self.context
                .native_caches
                .entries
                .lock()
                .size(&self.key, &self.path, size);
        }
        result
    }

    pub(super) fn bounded_data(
        device: &ash::Device,
        cache: vk::PipelineCache,
        cap: usize,
    ) -> Result<Vec<u8>, Refusal> {
        for _ in 0..2 {
            let mut size = 0usize;
            let result = unsafe {
                (device.fp_v1_0().get_pipeline_cache_data)(
                    device.handle(),
                    cache,
                    &mut size,
                    std::ptr::null_mut(),
                )
            };
            if result != vk::Result::SUCCESS {
                return Err(Refusal::Data(result));
            }
            if size > cap {
                return Err(Refusal::Limit { bytes: size, cap });
            }
            let mut data = vec![0; size];
            let result = unsafe {
                (device.fp_v1_0().get_pipeline_cache_data)(
                    device.handle(),
                    cache,
                    &mut size,
                    data.as_mut_ptr().cast(),
                )
            };
            match result {
                vk::Result::SUCCESS => {
                    if size > data.len() {
                        return Err(Refusal::Envelope);
                    }
                    data.truncate(size);
                    return Ok(data);
                }
                vk::Result::INCOMPLETE => {}
                other => return Err(Refusal::Data(other)),
            }
        }
        Err(Refusal::Data(vk::Result::INCOMPLETE))
    }

    pub(crate) fn handle(&self) -> vk::PipelineCache {
        self.allocation.handle
    }

    pub(crate) fn with_handle<R>(&self, create: impl FnOnce(vk::PipelineCache) -> R) -> R {
        with_cache_access(&self.allocation.access, self.context.compile_only, self.allocation.handle, create)
    }

    pub(crate) fn diagnostic(&self) -> Diagnostic {
        self.diagnostic_for(self.handle())
    }

    pub(crate) fn diagnostic_for(&self, handle: vk::PipelineCache) -> Diagnostic {
        use ash::vk::Handle as _;
        let original = handle == self.handle();
        Diagnostic {
            program: self.key.bucket,
            handle: handle.as_raw(),
            origin: if original { self.origin } else { Origin::BusyUncached },
            initial_payload: if original { self.allocation.initial_payload } else { None },
        }
    }

    pub(crate) fn save(&self) {
        let Some(_access) = self.allocation.access.try_lock() else {
            crate::runtime::drain::note_store_route("native_cache_save_busy");
            return;
        };
        if !self.allocation.owned {
            return;
        }
        let Some(path) = self.path.as_ref() else {
            return;
        };
        let data = match self.data() {
            Ok(data) => data,
            Err(refusal) => {
                report(&refusal);
                return;
            }
        };
        match encode(&self.key, &data) {
            Ok(data) => super::persist::submit(super::persist::Job::Native(Save::Write {
                path: path.clone(),
                data,
            })),
            Err(refusal) => report(&refusal),
        }
    }
}

pub(super) enum Save {
    Write { path: PathBuf, data: Vec<u8> },
    Touch { path: PathBuf },
}

impl Save {
    pub(super) fn path(&self) -> &Path {
        match self {
            Self::Write { path, .. } | Self::Touch { path } => path,
        }
    }

    pub(super) fn bytes(&self) -> usize {
        match self {
            Self::Write { data, .. } => data.len(),
            Self::Touch { .. } => 0,
        }
    }

    pub(super) fn is_touch(&self) -> bool {
        matches!(self, Self::Touch { .. })
    }

    pub(super) fn run(self) {
        let result = match self {
            Self::Write { path, data } => {
                let result = save_bounded(&path, &data, MAX_NATIVE_BYTES, MAX_NATIVE_ENTRIES);
                if result.is_ok() {
                    crate::observe::off(format!(
                        "vk_pipeline_cache_entry_save bytes={} path={}",
                        data.len(),
                        path.display(),
                    ));
                }
                result
            }
            Self::Touch { path } => match File::open(path) {
                Ok(file) => file
                    .set_modified(SystemTime::now())
                    .map_err(|error| io("touch", error)),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(io("touch", error)),
            },
        };
        if let Err(refusal) = result {
            report(&refusal);
        }
    }
}

fn native_entry(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "bin")
        && path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .is_some_and(|stem| {
                stem.len() == 32 && stem.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
}

fn make_room(path: &Path, bytes: usize, byte_cap: u64, entry_cap: usize) -> Result<(), Refusal> {
    if bytes as u64 > byte_cap || entry_cap == 0 {
        return Err(Refusal::Limit {
            bytes,
            cap: byte_cap as usize,
        });
    }
    let directory = path.parent().ok_or(Refusal::Envelope)?;
    let mut entries = Vec::new();
    let mut total = bytes as u64;
    for entry in fs::read_dir(directory).map_err(|error| io("scan", error))? {
        let entry = entry.map_err(|error| io("scan", error))?;
        let candidate = entry.path();
        if candidate == path
            || !native_entry(&candidate)
            || !entry
                .file_type()
                .map_err(|error| io("metadata", error))?
                .is_file()
        {
            continue;
        }
        let metadata = entry.metadata().map_err(|error| io("metadata", error))?;
        let modified = metadata.modified().map_err(|error| io("metadata", error))?;
        total = total.checked_add(metadata.len()).ok_or(Refusal::Envelope)?;
        entries.push((modified, candidate, metadata.len()));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut count = entries.len() + 1;
    for (_, candidate, len) in entries {
        if total <= byte_cap && count <= entry_cap {
            break;
        }
        fs::remove_file(&candidate).map_err(|error| io("evict", error))?;
        total -= len;
        count -= 1;
        crate::observe::off(format!(
            "vk_pipeline_cache_entry_evict bytes={len} path={}",
            candidate.display(),
        ));
    }
    Ok(())
}

fn save_bounded(path: &Path, data: &[u8], byte_cap: u64, entry_cap: usize) -> Result<(), Refusal> {
    if data.len() > MAX_ENTRY_BYTES {
        return Err(Refusal::Limit {
            bytes: data.len(),
            cap: MAX_ENTRY_BYTES,
        });
    }
    let directory = path.parent().ok_or(Refusal::Envelope)?;
    fs::create_dir_all(directory).map_err(|error| io("directory", error))?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(directory.join(".native-cache.lock"))
        .map_err(|error| io("lock", error))?;
    lock.lock().map_err(|error| io("lock", error))?;
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    let result = (|| {
        let mut file = File::create(&tmp).map_err(|error| io("write", error))?;
        file.write_all(data).map_err(|error| io("write", error))?;
        file.sync_all().map_err(|error| io("sync", error))?;
        make_room(path, data.len(), byte_cap, entry_cap)?;
        fs::rename(&tmp, path).map_err(|error| io("rename", error))
    })();
    if result.is_err() {
        match fs::remove_file(&tmp) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => report(&io("cleanup", error)),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_compile_hint_access_does_not_block_device_service() {
        use ash::vk::Handle;
        use std::sync::mpsc;
        use std::time::Duration;
        let access = parking_lot::Mutex::new(());
        let handle = vk::PipelineCache::from_raw(7);
        let (entered, started) = mpsc::channel();
        let (release, resume) = mpsc::channel();
        std::thread::scope(|scope| {
            let access_ref = &access;
            let thread = scope.spawn(move || with_cache_access(access_ref, true, handle, |seen| {
                assert_eq!(seen, handle);
                entered.send(()).unwrap();
                resume.recv_timeout(Duration::from_secs(2)).unwrap();
            }));
            started.recv_timeout(Duration::from_secs(2)).unwrap();
            assert!(access.try_lock().is_none());
            assert_eq!(with_cache_access(&access, false, handle, |seen| seen), vk::PipelineCache::null());
            release.send(()).unwrap();
            thread.join().unwrap();
        });
        assert_eq!(with_cache_access(&access, false, handle, |seen| seen), handle);
    }

    #[test]
    fn shared_program_budget_retains_one_payload_across_many_pipeline_views() {
        let shared = Arc::new(parking_lot::Mutex::new(RetainedPrograms::new(32, 128 * 1024 * 1024)));
        let views: Vec<_> = (0..64).map(|_| shared.clone()).collect();
        let first = shared.lock().insert(key(1), None, Arc::new(42), 60 * 1024 * 1024);
        for view in &views {
            assert!(Arc::ptr_eq(view, &shared));
            assert!(Arc::ptr_eq(&view.lock().get(&key(1), &None).unwrap(), &first));
        }
        let cache = shared.lock();
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.entries[0].bytes, 60 * 1024 * 1024);
    }
    use std::sync::atomic::{AtomicU64, Ordering};

    fn with_directory(test: impl FnOnce(&Path)) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "reims-vgpu-native-cache-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&directory).unwrap();
        test(&directory);
        fs::remove_dir_all(directory).unwrap();
    }

    fn props() -> vk::PhysicalDeviceProperties {
        vk::PhysicalDeviceProperties {
            vendor_id: 13,
            device_id: 27,
            pipeline_cache_uuid: [3; 16],
            ..Default::default()
        }
    }

    fn native(bytes: usize) -> Vec<u8> {
        let props = props();
        let mut blob = Vec::new();
        for word in [32, 1, props.vendor_id, props.device_id] {
            blob.extend_from_slice(&word.to_le_bytes());
        }
        blob.extend_from_slice(&props.pipeline_cache_uuid);
        blob.resize(bytes.max(32), 0x5a);
        blob
    }

    fn key(value: u32) -> Key {
        Key::new(0, &(value, "pipeline"), &[&[0x07230203, value]])
    }

    #[test]
    fn native_cache_exact_identity_survives_forced_filename_collision() {
        let first = key(1);
        let mut second = key(2);
        second.bucket = first.bucket;
        assert_eq!(
            first.path(Path::new("/cache")),
            second.path(Path::new("/cache"))
        );
        let blob = encode(&first, &native(128)).unwrap();
        assert!(matches!(
            decode(blob, &second, &props()),
            Err(Refusal::Identity)
        ));
    }

    #[test]
    fn native_cache_key_distinguishes_shader_content_stage_and_descriptor() {
        let first = key(1);
        assert_ne!(first, key(2));
        assert_ne!(first, Key::new(1, &(1u32, "pipeline"), &[&[0x07230203, 1]]));
        assert_ne!(first, Key::new(0, &(1u32, "pipeline"), &[&[0x07230203, 2]]));
        assert_ne!(
            first,
            Key::new(0, &(1u32, "pipeline"), &[&[0x07230203], &[1]])
        );
    }

    #[test]
    fn native_cache_payload_corruption_and_device_mismatch_never_reach_driver() {
        let key = key(1);
        let mut bytes = encode(&key, &native(128)).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        assert!(matches!(
            decode(bytes, &key, &props()),
            Err(Refusal::Checksum)
        ));
        let mut other = props();
        other.device_id += 1;
        assert!(matches!(
            decode(encode(&key, &native(128)).unwrap(), &key, &other),
            Err(Refusal::Device),
        ));
    }

    #[test]
    fn native_cache_rejects_truncated_and_oversized_envelopes() {
        let key = key(1);
        let bytes = encode(&key, &native(128)).unwrap();
        for length in [0, HEADER_BYTES - 1, HEADER_BYTES, bytes.len() - 1] {
            assert!(matches!(
                decode(bytes[..length].to_vec(), &key, &props()),
                Err(Refusal::Envelope)
            ));
        }
        assert!(matches!(
            encode(&key, &native(MAX_PROGRAM_BYTES + 1)),
            Err(Refusal::Limit { .. }),
        ));
        let huge = Key {
            identity: vec![0; MAX_IDENTITY_BYTES + 1],
            bucket: 0,
        };
        assert!(matches!(
            encode(&huge, &native(32)),
            Err(Refusal::Limit { .. })
        ));
    }

    #[test]
    fn native_cache_two_large_programs_warm_without_discarding_aggregate() {
        with_directory(|directory| {
            let payload = native(7_590_000);
            assert!(payload.len() * 2 > PIPELINE_CACHE_MAX_WARM_BYTES);
            for value in [1, 2] {
                let key = key(value);
                save_bounded(
                    &key.path(directory),
                    &encode(&key, &payload).unwrap(),
                    MAX_NATIVE_BYTES,
                    8,
                )
                .unwrap();
            }
            for value in [1, 2] {
                let key = key(value);
                let loaded = load(&key.path(directory), &key, &props()).unwrap().unwrap();
                assert_eq!(loaded.native(), payload);
            }
        });
    }

    #[test]
    fn native_cache_atomic_replacement_accepts_newer_smaller_content() {
        with_directory(|directory| {
            let key = key(1);
            let path = key.path(directory);
            save_bounded(&path, &encode(&key, &native(4096)).unwrap(), 8192, 8).unwrap();
            save_bounded(&path, &encode(&key, &native(128)).unwrap(), 8192, 8).unwrap();
            assert_eq!(
                load(&path, &key, &props()).unwrap().unwrap().native(),
                native(128)
            );
            assert!(!path
                .with_extension(format!("tmp.{}", std::process::id()))
                .exists());
        });
    }

    #[test]
    fn native_cache_budget_evicts_only_oldest_entry_and_preserves_unrelated_files() {
        with_directory(|directory| {
            let old = key(1).path(directory);
            let hot = key(2).path(directory);
            let fresh = key(3).path(directory);
            let data = encode(&key(1), &native(128)).unwrap();
            let cap = (data.len() * 2) as u64;
            save_bounded(&old, &data, cap, 8).unwrap();
            save_bounded(&hot, &data, cap, 8).unwrap();
            File::open(&old)
                .unwrap()
                .set_modified(SystemTime::UNIX_EPOCH)
                .unwrap();
            fs::write(directory.join("shared.bin"), b"utility-cache").unwrap();
            fs::write(directory.join("unrelated"), b"untouched").unwrap();
            save_bounded(&fresh, &data, cap, 8).unwrap();
            assert!(!old.exists());
            assert!(hot.exists() && fresh.exists());
            assert_eq!(
                fs::read(directory.join("shared.bin")).unwrap(),
                b"utility-cache"
            );
            assert_eq!(fs::read(directory.join("unrelated")).unwrap(), b"untouched");
        });
    }

    #[test]
    fn native_cache_oversized_admission_preserves_existing_file() {
        with_directory(|directory| {
            let key = key(1);
            let path = key.path(directory);
            let original = encode(&key, &native(128)).unwrap();
            save_bounded(&path, &original, 8192, 8).unwrap();
            assert!(
                save_bounded(&path, &vec![0; MAX_ENTRY_BYTES + 1], MAX_NATIVE_BYTES, 8).is_err()
            );
            assert_eq!(fs::read(path).unwrap(), original);
        });
    }

    #[test]
    fn native_cache_pending_coalesces_same_path_but_keeps_other_programs() {
        use super::super::persist::{Job, Pending};
        let mut queue = Pending::default();
        let write = |name: &str, bytes: usize| {
            Job::Native(Save::Write {
                path: PathBuf::from(name),
                data: vec![0; bytes],
            })
        };
        assert!(queue.push(write("a", 100)));
        assert!(queue.push(write("b", 20)));
        assert!(queue.push(write("a", 50)));
        assert!(queue.push(Job::Native(Save::Touch {
            path: PathBuf::from("a")
        })));
        let Job::Native(first) = queue.pop().unwrap() else {
            panic!("native entry")
        };
        assert_eq!(first.path(), Path::new("a"));
        assert_eq!(first.bytes(), 50);
        let Job::Native(second) = queue.pop().unwrap() else {
            panic!("native entry")
        };
        assert_eq!(second.path(), Path::new("b"));
        assert!(queue.pop().is_none());
    }

    #[test]
    fn native_cache_pending_budget_refusal_does_not_drop_accepted_entries() {
        use super::super::persist::{Job, Pending};
        let mut queue = Pending::default();
        for name in ["a", "b"] {
            assert!(queue.push(Job::Native(Save::Write {
                path: name.into(),
                data: vec![0; MAX_ENTRY_BYTES],
            })));
        }
        assert!(!queue.push(Job::Native(Save::Write {
            path: "c".into(),
            data: vec![0; 1],
        })));
        assert!(queue.pop().is_some());
        assert!(queue.pop().is_some());
        assert!(queue.pop().is_none());
    }

    #[test]
    fn native_cache_live_eviction_keeps_an_active_compile_lease_alive() {
        struct Counted(Arc<AtomicU64>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let drops = Arc::new(AtomicU64::new(0));
        let mut held = RetainedPrograms::new(1, 1024);
        let first = held.insert(key(1), None, Arc::new(Counted(Arc::clone(&drops))), 512);
        let same = held.get(&key(1), &None).unwrap();
        assert!(Arc::ptr_eq(&first, &same));
        drop(same);
        drop(held.insert(key(2), None, Arc::new(Counted(Arc::clone(&drops))), 512));
        assert!(held.get(&key(1), &None).is_none());
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(first);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        held.entries.clear();
        assert_eq!(drops.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn native_cache_live_byte_budget_and_namespace_are_independent() {
        let mut held = RetainedPrograms::new(8, 100);
        let directory_a = Some(PathBuf::from("a/program.bin"));
        let directory_b = Some(PathBuf::from("b/program.bin"));
        let a = held.insert(key(1), directory_a.clone(), Arc::new(1), 40);
        let b = held.insert(key(1), directory_b.clone(), Arc::new(2), 40);
        assert!(!Arc::ptr_eq(&a, &b));
        assert!(Arc::ptr_eq(&a, &held.get(&key(1), &directory_a).unwrap()));
        held.size(&key(1), &directory_a, 70);
        assert!(held.get(&key(1), &directory_b).is_none());
        assert_eq!(held.entries.len(), 1);
        assert!(Arc::ptr_eq(&a, &held.get(&key(1), &directory_a).unwrap()));
    }

    #[test]
    #[ignore = "requires an exclusive Vulkan GPU with native cache serialization"]
    fn native_cache_gpu_warm_program_and_variant_owner() {
        use crate::backend::vulkan::sampled_shader::graphics_tests::assemble;
        let words = assemble(
            r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %block BufferBlock
OpMemberDecorate %block 0 Offset 0
OpDecorate %output DescriptorSet 0
OpDecorate %output Binding 0
OpDecorate %value SpecId 7
%void = OpTypeVoid
%function = OpTypeFunction %void
%uint = OpTypeInt 32 0
%block = OpTypeStruct %uint
%block_ptr = OpTypePointer Uniform %block
%uint_ptr = OpTypePointer Uniform %uint
%output = OpVariable %block_ptr Uniform
%zero = OpConstant %uint 0
%value = OpSpecConstant %uint 1
%main = OpFunction %void None %function
%label = OpLabel
%address = OpAccessChain %uint_ptr %output %zero
OpStore %address %value
OpReturn
OpFunctionEnd
"#,
        );
        let mut context = unsafe { DeviceContext::create().unwrap() };
        with_directory(|directory| {
            let program = || Key::new(1, &"main", &[&words]);
            let properties = unsafe { context.instance.get_physical_device_properties(context.pd) };
            let shader = unsafe {
                context
                    .device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                    .unwrap()
            };
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let descriptor_layout = unsafe {
                context
                    .device
                    .create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                        None,
                    )
                    .unwrap()
            };
            let set_layouts = [descriptor_layout];
            let plain_layout = unsafe {
                context
                    .device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                        None,
                    )
                    .unwrap()
            };
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(4)];
            let other_layout = unsafe {
                context
                    .device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&set_layouts)
                            .push_constant_ranges(&ranges),
                        None,
                    )
                    .unwrap()
            };
            let entry = std::ffi::CString::new("main").unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(&entry);
            let plain = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(plain_layout);
            let special_value = 2u32.to_ne_bytes();
            let entries = [vk::SpecializationMapEntry::default()
                .constant_id(7)
                .offset(0)
                .size(4)];
            let specialization = vk::SpecializationInfo::default()
                .map_entries(&entries)
                .data(&special_value);
            let variant = vk::ComputePipelineCreateInfo::default()
                .stage(stage.specialization_info(&specialization))
                .layout(other_layout);
            let cache = CompileCache::at(&context, program(), Some(directory));
            let handle = cache.handle();
            let cold = std::time::Instant::now();
            let first = unsafe {
                context
                    .device
                    .create_compute_pipelines(handle, &[plain], None)
                    .unwrap()[0]
            };
            let cold_us = cold.elapsed().as_micros();
            drop(cache);
            let cache = CompileCache::at(&context, program(), Some(directory));
            assert_eq!(
                cache.handle(),
                handle,
                "variants retain the same native program cache"
            );
            let second = unsafe {
                context
                    .device
                    .create_compute_pipelines(cache.handle(), &[variant], None)
                    .unwrap()[0]
            };
            cache.save();
            let path = program().path(directory);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let loaded = loop {
                if let Some(blob) = load(&path, &program(), &properties).unwrap() {
                    break blob;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "native persistence did not complete"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            };
            assert!(loaded.native().len() > super::super::PIPELINE_CACHE_HEADER_ONE_LEN);
            drop(cache);
            context.native_caches.clear();
            let cache = CompileCache::at(&context, program(), Some(directory));
            assert!(cache.data().unwrap().len() > super::super::PIPELINE_CACHE_HEADER_ONE_LEN);
            let warm = std::time::Instant::now();
            let third = unsafe {
                context
                    .device
                    .create_compute_pipelines(cache.handle(), &[plain], None)
                    .unwrap()[0]
            };
            let warm_us = warm.elapsed().as_micros();
            eprintln!(
                "native_cache_probe cold_create_us={cold_us} warm_create_us={warm_us} native_bytes={}",
                loaded.native().len(),
            );
            drop(cache);
            unsafe {
                let buffer = context
                    .device
                    .create_buffer(
                        &vk::BufferCreateInfo::default()
                            .size(4)
                            .usage(vk::BufferUsageFlags::STORAGE_BUFFER),
                        None,
                    )
                    .unwrap();
                let requirements = context.device.get_buffer_memory_requirements(buffer);
                let memory_type = context
                    .memory_type_for(
                        requirements.memory_type_bits,
                        requirements.size,
                        crate::backend::vulkan::caps::memory_topology::MemoryClass::Readback,
                    )
                    .unwrap();
                let memory = context
                    .device
                    .allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(requirements.size)
                            .memory_type_index(memory_type),
                        None,
                    )
                    .unwrap();
                context
                    .device
                    .bind_buffer_memory(buffer, memory, 0)
                    .unwrap();
                let mapped = context
                    .device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .unwrap()
                    .cast::<u32>();
                let mapped_range = vk::MappedMemoryRange::default()
                    .memory(memory)
                    .size(vk::WHOLE_SIZE);
                let pool_sizes = [vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                }];
                let descriptors = context
                    .device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(1)
                            .pool_sizes(&pool_sizes),
                        None,
                    )
                    .unwrap();
                let set = context
                    .device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptors)
                            .set_layouts(&set_layouts),
                    )
                    .unwrap()[0];
                let buffers = [vk::DescriptorBufferInfo::default().buffer(buffer).range(4)];
                let writes = [vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffers)];
                context.device.update_descriptor_sets(&writes, &[]);
                let pool = context
                    .device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(context.gq)
                            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None,
                    )
                    .unwrap();
                let command = context
                    .device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .unwrap()[0];
                let fence = context
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .unwrap();
                for (pipeline, layout, expected) in [
                    (first, plain_layout, 1),
                    (second, other_layout, 2),
                    (third, plain_layout, 1),
                ] {
                    mapped.write(0);
                    context
                        .device
                        .flush_mapped_memory_ranges(&[mapped_range])
                        .unwrap();
                    context
                        .device
                        .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                        .unwrap();
                    context.device.cmd_bind_pipeline(
                        command,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline,
                    );
                    context.device.cmd_bind_descriptor_sets(
                        command,
                        vk::PipelineBindPoint::COMPUTE,
                        layout,
                        0,
                        &[set],
                        &[],
                    );
                    context.device.cmd_dispatch(command, 1, 1, 1);
                    let barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ);
                    context.device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                    context.device.end_command_buffer(command).unwrap();
                    context.submit_guest_work(&[command], fence).unwrap();
                    context
                        .device
                        .wait_for_fences(&[fence], true, super::super::FENCE_TIMEOUT_NS)
                        .unwrap();
                    context
                        .device
                        .invalidate_mapped_memory_ranges(&[mapped_range])
                        .unwrap();
                    assert_eq!(mapped.read(), expected, "cold/variant/warm pipeline output");
                    context.device.reset_fences(&[fence]).unwrap();
                    context
                        .device
                        .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
                        .unwrap();
                }
                context.device.destroy_fence(fence, None);
                context.device.destroy_command_pool(pool, None);
                context.device.destroy_descriptor_pool(descriptors, None);
                context.device.unmap_memory(memory);
                context.device.destroy_buffer(buffer, None);
                context.device.free_memory(memory, None);
                for pipeline in [first, second, third] {
                    context.device.destroy_pipeline(pipeline, None);
                }
                context.device.destroy_pipeline_layout(plain_layout, None);
                context.device.destroy_pipeline_layout(other_layout, None);
                context.device.destroy_shader_module(shader, None);
                context
                    .device
                    .destroy_descriptor_set_layout(descriptor_layout, None);
            }
        });
        unsafe { context.destroy() };
    }
}
