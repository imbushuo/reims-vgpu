use super::{log_token, ShaderReflection, Stage};
use crate::observe::Decline;
use bincode::Options;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8; 8] = b"RM2VC02\0";
const HEADER: usize = 48;
const MAX_ENTRY: usize = 24 * 1024 * 1024;
const MAX_BYTES: u64 = 512 * 1024 * 1024;
const MAX_ENTRIES: usize = 2048;

#[derive(Debug)]
enum Refusal {
    Identity(String),
    Io(&'static str, std::io::Error),
    Corrupt(&'static str),
    Reflection(String),
    Limit,
}

impl Decline for Refusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::Identity(_) => "m2v_disk_cache_identity",
            Self::Io(..) => "m2v_disk_cache_io",
            Self::Corrupt(_) => "m2v_disk_cache_corrupt",
            Self::Reflection(_) => "m2v_disk_cache_reflection",
            Self::Limit => "m2v_disk_cache_limit",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Identity(detail) | Self::Reflection(detail) => {
                vec![("detail", log_token(detail))]
            }
            Self::Io(op, error) => vec![
                ("op", (*op).into()),
                ("detail", log_token(&error.to_string())),
            ],
            Self::Corrupt(reason) => vec![("detail", (*reason).into())],
            Self::Limit => vec![("max_entry_bytes", MAX_ENTRY.to_string())],
        }
    }
}

fn report(error: &Refusal) {
    crate::observe::Emit::decline("m2v_disk_cache", error).fail_once(0);
}

pub(super) struct Entry {
    key: Vec<u8>,
    stage: Stage,
    local: Option<[u32; 3]>,
    root: PathBuf,
    path: PathBuf,
}

impl Entry {
    pub(super) fn for_air(
        air: &[u8],
        stage: Stage,
        options: &metal2vulkan::passes::TransformOptions,
    ) -> Option<Self> {
        if air.len() > MAX_ENTRY / 2 {
            report(&Refusal::Limit);
            return None;
        }
        let (product, llvm) = match metal2vulkan::tools::translation_cache_identity() {
            Ok(identity) => identity,
            Err(error) => {
                report(&Refusal::Identity(error));
                return None;
            }
        };
        let option_identity = match options.cache_identity() {
            Ok(identity) => identity,
            Err(error) => {
                report(&Refusal::Identity(error));
                return None;
            }
        };
        let local = (stage == Stage::Kernel).then_some(options.kernel_local_size);
        let mut key = Vec::with_capacity(air.len() + product.len() + llvm.len() + 32);
        key.extend_from_slice(product.as_bytes());
        key.push(0);
        key.extend_from_slice(llvm.as_bytes());
        key.push(0);
        key.push(match stage {
            Stage::Vertex => 0,
            Stage::Fragment => 1,
            Stage::Kernel => 2,
        });
        key.extend_from_slice(&(option_identity.len() as u64).to_le_bytes());
        key.extend_from_slice(&option_identity);
        key.extend_from_slice(air);
        let root = std::env::temp_dir().join("reims-vgpu-translations-v2");
        let path = root.join(format!("{:032x}.m2v", xxhash_rust::xxh3::xxh3_128(&key)));
        Some(Self {
            key,
            stage,
            local,
            root,
            path,
        })
    }

    pub(super) fn load(&self) -> Option<(Vec<u8>, ShaderReflection)> {
        match self.read() {
            Ok(Some(result)) => {
                crate::runtime::drain::note_store_route("translation_disk_hit");
                crate::observe::off(format!("m2v_disk_cache_hit bytes={}", result.0.len()));
                Some(result)
            }
            Ok(None) => {
                crate::runtime::drain::note_store_route("translation_disk_miss");
                None
            }
            Err(error) => {
                report(&error);
                None
            }
        }
    }

    fn read(&self) -> Result<Option<(Vec<u8>, ShaderReflection)>, Refusal> {
        match private_directory(&self.root) {
            Ok(false) => return Ok(None),
            Ok(true) => {}
            Err(error) => return Err(error),
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        let file = match options.open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(Refusal::Io("open", error)),
        };
        if !file
            .metadata()
            .map_err(|error| Refusal::Io("entry_metadata", error))?
            .is_file()
        {
            return Err(Refusal::Corrupt("entry_type"));
        }
        let mut bytes = Vec::new();
        (&file)
            .take(MAX_ENTRY as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| Refusal::Io("read", error))?;
        let (spirv, reflection) = decode(&bytes, &self.key)?;
        if reflection.stage != self.stage.into()
            || self
                .local
                .is_some_and(|local| reflection.local_size != Some(local))
        {
            return Err(Refusal::Corrupt("stage_or_local_size"));
        }
        if let Err(error) =
            file.set_times(fs::FileTimes::new().set_modified(std::time::SystemTime::now()))
        {
            report(&Refusal::Io("touch", error));
        }
        Ok(Some((spirv, reflection)))
    }

    pub(super) fn save(&self, spirv: &[u8], reflection: &ShaderReflection) {
        if let Err(error) = self.write(spirv, reflection) {
            report(&error);
        }
    }

    fn write(&self, spirv: &[u8], reflection: &ShaderReflection) -> Result<(), Refusal> {
        let meta = reflection_codec()
            .serialize(reflection)
            .map_err(|error| Refusal::Reflection(error.to_string()))?;
        let bytes = encode(&self.key, spirv, &meta)?;
        let mut directory = fs::DirBuilder::new();
        directory.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            directory.mode(0o700);
        }
        directory
            .create(&self.root)
            .map_err(|error| Refusal::Io("mkdir", error))?;
        private_directory(&self.root)?;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let temporary = self.root.join(format!(
            ".{}.{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let result = (|| {
            let mut file = options
                .open(&temporary)
                .map_err(|error| Refusal::Io("create", error))?;
            file.write_all(&bytes)
                .map_err(|error| Refusal::Io("write", error))?;
            file.sync_all()
                .map_err(|error| Refusal::Io("sync", error))?;
            fs::rename(&temporary, &self.path).map_err(|error| Refusal::Io("rename", error))?;
            trim(&self.root)
        })();
        if result.is_err() {
            if let Err(error) = fs::remove_file(&temporary) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    report(&Refusal::Io("cleanup", error));
                }
            }
        }
        result
    }
}

fn encode(key: &[u8], spirv: &[u8], meta: &[u8]) -> Result<Vec<u8>, Refusal> {
    let len = HEADER
        .checked_add(key.len())
        .and_then(|n| n.checked_add(spirv.len()))
        .and_then(|n| n.checked_add(meta.len()))
        .ok_or(Refusal::Limit)?;
    if len > MAX_ENTRY {
        return Err(Refusal::Limit);
    }
    let mut bytes = Vec::with_capacity(len);
    bytes.extend_from_slice(MAGIC);
    for n in [key.len(), spirv.len(), meta.len()] {
        bytes.extend_from_slice(&(n as u64).to_le_bytes());
    }
    bytes.extend_from_slice(&[0; 16]);
    bytes.extend_from_slice(key);
    bytes.extend_from_slice(spirv);
    bytes.extend_from_slice(meta);
    let checksum = xxhash_rust::xxh3::xxh3_128(&bytes[HEADER..]).to_le_bytes();
    bytes[32..HEADER].copy_from_slice(&checksum);
    Ok(bytes)
}

fn decode(bytes: &[u8], key: &[u8]) -> Result<(Vec<u8>, ShaderReflection), Refusal> {
    if bytes.len() < HEADER || bytes.len() > MAX_ENTRY || &bytes[..8] != MAGIC {
        return Err(Refusal::Corrupt("envelope"));
    }
    let length = |offset| -> Result<usize, Refusal> {
        usize::try_from(u64::from_le_bytes(
            bytes[offset..offset + 8].try_into().unwrap(),
        ))
        .map_err(|_| Refusal::Limit)
    };
    let key_end = HEADER.checked_add(length(8)?).ok_or(Refusal::Limit)?;
    let spirv_end = key_end.checked_add(length(16)?).ok_or(Refusal::Limit)?;
    let meta_end = spirv_end.checked_add(length(24)?).ok_or(Refusal::Limit)?;
    if meta_end != bytes.len() || key_end > spirv_end || spirv_end > meta_end {
        return Err(Refusal::Corrupt("lengths"));
    }
    if xxhash_rust::xxh3::xxh3_128(&bytes[HEADER..]).to_le_bytes() != bytes[32..HEADER] {
        return Err(Refusal::Corrupt("checksum"));
    }
    if &bytes[HEADER..key_end] != key {
        return Err(Refusal::Corrupt("key"));
    }
    let spirv = &bytes[key_end..spirv_end];
    // Validate persisted executable bytes even though only successfully validated
    // translations are written. Corruption must not become a driver submission.
    metal2vulkan::tools::spirv_val_bytes(spirv, Path::new(""))
        .map_err(|_| Refusal::Corrupt("spirv"))?;
    let reflection: ShaderReflection = reflection_codec()
        .deserialize(&bytes[spirv_end..])
        .map_err(|error| Refusal::Reflection(error.to_string()))?;
    if reflection.reflection_version != metal2vulkan::reflect::REFLECTION_VERSION {
        return Err(Refusal::Corrupt("reflection_version"));
    }
    reflection
        .validate_descriptor_abi()
        .map_err(Refusal::Reflection)?;
    Ok((spirv.to_vec(), reflection))
}

fn reflection_codec() -> impl Options {
    // Binary float encoding preserves non-finite AIR sampler limits and signed
    // zero; JSON would serialize non-finite values as null and lose the payload.
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .with_limit(MAX_ENTRY as u64)
        .reject_trailing_bytes()
}

fn trim(root: &Path) -> Result<(), Refusal> {
    trim_with_limits(root, MAX_BYTES, MAX_ENTRIES)
}

fn trim_with_limits(root: &Path, max_bytes: u64, max_entries: usize) -> Result<(), Refusal> {
    let mut files = Vec::new();
    for entry in fs::read_dir(root).map_err(|error| Refusal::Io("list", error))? {
        let entry = entry.map_err(|error| Refusal::Io("list_entry", error))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(stem) = name.strip_suffix(".m2v") else {
            continue;
        };
        if stem.len() != 32 || !stem.bytes().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(Refusal::Io("metadata", error)),
        };
        if !metadata.is_file() {
            continue;
        }
        let modified = metadata
            .modified()
            .map_err(|error| Refusal::Io("modified", error))?;
        files.push((modified, metadata.len(), entry.path()));
    }

    files.sort_by_key(|(modified, _, _)| *modified);
    let mut bytes: u128 = files.iter().map(|(_, len, _)| u128::from(*len)).sum();
    let mut count = files.len();
    for (_, len, path) in files {
        if bytes <= u128::from(max_bytes) && count <= max_entries {
            break;
        }
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(Refusal::Io("evict", error)),
        }
        bytes -= u128::from(len);
        count -= 1;
    }
    Ok(())
}

fn private_directory(root: &Path) -> Result<bool, Refusal> {
    let metadata = match fs::symlink_metadata(root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(Refusal::Io("directory", error)),
    };
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(Refusal::Corrupt("cache_directory_type"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.uid() != unsafe { libc::geteuid() } || metadata.mode() & 0o077 != 0 {
            return Err(Refusal::Corrupt("cache_directory_owner_or_permissions"));
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload() -> (Vec<u8>, ShaderReflection) {
        metal2vulkan::translate_sanitized_native_reflected(
            r#"
define void @k(ptr addrspace(1) %out) {
entry:
  store i32 7, ptr addrspace(1) %out, align 4
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#,
            Stage::Kernel,
            Path::new(""),
            Default::default(),
        ).expect("authored cache payload")
    }

    #[test]
    fn envelope_requires_exact_identity_checksum_lengths_and_valid_spirv() {
        let (spv, reflection) = payload();
        let meta = reflection_codec().serialize(&reflection).unwrap();
        let encoded = encode(b"exact-input-and-build", &spv, &meta).unwrap();
        let (decoded, reflected) = decode(&encoded, b"exact-input-and-build").unwrap();
        assert_eq!(decoded, spv);
        assert_eq!(reflection_codec().serialize(&reflected).unwrap(), meta);
        assert!(decode(&encoded, b"different-input-or-build").is_err());
        assert!(decode(&encoded[..encoded.len() - 1], b"exact-input-and-build").is_err());
        let mut corrupt = encoded.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(decode(&corrupt, b"exact-input-and-build").is_err());
        let invalid = encode(b"exact-input-and-build", b"invalid", &meta).unwrap();
        assert!(decode(&invalid, b"exact-input-and-build").is_err());
        let mut lengths = encoded;
        lengths[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(decode(&lengths, b"exact-input-and-build").is_err());
    }

    #[test]
    fn reflection_codec_preserves_nonfinite_sampler_bits() {
        use metal2vulkan::reflect::*;
        let sampler = StaticSamplerState {
            min_filter: SamplerFilter::Nearest,
            mag_filter: SamplerFilter::Nearest,
            mip_filter: SamplerMipFilter::None,
            address_mode_s: SamplerAddressMode::ClampToEdge,
            address_mode_t: SamplerAddressMode::ClampToEdge,
            address_mode_r: SamplerAddressMode::ClampToEdge,
            coordinates: SamplerCoordinates::Normalized,
            compare_function: SamplerCompareFunction::None,
            max_anisotropy: 1,
            lod_min_clamp: -0.0,
            lod_max_clamp: f32::INFINITY,
            border_color: SamplerBorderColor::TransparentBlack,
            reduction: SamplerReduction::WeightedAverage,
            lod_bias: f32::from_bits(0x7fc01234),
            raw_words: [u64::MAX, 1],
        };
        let bytes = reflection_codec().serialize(&sampler).unwrap();
        let decoded: StaticSamplerState = reflection_codec().deserialize(&bytes).unwrap();
        assert_eq!(
            decoded.lod_min_clamp.to_bits(),
            sampler.lod_min_clamp.to_bits()
        );
        assert_eq!(
            decoded.lod_max_clamp.to_bits(),
            sampler.lod_max_clamp.to_bits()
        );
        assert_eq!(decoded.lod_bias.to_bits(), sampler.lod_bias.to_bits());
        assert_eq!(decoded.raw_words, sampler.raw_words);
    }

    #[test]
    fn disk_entry_survives_owner_recreation_without_scratch_files() {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "m2v-disk-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let path = root.join("00000000000000000000000000000001.m2v");
        let first = Entry {
            key: b"exact".to_vec(),
            stage: Stage::Kernel,
            local: Some([64, 1, 1]),
            root: root.clone(),
            path: path.clone(),
        };
        assert!(first.read().unwrap().is_none());
        let (spv, reflected) = payload();
        first.write(&spv, &reflected).unwrap();
        drop(first);
        let second = Entry {
            key: b"exact".to_vec(),
            stage: Stage::Kernel,
            local: Some([64, 1, 1]),
            root: root.clone(),
            path,
        };
        let (loaded, _) = second.read().unwrap().unwrap();
        assert_eq!(loaded, spv);
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn disk_limits_evict_only_old_owned_entry_names() {
        let root = std::env::temp_dir().join(format!("m2v-disk-limits-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        for index in 0..3 {
            let path = root.join(format!("{index:032x}.m2v"));
            fs::write(&path, b"123").unwrap();
            OpenOptions::new()
                .write(true)
                .open(path)
                .unwrap()
                .set_times(fs::FileTimes::new().set_modified(
                    std::time::UNIX_EPOCH + std::time::Duration::from_secs(index + 1),
                ))
                .unwrap();
        }
        fs::write(root.join("unrelated.txt"), b"keep").unwrap();
        trim_with_limits(&root, 4, 2).unwrap();
        assert!(!root.join(format!("{:032x}.m2v", 0)).exists());
        assert!(!root.join(format!("{:032x}.m2v", 1)).exists());
        assert!(root.join(format!("{:032x}.m2v", 2)).exists());
        assert_eq!(fs::read(root.join("unrelated.txt")).unwrap(), b"keep");
        fs::remove_dir_all(root).unwrap();
    }
}
