use super::*;
use metal2vulkan::reflect::{BufferExtent, ResourceKind};

const FRAGMENT: &str = r#"
target datalayout = "e-p:64:64-i64:64-n8:16:32"
define {float} @declared_reach(ptr addrspace(2) %object, ptr addrspace(1) %array, ptr addrspace(1) %unknown) {
  %index = load i32, ptr addrspace(2) %object, align 4
  %address = getelementptr float, ptr addrspace(1) %array, i32 %index
  %value = load float, ptr addrspace(1) %address, align 4
  %result = insertvalue {float} undef, float %value, 0
  ret {float} %result
}
!air.fragment = !{!0}
!0 = !{ptr @declared_reach, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float"}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"Params"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1}
"#;

pub(crate) fn unbounded_readonly() -> BufferRead {
    BufferExtents {
        stage: Stage::Fragment,
        reflection: reflect_text(FRAGMENT, Stage::Fragment),
    }
    .bound(1)
    .unwrap()
}

pub(crate) fn writable_object() -> BufferRead {
    let source = FRAGMENT.replace(
        "!\"air.read\", !\"air.address_space\", i32 2",
        "!\"air.read_write\", !\"air.address_space\", i32 2",
    );
    BufferExtents {
        stage: Stage::Fragment,
        reflection: reflect_text(&source, Stage::Fragment),
    }
    .bound(0)
    .unwrap()
}

pub(crate) fn object(stage: Stage, index: u32, bytes: u32) -> BufferRead {
    let source = match stage {
        Stage::Fragment => format!(
            r#"
define {{float}} @f(ptr addrspace(2) %p) {{ ret {{float}} zeroinitializer }}
!air.fragment = !{{!0}}
!0 = !{{ptr @f, !1, !3}}
!1 = !{{!2}}
!2 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float"}}
!3 = !{{!4}}
!4 = !{{i32 0, !"air.buffer", !"air.buffer_size", i32 {bytes}, !"air.location_index", i32 {index}, i32 1, !"air.read", !"air.address_space", i32 2}}
"#
        ),
        Stage::Vertex => format!(
            r#"
define <4 x float> @v(ptr addrspace(2) %p) {{ ret <4 x float> zeroinitializer }}
!air.vertex = !{{!0}}
!0 = !{{ptr @v, !1, !3}}
!1 = !{{!2}}
!2 = !{{!"air.position", !"air.arg_type_name", !"float4"}}
!3 = !{{!4}}
!4 = !{{i32 0, !"air.buffer", !"air.buffer_size", i32 {bytes}, !"air.location_index", i32 {index}, i32 1, !"air.read", !"air.address_space", i32 2}}
"#
        ),
    };
    BufferExtents {
        stage,
        reflection: reflect_text(&source, stage),
    }
    .bound(index)
    .unwrap()
}

#[test]
fn only_air_object_extent_bounds_capture_not_dynamic_pointer_or_type_size() {
    let reflection = reflect_text(FRAGMENT, Stage::Fragment).unwrap();
    let extent = |index| {
        reflection
            .bindings
            .iter()
            .find(|binding| binding.kind == ResourceKind::Buffer && binding.metal_index == index)
            .unwrap()
            .extent
    };
    assert_eq!(extent(0), Some(BufferExtent::Object { bytes: 4 }));
    assert_eq!(extent(1), Some(BufferExtent::Unbounded));
    assert_eq!(extent(2), Some(BufferExtent::Unknown));
    let extents = BufferExtents {
        stage: Stage::Fragment,
        reflection: Ok(reflection),
    };
    let proof = extents.bound(0).unwrap();
    assert_eq!(proof.bytes_for(Class::Fragment, 0), Some(4));
    assert!(proof.bytes_for(Class::Vertex, 0).is_none());
    assert!(proof.bytes_for(Class::Fragment, 1).is_none());
    for index in [1, 2] {
        let (extent, readonly) = extents
            .bound(index)
            .unwrap()
            .capture_for(Class::Fragment, index)
            .unwrap();
        assert!(extent.is_none(), "pointee size must not narrow capture");
        assert!(
            readonly.unwrap().bytes().is_none(),
            "unbounded full capture"
        );
    }
    assert!(extents.bound(3).is_none(), "absent slot has no certificate");
    assert!(reflect_text(FRAGMENT, Stage::Vertex).is_err());
}

#[test]
fn immutable_reuse_requires_no_write_access_independently_of_reach() {
    let readonly = BufferExtents {
        stage: Stage::Fragment,
        reflection: reflect_text(FRAGMENT, Stage::Fragment),
    };
    let (bytes, proof) = readonly
        .bound(0)
        .unwrap()
        .capture_for(Class::Fragment, 0)
        .unwrap();
    assert_eq!(bytes, Some(4));
    assert_eq!(proof.unwrap().bytes(), Some(4));
    assert!(readonly
        .bound(0)
        .unwrap()
        .capture_for(Class::Vertex, 0)
        .is_none());
    assert!(
        readonly
            .bound(1)
            .unwrap()
            .capture_for(Class::Fragment, 1)
            .unwrap()
            .1
            .unwrap()
            .bytes()
            .is_none(),
        "an unbounded readonly pointer captures the full suffix"
    );
    let writable = FRAGMENT.replace(
        "!\"air.read\", !\"air.address_space\", i32 2",
        "!\"air.read_write\", !\"air.address_space\", i32 2",
    );
    let writable = BufferExtents {
        stage: Stage::Fragment,
        reflection: reflect_text(&writable, Stage::Fragment),
    };
    let (bytes, proof) = writable
        .bound(0)
        .unwrap()
        .capture_for(Class::Fragment, 0)
        .unwrap();
    assert_eq!(
        bytes,
        Some(4),
        "writability does not remove the existing capture bound"
    );
    assert!(
        proof.is_none(),
        "no readonly claim may be inferred from a bounded size"
    );
}

fn wrapped_air(source: &str) -> Vec<u8> {
    let scratch = Scratch::write(source.as_bytes(), "ll").unwrap();
    let (bitcode, _) = metal2vulkan::tools::run_with_timeout(
        "llvm-as",
        &[scratch.0.to_str().unwrap(), "-o", "-"],
        20,
    )
    .unwrap();
    let mut blob = b"MTLB-owned-metadata-test".to_vec();
    blob.extend(crate::runtime::mtlb::AIR_WRAP_MAGIC);
    for word in [0, 20, u32::try_from(bitcode.len()).unwrap(), 0] {
        blob.extend(word.to_le_bytes());
    }
    blob.extend(bitcode);
    blob
}

#[test]
fn cold_disassembly_is_cached_by_exact_bytes_and_stage_not_ref_or_pointer() {
    let cache = Mutex::new(ContentCache::new());
    let first = wrapped_air(FRAGMENT);
    let second = wrapped_air(
        &FRAGMENT.replace("!\"air.buffer_size\", i32 4", "!\"air.buffer_size\", i32 8"),
    );
    let mut reused = Vec::with_capacity(first.len().max(second.len()));
    reused.extend(&first);
    let address = reused.as_ptr();
    let key = BlobKey::new(&reused);
    let original_hash = key.hash;
    let extents = cached_with(&cache, key, Stage::Fragment, || {
        reflect_mtlb(&reused, Stage::Fragment)
    });
    assert_eq!(
        extents.bound(0).unwrap().bytes_for(Class::Fragment, 0),
        Some(4)
    );
    let copy = first.clone();
    let warm = cached_with(&cache, BlobKey::new(&copy), Stage::Fragment, || {
        panic!("a warm metadata lookup must not disassemble or reflect")
    });
    assert!(Arc::ptr_eq(&extents, &warm));
    let wrong_stage = cached_with(&cache, BlobKey::new(&copy), Stage::Vertex, || {
        reflect_mtlb(&copy, Stage::Vertex)
    });
    assert!(wrong_stage.bound(0).is_none());

    reused.clear();
    reused.extend(&second);
    assert_eq!(reused.as_ptr(), address);
    // Force a digest collision as well as reusing the source allocation.
    let changed = cached_with(
        &cache,
        BlobKey {
            hash: original_hash,
            bytes: &reused,
        },
        Stage::Fragment,
        || reflect_mtlb(&reused, Stage::Fragment),
    );
    assert!(!Arc::ptr_eq(&extents, &changed));
    assert_eq!(
        changed.bound(0).unwrap().bytes_for(Class::Fragment, 0),
        Some(8)
    );
    assert_eq!(
        extents.bound(0).unwrap().bytes_for(Class::Fragment, 0),
        Some(4)
    );
    assert_eq!(cache.lock().len(), 3);
}

#[test]
fn reflection_failure_is_cached_and_never_supplies_a_cap() {
    let cache = Mutex::new(ContentCache::new());
    let key = BlobKey::new(b"not an MTLB");
    let failed = cached_with(&cache, key, Stage::Vertex, || {
        reflect_mtlb(key.bytes, Stage::Vertex)
    });
    assert!(failed.bound(0).is_none());
    let again = cached_with(&cache, key, Stage::Vertex, || {
        panic!("failed optional reflection must not retry on every draw")
    });
    assert!(Arc::ptr_eq(&failed, &again));
}
