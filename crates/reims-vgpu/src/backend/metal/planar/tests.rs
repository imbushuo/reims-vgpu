use super::*;
use crate::protocol::endian::{st16, st32, st64};
use crate::protocol::planar::{self, BackingFormat, SampleFormat};
use metal::{CompileOptions, MTLResourceOptions, MTLSize};
use objc::rc::WeakPtr;

pub(crate) fn device_descriptor(backing: BackingFormat) -> [u8; 512] {
    let mut b = [0; 512];
    st32(&mut b[4..], backing.word());
    st32(&mut b[16..], 4096);
    st64(&mut b[20..], 0x80 | (8 << 8) | (0x80 << 32) | (4 << 40));
    st32(&mut b[28..], 64);
    st16(&mut b[32..], 1);
    b[36] = 2;
    for (i, (w, h, bpe, base, names)) in [
        (8, 4, 2, 0, &[5][..]), (4, 2, 4, 2048, &[7, 6][..]),
    ].into_iter().enumerate() {
        let p = &mut b[64 + i * 64..128 + i * 64];
        st32(&mut p[8..], base + 128);
        st32(&mut p[12..], base);
        st32(&mut p[16..], 1024);
        st64(&mut p[20..], 0x80 | (w << 8) | (0x80 << 32) | (h << 40));
        st32(&mut p[28..], 64);
        st16(&mut p[32..], bpe);
        p[planar::PLANE_COMPONENT_COUNT] = names.len() as u8;
        p[planar::PLANE_EXTENDED_PIXELS..planar::PLANE_EXTENDED_PIXELS + 4].fill(1);
        for (c, &name) in names.iter().enumerate() {
            p[planar::PLANE_COMPONENT_DEPTHS + c] = 10;
            p[planar::PLANE_COMPONENT_NAMES + c] = name;
            p[planar::PLANE_COMPONENT_RANGES + c] = backing.component_range();
        }
    }
    b
}

pub(crate) fn texture_descriptor(format: SampleFormat) -> [u8; 56] {
    let mut b = [0; 56];
    st32(&mut b, 5);
    st32(&mut b[8..], 0x2f);
    st32(&mut b[12..], reims_vgpu_wire::ops::backed_texture::IOSURFACE_TEXTURE_TOTAL_LEN);
    st32(&mut b[16..], 11);
    st32(&mut b[20..], (u32::from(format.word()) << 16) | 0x0142);
    st32(&mut b[24..], 8);
    st32(&mut b[28..], 4);
    st32(&mut b[32..], 1);
    st16(&mut b[36..], 1);
    st16(&mut b[38..], 1);
    st16(&mut b[40..], 1);
    st16(&mut b[42..], 0x10);
    b
}

pub(crate) fn image(format: SampleFormat, backing: BackingFormat) -> SampledImage {
    let layout = Layout::decode(&device_descriptor(backing), 8, 4).unwrap();
    let description = TextureDescription::decode(&texture_descriptor(format), 11).unwrap();
    let mut planes = [vec![0; 1024], vec![0; 1024]];
    for word in planes[0].chunks_exact_mut(2) { word.copy_from_slice(&(512u16 << 6).to_le_bytes()); }
    for pair in planes[1].chunks_exact_mut(4) {
        pair[..2].copy_from_slice(&(640u16 << 6).to_le_bytes());
        pair[2..].copy_from_slice(&(768u16 << 6).to_le_bytes());
    }
    SampledImage::new(description, layout, planes).unwrap()
}

#[test]
fn planar_native_samples_both_private_formats_and_ranges() {
    objc::rc::autoreleasepool(|| {
        let device = metal::Device::system_default().expect("native Metal device");
        let source = "#include <metal_stdlib>\nusing namespace metal;\n\
            kernel void planar_read(texture2d<float, access::sample> t [[texture(0)]], \
            device float4 *out [[buffer(0)]], uint2 p [[thread_position_in_grid]]) { \
            out[p.y*t.get_width()+p.x]=t.read(p); }";
        let library = device.new_library_with_source(source, &CompileOptions::new()).unwrap();
        let function = library.get_function("planar_read", None).unwrap();
        let pipeline = device.new_compute_pipeline_state_with_function(&function).unwrap();
        let queue = device.new_command_queue();
        for format in [SampleFormat::Ycbcr10_420TwoPlane, SampleFormat::Rgb10_420TwoPlane] {
            for backing in [BackingFormat::VideoRange, BackingFormat::FullRange] {
                let image = image(format, backing);
                let texture = upload(&device, &image).unwrap();
                let ordinal: u64 = unsafe { msg_send![texture, pixelFormat] };
                assert_eq!(ordinal, u64::from(format.word()));
                let output = device.new_buffer(8 * 4 * 16, MTLResourceOptions::StorageModeShared);
                let command = queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_texture(0, Some(&texture));
                encoder.set_buffer(0, Some(&output), 0);
                encoder.dispatch_threads(MTLSize::new(8, 4, 1), MTLSize::new(8, 4, 1));
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
                let expected = if format == SampleFormat::Rgb10_420TwoPlane {
                    [768.0 / 1023.0, 512.0 / 1023.0, 640.0 / 1023.0, 1.0]
                } else {
                    let (y, u, v) = if backing == BackingFormat::VideoRange {
                        (448.0f32 / 876.0, 128.0 / 896.0, 256.0 / 896.0)
                    } else {
                        (512.0f32 / 1023.0, 128.0 / 1023.0, 256.0 / 1023.0)
                    };
                    [y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u, 1.0]
                };
                // SAFETY: the completed command wrote this entire shared buffer.
                let values = unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), 128) };
                for pixel in values.chunks_exact(4) {
                    for (actual, expected) in pixel.iter().zip(expected) {
                        assert!((actual - expected).abs() <= 0.001, "{format:?} {backing:?}: {pixel:?}");
                    }
                }
            }
        }
    });
}

#[test]
fn planar_texture_retains_iosurface_beyond_upload_pool() {
    objc::rc::autoreleasepool(|| {
        let device = metal::Device::system_default().expect("native Metal device");
        let image = image(SampleFormat::Rgb10_420TwoPlane, BackingFormat::VideoRange);
        let texture = upload(&device, &image).unwrap();
        let surface: *mut Object = unsafe { msg_send![texture, iosurface] };
        assert!(!surface.is_null());
        // SAFETY: the texture's native IOSurface getter returns its live object.
        let weak = unsafe { WeakPtr::new(surface) };
        assert!(!weak.load().is_null(), "the upload pool and creator reference have drained");
        drop(texture);
        assert!(weak.load().is_null(), "no autoreleased texture may keep the surface alive");
    });
}
