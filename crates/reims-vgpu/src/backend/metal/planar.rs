//! Native composite P010 textures shared by fragment and compute bindings.
//!
//! No color conversion, repacking, or chroma reconstruction occurs here.
//! IOSurface metadata and the private Metal ordinal jointly define those.

use super::error::Status;
use crate::protocol::planar::{Layout, Refusal, TextureDescription};
use foreign_types::ForeignType;
use metal::{DeviceRef, MTLStorageMode, MTLTextureType, MTLTextureUsage, Texture, TextureDescriptor};
use objc::{class, msg_send, sel, sel_impl};
use objc::runtime::{Object, BOOL, NO, YES};
use std::ffi::{c_void, CStr};
use std::fmt;

pub(crate) struct SampledImage {
    pub(crate) description: TextureDescription,
    pub(crate) layout: Layout,
    planes: [Vec<u8>; 2],
}

impl fmt::Debug for SampledImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlanarSampledImage")
            .field("description", &self.description)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl SampledImage {
    #[cfg(test)]
    pub(crate) fn plane_bytes(&self, index: usize) -> &[u8] { &self.planes[index] }

    pub(crate) fn new(
        description: TextureDescription,
        layout: Layout,
        planes: [Vec<u8>; 2],
    ) -> Result<Self, Refusal> {
        if description.width != layout.width || description.height != layout.height {
            return Err(Refusal::Extent);
        }
        if planes.iter().zip(&layout.planes).any(|(bytes, plane)| bytes.len() as u64 != plane.size) {
            return Err(Refusal::ImageBytes);
        }
        Ok(Self { description, layout, planes })
    }
}

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceCreate(properties: *const Object) -> *mut c_void;
    fn IOSurfaceLock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
    fn IOSurfaceGetAllocSize(surface: *mut c_void) -> usize;
    fn IOSurfaceGetPlaneCount(surface: *mut c_void) -> usize;
    fn IOSurfaceGetBaseAddressOfPlane(surface: *mut c_void, plane: usize) -> *mut c_void;
    fn IOSurfaceGetWidthOfPlane(surface: *mut c_void, plane: usize) -> usize;
    fn IOSurfaceGetHeightOfPlane(surface: *mut c_void, plane: usize) -> usize;
    fn IOSurfaceGetBytesPerRowOfPlane(surface: *mut c_void, plane: usize) -> usize;
    fn IOSurfaceGetBytesPerElementOfPlane(surface: *mut c_void, plane: usize) -> usize;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(object: *const c_void);
}

struct Surface(*mut c_void);

impl Drop for Surface {
    fn drop(&mut self) {
        // SAFETY: `IOSurfaceCreate` returned this owned reference.
        unsafe { CFRelease(self.0) };
    }
}

struct LockedSurface<'a> {
    surface: &'a Surface,
    active: bool,
}

impl LockedSurface<'_> {
    fn unlock(mut self) -> Result<(), Status> {
        self.active = false;
        // SAFETY: this guard owns the successful lock of this live surface.
        let result = unsafe { IOSurfaceUnlock(self.surface.0, 0, std::ptr::null_mut()) };
        if result == 0 { Ok(()) } else {
            Err(Status::execute("metal_planar_surface_unlock").field("ioreturn", result as u32))
        }
    }
}

impl Drop for LockedSurface<'_> {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: unwind cleanup of a successful lock.
            let result = unsafe { IOSurfaceUnlock(self.surface.0, 0, std::ptr::null_mut()) };
            if result != 0 {
                crate::observe::Emit::refusal(
                    "metal_planar_cleanup",
                    &Status::execute("metal_planar_surface_unlock").field("ioreturn", result as u32),
                ).expect("unlock failure is a refusal").fail();
            }
        }
    }
}

fn dictionary() -> *mut Object {
    // The caller's autorelease pool owns these construction-only objects.
    unsafe { msg_send![class!(NSMutableDictionary), dictionary] }
}

fn array() -> *mut Object {
    unsafe { msg_send![class!(NSMutableArray), array] }
}

fn number(value: u64) -> *mut Object {
    unsafe { msg_send![class!(NSNumber), numberWithUnsignedLongLong: value] }
}

fn key(value: &CStr) -> *mut Object {
    unsafe { msg_send![class!(NSString), stringWithUTF8String: value.as_ptr()] }
}

fn put(dict: *mut Object, name: &CStr, value: u64) {
    put_object(dict, name, number(value));
}

fn put_object(dict: *mut Object, name: &CStr, value: *mut Object) {
    unsafe {
        let _: () = msg_send![dict, setObject: value forKey: key(name)];
    }
}

fn append(array: *mut Object, value: *mut Object) {
    unsafe { let _: () = msg_send![array, addObject: value]; }
}

fn put_components(dict: *mut Object, name: &CStr, values: &[u8]) {
    let a = array();
    for &value in values { append(a, number(u64::from(value))); }
    put_object(dict, name, a);
}

fn properties(image: &SampledImage) -> *mut Object {
    let layout = &image.layout;
    let d = dictionary();
    put(d, c"IOSurfaceWidth", layout.width.into());
    put(d, c"IOSurfaceHeight", layout.height.into());
    put(d, c"IOSurfacePixelFormat", layout.backing_format.word().into());
    put(d, c"IOSurfaceAllocSize", layout.allocation_size);
    put(d, c"IOSurfaceBytesPerRow", layout.bytes_per_row.into());
    put(d, c"IOSurfaceBytesPerElement", 1);
    put(d, c"IOSurfaceElementWidth", 1);
    put(d, c"IOSurfaceElementHeight", 1);
    let planes = array();
    for (index, p) in layout.planes.iter().enumerate() {
        let pd = dictionary();
        put(pd, c"IOSurfacePlaneBase", p.base);
        put(pd, c"IOSurfacePlaneOffset", p.offset);
        put(pd, c"IOSurfacePlaneSize", p.size);
        put(pd, c"IOSurfacePlaneWidth", p.width.into());
        put(pd, c"IOSurfacePlaneHeight", p.height.into());
        put(pd, c"IOSurfacePlaneBytesPerRow", p.bytes_per_row.into());
        put(pd, c"IOSurfacePlaneBytesPerElement", p.bytes_per_element.into());
        put(pd, c"IOSurfacePlaneElementWidth", 1);
        put(pd, c"IOSurfacePlaneElementHeight", 1);
        put(pd, c"IOSurfacePlaneCompressionType", 0);
        put(pd, c"IOSurfacePlaneCompressionFootprint", 0);
        put(pd, c"IOSurfaceAddressFormat", 0);
        put(pd, c"IOSurfacePlaneExtendedPixelsOnLeft", p.extended.left.into());
        put(pd, c"IOSurfacePlaneExtendedPixelsOnTop", p.extended.top.into());
        put(pd, c"IOSurfacePlaneExtendedPixelsOnRight", p.extended.right.into());
        put(pd, c"IOSurfacePlaneExtendedPixelsOnBottom", p.extended.bottom.into());
        let names: &[u8] = if index == 0 { &[5] } else { &[7, 6] };
        put_components(pd, c"IOSurfacePlaneComponentNames", names);
        put_components(pd, c"IOSurfacePlaneComponentTypes", &vec![0; names.len()]);
        put_components(pd, c"IOSurfacePlaneComponentBitDepths", &vec![10; names.len()]);
        put_components(pd, c"IOSurfacePlaneComponentBitOffsets", &vec![0; names.len()]);
        put_components(
            pd, c"IOSurfacePlaneComponentRanges",
            &vec![layout.backing_format.component_range(); names.len()],
        );
        append(planes, pd);
    }
    put_object(d, c"IOSurfacePlaneInfo", planes);
    d
}

/// Upload a fully validated whole surface, retaining the IOSurface through the
/// returned Metal texture. The native ownership regression checks this retain.
pub(crate) fn upload(device: &DeviceRef, image: &SampledImage) -> Result<Texture, Status> {
    objc::rc::autoreleasepool(|| upload_pooled(device, image))
}

fn upload_pooled(device: &DeviceRef, image: &SampledImage) -> Result<Texture, Status> {
    // SAFETY: the property dictionary lives through this synchronous create.
    let pointer = unsafe { IOSurfaceCreate(properties(image)) };
    if pointer.is_null() {
        return Err(Status::execute("metal_planar_surface_create"));
    }
    let surface = Surface(pointer);
    // SAFETY: these queries borrow the live owned IOSurface.
    let base = unsafe { IOSurfaceGetBaseAddress(surface.0) }.cast::<u8>();
    let len = unsafe { IOSurfaceGetAllocSize(surface.0) };
    if base.is_null() || len < image.layout.allocation_size as usize
        || unsafe { IOSurfaceGetPlaneCount(surface.0) } != 2
    {
        return Err(Status::execute("metal_planar_surface_geometry"));
    }
    for (index, p) in image.layout.planes.iter().enumerate() {
        let address = unsafe { IOSurfaceGetBaseAddressOfPlane(surface.0, index) } as usize;
        if address.checked_sub(base as usize) != Some(p.offset as usize)
            || unsafe { IOSurfaceGetWidthOfPlane(surface.0, index) } != p.width as usize
            || unsafe { IOSurfaceGetHeightOfPlane(surface.0, index) } != p.height as usize
            || unsafe { IOSurfaceGetBytesPerRowOfPlane(surface.0, index) } != p.bytes_per_row as usize
            || unsafe { IOSurfaceGetBytesPerElementOfPlane(surface.0, index) } != p.bytes_per_element as usize
            || p.end() > len as u64
        {
            return Err(Status::execute("metal_planar_plane_geometry").field("plane", index));
        }
    }
    if unsafe { IOSurfaceLock(surface.0, 0, std::ptr::null_mut()) } != 0 {
        return Err(Status::execute("metal_planar_surface_lock"));
    }
    {
        let lock = LockedSurface { surface: &surface, active: true };
        for (p, bytes) in image.layout.planes.iter().zip(&image.planes) {
            // SAFETY: the immutable image owns exactly p.size bytes; the native
            // allocation and both plane ends were checked above. Copying the
            // entire plane preserves extended pixels and row/leading padding.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(p.base as usize), bytes.len());
            }
        }
        lock.unlock()?;
    }
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_width(image.layout.width.into());
    descriptor.set_height(image.layout.height.into());
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(MTLTextureUsage::ShaderRead);
    // The metal crate's public pixel-format enum excludes these private
    // ordinals. Passing an integer to the native setter avoids an invalid Rust
    // enum, and callers must not call TextureRef::pixel_format on this texture.
    unsafe {
        let _: () = msg_send![descriptor,
            setPixelFormat: u64::from(image.description.format.word())];
        let allow = if image.description.allow_gpu_optimized_contents { YES } else { NO };
        let _: () = msg_send![descriptor, setAllowGPUOptimizedContents: allow];
        let available: BOOL = msg_send![descriptor,
            respondsToSelector: sel!(colorSpaceConversionMatrix)];
        if available == NO {
            return Err(Status::execute("metal_planar_matrix_unavailable"));
        }
        let matrix: u64 = msg_send![descriptor, colorSpaceConversionMatrix];
        if matrix != 0 {
            return Err(Status::execute("metal_planar_matrix_nondefault").field("matrix", matrix));
        }
        let texture: *mut metal::MTLTexture = msg_send![device,
            newTextureWithDescriptor: descriptor.as_ref()
            iosurface: surface.0
            plane: 0u64];
        if texture.is_null() {
            return Err(Status::execute("metal_planar_texture_create")
                .field("format", image.description.format.word()));
        }
        // SAFETY: newTexture returns an owned object and retains its IOSurface.
        Ok(Texture::from_ptr(texture))
    }
}

#[cfg(test)]
pub(crate) mod tests;
