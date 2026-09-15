use super::*;
use crate::runtime::draw::{ColorLoadSeed, NativeColorSeed};

pub(super) fn requires_native(format: u16) -> bool {
    !crate::runtime::draw::texture_view::native_color_layout(format)
        .is_some_and(TexelLayout::is_four_byte_color)
}

pub(crate) fn load<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    span: GvaSpan,
    guest_mip_level: u32,
) -> Option<ColorLoadSeed> {
    if !requires_native(span.format) && guest_mip_level == 0 {
        return crate::runtime::draw::seed_color_load(
            state,
            host,
            task_id,
            span.texture_ref,
            span.gva,
            span.width,
            span.height,
        )
        .map(ColorLoadSeed::Rgba8);
    }
    let Some(layout) = crate::runtime::draw::texture_view::native_color_layout(span.format) else {
        crate::observe::fail(format!(
            "native_color_load_seed reason=unsupported_layout task={task_id} ref={} format={:#x}",
            span.texture_ref, span.format
        ));
        return None;
    };
    let bytes = if let Some(buffer) =
        buffer_texture_descriptor(state, host, task_id, span.texture_ref, None)
    {
        let address = objects::resolve_buffer_span(state, host, task_id, buffer.buffer_ref)
            .ok()
            .and_then(|(base, _)| base.checked_add(buffer.offset));
        if guest_mip_level != 0
            || address != Some(span.gva)
            || buffer.bytes_per_row != u64::from(span.row_stride)
        {
            crate::observe::fail(format!(
                "native_color_load_seed reason=buffer_address task={task_id} ref={} source={address:?} target={:#x} source_pitch={} target_pitch={}",
                span.texture_ref, span.gva, buffer.bytes_per_row, span.row_stride,
            ));
            return None;
        }
        let staged = match crate::runtime::compute_exec::stage_buffer_texture::<
            crate::runtime::compute_exec::vulkan::VulkanStage,
            _,
        >(state, host, task_id, span.texture_ref, 0, false, &buffer)
        {
            Ok(staged) => staged,
            Err(error) => {
                crate::observe::fail(format!("native_color_load_seed reason=buffer_stage task={task_id} ref={} detail={error:?}", span.texture_ref));
                return None;
            }
        };
        if staged.width != span.width
            || staged.height != span.height
            || crate::runtime::draw::texture_view::native_color_layout(staged.pixel_format)
                != Some(layout)
        {
            crate::observe::fail(format!("native_color_load_seed reason=buffer_layout task={task_id} ref={} source={}x{}:{:#x} target={}x{}:{:#x}",
                span.texture_ref, staged.width, staged.height, staged.pixel_format,
                span.width, span.height, span.format));
            return None;
        }
        staged.bytes
    } else {
        let (reference, format) = resolve_texture_view(state, host, task_id, span.texture_ref)
            .map(|view| (view.base_texture_ref, view.pixel_format))
            .unwrap_or((span.texture_ref, None));
        match crate::runtime::draw::texture_view::load_linear_color_native(
            state,
            host,
            task_id,
            reference,
            guest_mip_level,
            format,
            (
                layout,
                span.width,
                span.height,
                span.gva,
                u64::from(span.row_stride),
            ),
        ) {
            Ok((bytes, actual)) if actual.layout() == layout => bytes,
            Ok(_) => {
                crate::observe::fail(format!(
                    "native_color_load_seed reason=layout_mismatch task={task_id} ref={}",
                    span.texture_ref
                ));
                return None;
            }
            Err(reason) => {
                crate::observe::Emit::decline("native_color_load_seed", &reason)
                    .field("task", task_id)
                    .field("ref", span.texture_ref)
                    .fail();
                return None;
            }
        }
    };
    crate::runtime::drain::note_store_route("load_seed_color_native");
    Some(ColorLoadSeed::Native(NativeColorSeed {
        layout,
        bytes: std::sync::Arc::new(bytes),
    }))
}

#[cfg(test)]
mod tests;
