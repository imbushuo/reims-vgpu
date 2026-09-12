use super::*;
use crate::backend::metal::input::{self, Class, FillError};
use crate::backend::metal::render::NativeBuffer;

pub(crate) enum PreparedInput {
    Cpu(Vec<u8>),
    Native(NativeBuffer),
}

pub(super) struct InputPlan {
    pub native_plain: bool,
    stage_in: std::collections::BTreeSet<u32>,
}

impl InputPlan {
    pub fn new(req: &DrawEncodeRequest, pipeline: &RenderPipelineDescriptor) -> Self {
        Self {
            native_plain: plain_input_draw(req),
            stage_in: pipeline
                .vertex_attributes
                .iter()
                .filter(|a| a.format != 0 && a.stride != 0)
                .map(|a| a.buffer_index)
                .collect(),
        }
    }

    pub fn native_vertex(&self, index: u32) -> bool {
        self.native_plain && !self.stage_in.contains(&index)
    }
}

/// Texture access/native-output reflection occurs after the existing buffer
/// reads. Keep those draws on the CPU route rather than moving reads across
/// their preflight/materialization. Attribute CPU views are excluded per bind.
pub(super) fn plain_input_draw(req: &DrawEncodeRequest) -> bool {
    req.visibility.is_none()
        && req.depth_attach.is_none()
        && req.stencil_attach.is_none()
        && req
            .vertex_textures
            .iter()
            .chain(req.fragment_textures.iter())
            .all(|bind| bind.texture_ref == 0)
}

pub(crate) fn prepare<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task: u32,
    bind: &BufferBind,
    class: Class,
    native: bool,
    miss_reason: &'static str,
) -> Result<PreparedInput, EncodeStatus> {
    let window = prepare_bound_buffer_read(state, host, task, bind)
        .ok_or(EncodeStatus::MetalFailed(miss_reason))?;
    if !native {
        return window
            .read_vec()
            .map(PreparedInput::Cpu)
            .ok_or(EncodeStatus::MetalFailed(miss_reason));
    }
    let device = crate::backend::metal::runtime::system_device().ok_or_else(|| {
        EncodeStatus::RailRefused(crate::backend::metal::util::Status::execute(
            "metal_render_device_unavailable",
        ))
    })?;
    // Preparation paid reference debt and settled overlapping GPU work. The
    // synchronous fill owns no pool/TLS borrow, HostOps or guest-memory alias.
    let bytes = input::fill(
        device,
        window.len,
        class,
        "metal_render_buffer_create_failed",
        |bytes| window.read_into(bytes).map(|()| bytes.len()),
    )
    .map_err(|error| match error {
        FillError::Backend(status) => EncodeStatus::RailRefused(status),
        FillError::Callback(error) => {
            crate::observe::Emit::decline("metal_draw_buffer_read", &error)
                .field("task", task)
                .field("ref", bind.buffer_ref)
                .field("index", bind.index)
                .fail();
            EncodeStatus::MetalFailed(miss_reason)
        }
    })?;
    debug_assert_eq!(bytes.len(), window.len);
    Ok(PreparedInput::Native(NativeBuffer {
        binding: bind.index,
        attribute_stride: bind.attribute_stride,
        bytes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_fill_keeps_query_depth_and_texture_participants_on_cpu_route() {
        let mut request = DrawEncodeRequest::default();
        assert!(plain_input_draw(&request));
        std::sync::Arc::make_mut(&mut request.fragment_textures).push(TextureBind {
            texture_ref: 7,
            ..Default::default()
        });
        assert!(
            !plain_input_draw(&request),
            "access is not reflected before buffer reads"
        );
        std::sync::Arc::make_mut(&mut request.fragment_textures)[0].texture_ref = 0;
        assert!(
            plain_input_draw(&request),
            "an unbound slot is not a texture participant"
        );
        request.depth_attach = Some(Default::default());
        assert!(!plain_input_draw(&request));
        request.depth_attach = None;
        request.stencil_attach = Some(Default::default());
        assert!(!plain_input_draw(&request));
        request.stencil_attach = None;
        request.visibility = Some(VisibilityArming { mode: 1, offset: 0 });
        assert!(!plain_input_draw(&request));
    }

    #[test]
    fn direct_fill_excludes_every_binding_that_supplies_an_attribute_cpu_view() {
        use crate::runtime::decode::resource::VertexAttribute;
        let pipeline = RenderPipelineDescriptor {
            vertex_attributes: vec![
                VertexAttribute {
                    buffer_index: 2,
                    format: 29,
                    stride: 8,
                    ..Default::default()
                },
                VertexAttribute {
                    buffer_index: 4,
                    format: 0,
                    stride: 8,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let plan = InputPlan::new(&DrawEncodeRequest::default(), &pipeline);
        assert!(
            plan.native_plain,
            "plain fragment input is independently eligible"
        );
        assert!(
            !plan.native_vertex(2),
            "attribute and storage uses must share the CPU snapshot"
        );
        assert!(plan.native_vertex(1));
        assert!(
            plan.native_vertex(4),
            "absent attribute format does not create a CPU view"
        );

        let mut fixture =
            crate::runtime::draw::buffer_read_tests::Fixture::new(crate::model::PAGE_SHIFT_ARM64E);
        let mut bind = fixture.bind(7, 1, 32, 8);
        bind.index = 2;
        let input = prepare(
            &mut fixture.state,
            &mut fixture.host,
            1,
            &bind,
            Class::Vertex,
            plan.native_vertex(bind.index),
            "draw_mtl_vertex_buffer_miss",
        )
        .unwrap();
        let PreparedInput::Cpu(bytes) = input else {
            panic!("attribute source escaped its CPU owner")
        };
        assert_eq!(bytes, [0x11; 24]);
    }
}
