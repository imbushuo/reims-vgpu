import Foundation
import Metal

private enum CornerSnapshot: String, CaseIterable {
    case none, blit, fragment
}

private func cornerEncoderSource(suffix: String) -> String {
    let renames = ["layer_fill", "layer_clip", "layer_composite", "layer_restore", "layer_save"]
        .map { "#define \($0) \($0)_\(suffix)" }.joined(separator: "\n")
    return renames + "\n" + """
    #include <metal_stdlib>
    using namespace metal;
    fragment half4 layer_fill() { return half4(0.5h, 0.5h, 0.5h, 1.0h); }
    fragment half4 layer_clip(float4 p [[position]], half4 destination [[color(0)]]) {
        float2 q = abs(p.xy - 16.0f) - 8.0f;
        float distance = length(max(q, 0.0f)) - 8.0f;
        return distance < 0.0f ? destination : half4(0.0h);
    }
    fragment half4 layer_composite(float4 p [[position]],
        texture2d<half, access::read> image [[texture(0)]]) {
        return image.read(uint2(p.xy));
    }
    fragment half4 layer_restore(float4 p [[position]], half4 destination [[color(0)]],
        texture2d<half, access::read> image [[texture(0)]]) {
        float2 q = abs(p.xy - 16.0f) - 8.0f;
        return length(max(q, 0.0f)) < 8.0f ? destination : image.read(uint2(p.xy));
    }
    fragment void layer_save(float4 p [[position]], half4 destination [[color(0)]],
        texture2d<half, access::write> image [[texture(0), raster_order_group(0)]]) {
        image.write(destination, uint2(p.xy));
    }
    """
}

private func cornerEncoderCase(source: String, suffix: String, temperature: String,
                               separateCommands: Bool, format: MTLPixelFormat,
                               formatName: String, snapshotMode: CornerSnapshot) throws {
    let label = "corner_encoders_\(temperature)_\(snapshotMode.rawValue)_"
        + "\(separateCommands ? "separate" : "shared")_\(formatName)"
    let shaders = try dev.makeLibrary(source: source, options: nil)
    func pipeline(_ name: String, _ format: MTLPixelFormat,
                  blend: Bool = false) throws -> MTLRenderPipelineState {
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "quad_vs")
        descriptor.fragmentFunction = shaders.makeFunction(name: "\(name)_\(suffix)")
        let color = descriptor.colorAttachments[0]!
        color.pixelFormat = format
        color.isBlendingEnabled = blend
        color.sourceRGBBlendFactor = .one
        color.sourceAlphaBlendFactor = .one
        color.destinationRGBBlendFactor = .oneMinusSourceAlpha
        color.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        return try dev.makeRenderPipelineState(descriptor: descriptor)
    }
    func texture(_ format: MTLPixelFormat) -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: 32, height: 32, mipmapped: false)
        descriptor.storageMode = .private
        descriptor.usage = [.renderTarget, .shaderRead, .shaderWrite]
        return dev.makeTexture(descriptor: descriptor)!
    }
    func whitePass(_ target: MTLTexture) -> MTLRenderPassDescriptor {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 1, 1, 1)
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        return pass
    }
    func complete(_ command: MTLCommandBuffer) -> Bool {
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            report(label, false, "command failed \(String(describing: command.error))")
            return false
        }
        return true
    }

    let layer = texture(format), output = texture(.bgra8Unorm), snapshot = texture(format)
    let fill = try pipeline("layer_fill", format)
    let clip = try pipeline(snapshotMode == .none ? "layer_clip" : "layer_restore", format)
    let composite = try pipeline("layer_composite", .bgra8Unorm, blend: snapshotMode == .none)
    let save = try pipeline("layer_save", format)
    var first = queue.makeCommandBuffer()!
    if snapshotMode == .blit {
        first.makeRenderCommandEncoder(descriptor: whitePass(layer))!.endEncoding()
        let blit = first.makeBlitCommandEncoder()!
        blit.copy(from: layer, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: 32, height: 32, depth: 1),
                  to: snapshot, destinationSlice: 0, destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        if separateCommands {
            guard complete(first) else { return }
            first = queue.makeCommandBuffer()!
        }
    }
    let encoder = first.makeRenderCommandEncoder(descriptor: whitePass(layer))!
    encoder.setVertexBytes(quadVerts, length: quadVerts.count * 4, index: 0)
    if snapshotMode == .fragment {
        encoder.setRenderPipelineState(save)
        encoder.setFragmentTexture(snapshot, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
    encoder.setRenderPipelineState(fill)
    encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    encoder.setRenderPipelineState(clip)
    if snapshotMode != .none { encoder.setFragmentTexture(snapshot, index: 0) }
    encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    encoder.endEncoding()
    if separateCommands {
        guard complete(first) else { return }
    }
    let second = separateCommands ? queue.makeCommandBuffer()! : first
    let screen = second.makeRenderCommandEncoder(descriptor: whitePass(output))!
    screen.setVertexBytes(quadVerts, length: quadVerts.count * 4, index: 0)
    screen.setRenderPipelineState(composite)
    screen.setFragmentTexture(layer, index: 0)
    screen.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    screen.endEncoding()
    guard complete(second) else { return }
    guard let pixels = readBack(readPipe, output, 32, 32) else {
        report(label, false, "readback failed")
        return
    }
    var wrong = 0
    for y in 0..<32 {
        for x in 0..<32 {
            let qx = max(abs(Float(x) + 0.5 - 16) - 8, 0)
            let qy = max(abs(Float(y) + 0.5 - 16) - 8, 0)
            let expected = (qx * qx + qy * qy).squareRoot() < 8 ? 128 : 255
            let pixel = pixels[y * 32 + x]
            if (0..<3).contains(where: { abs(Int((pixel >> ($0 * 8)) & 255) - expected) > 1 })
                || pixel >> 24 != 255 {
                wrong += 1
            }
        }
    }
    report(label, wrong == 0,
           "wrong=\(wrong)/1024 corner=\(hex(pixels[0])) center=\(hex(pixels[16 * 32 + 16]))")
}

/// Destroy/recreate pipeline states between cases, then repeat with warm shader translations.
/// Fresh function names keep repeated suite invocations cold without changing shader operations.
func cornerEncoderReadinessCases() {
    let suffix = UUID().uuidString.replacingOccurrences(of: "-", with: "")
    let source = cornerEncoderSource(suffix: suffix)
    do {
        for temperature in ["cold", "warm"] {
            for (format, name) in [(MTLPixelFormat.bgra8Unorm, "bgra8unorm"),
                                   (.rgba16Float, "rgba16float")] {
                for snapshot in CornerSnapshot.allCases {
                    for separate in [true, false] {
                        try cornerEncoderCase(source: source, suffix: suffix,
                                              temperature: temperature,
                                              separateCommands: separate, format: format,
                                              formatName: name, snapshotMode: snapshot)
                    }
                }
            }
        }
    } catch {
        report("corner_encoder_readiness_setup", false, "\(error)")
    }
}
