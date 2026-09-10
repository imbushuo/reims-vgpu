import Metal
import Foundation

private let fragmentSnapshotSource = """
#include <metal_stdlib>
using namespace metal;
fragment void snapshot_save(float4 p [[position]], half4 destination [[color(0)]],
    texture2d<half, access::write> snapshot [[texture(0), raster_order_group(0)]]) {
    snapshot.write(destination, uint2(p.xy));
}
fragment half4 snapshot_fill() { return half4(0.5h, 0.5h, 0.5h, 1.0h); }
fragment half4 snapshot_restore(float4 p [[position]], half4 destination [[color(0)]],
    texture2d<half, access::read> snapshot [[texture(0)]]) {
    float2 q = max(abs(p.xy - 16.0f) - 8.0f, 0.0f);
    return length(q) < 8.0f ? destination : snapshot.read(uint2(p.xy));
}
fragment half4 snapshot_sample(float4 p [[position]],
    texture2d<half, access::read> image [[texture(0)]]) {
    return image.read(uint2(p.xy));
}
"""

private func fragmentSnapshotCase(_ format: MTLPixelFormat, separateCommands: Bool) throws {
    let formatName = format == .rgba16Float ? "rgba16float" : "bgra8unorm"
    let label = "fragment_texture_snapshot_\(formatName)_\(separateCommands ? "separate" : "shared")"
    let shaders = try dev.makeLibrary(source: fragmentSnapshotSource, options: nil)
    func pipeline(_ name: String, _ format: MTLPixelFormat) throws -> MTLRenderPipelineState {
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "quad_vs")
        descriptor.fragmentFunction = shaders.makeFunction(name: name)
        descriptor.colorAttachments[0].pixelFormat = format
        return try dev.makeRenderPipelineState(descriptor: descriptor)
    }
    func texture(_ format: MTLPixelFormat) -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: 32, height: 32, mipmapped: false)
        descriptor.storageMode = .private
        descriptor.usage = [.renderTarget, .shaderRead, .shaderWrite]
        return dev.makeTexture(descriptor: descriptor)!
    }
    let layer = texture(format), snapshot = texture(format), output = texture(.bgra8Unorm)
    let save = try pipeline("snapshot_save", format)
    let fill = try pipeline("snapshot_fill", format)
    let restore = try pipeline("snapshot_restore", format)
    let sample = try pipeline("snapshot_sample", .bgra8Unorm)
    let command = queue.makeCommandBuffer()!
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = layer
    pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 1, 1, 1)
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].storeAction = .store
    let encoder = command.makeRenderCommandEncoder(descriptor: pass)!
    encoder.setVertexBytes(quadVerts, length: quadVerts.count * 4, index: 0)
    encoder.setFragmentTexture(snapshot, index: 0)
    // The first draw has no color output: its only result is the writable texture.
    for pipeline in [save, fill, restore] {
        encoder.setRenderPipelineState(pipeline)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
    encoder.endEncoding()
    if separateCommands {
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            report(label, false, "snapshot command failed"); return
        }
    }
    let final = separateCommands ? queue.makeCommandBuffer()! : command
    let display = MTLRenderPassDescriptor()
    display.colorAttachments[0].texture = output
    display.colorAttachments[0].loadAction = .clear
    display.colorAttachments[0].storeAction = .store
    let screen = final.makeRenderCommandEncoder(descriptor: display)!
    screen.setVertexBytes(quadVerts, length: quadVerts.count * 4, index: 0)
    screen.setRenderPipelineState(sample)
    screen.setFragmentTexture(layer, index: 0)
    screen.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    screen.endEncoding()
    final.commit()
    final.waitUntilCompleted()
    guard final.status == .completed,
          let pixels = readBack(readPipe, output, 32, 32) else {
        report(label, false, "final command/readback failed"); return
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

func fragmentTextureWriteCases() {
    for format in [MTLPixelFormat.bgra8Unorm, .rgba16Float] {
        for separateCommands in [false, true] {
            do {
                try fragmentSnapshotCase(format, separateCommands: separateCommands)
            } catch {
                report("fragment_texture_snapshot_setup", false, "\(error)")
            }
        }
    }
}
