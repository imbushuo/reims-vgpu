import Metal
import Foundation
import IOSurface

// A8 sampling does not turn an R8 attachment into an alpha attachment.
// Render and sample the same IOSurface through separate R8/A8 texture objects.
// Keep the raw byte, R8 sample, and A8 sample as independent observations.
func alphaSurfaceRenderCases(_ w: Int, _ h: Int) {
    let source = """
    #include <metal_stdlib>
    using namespace metal;
    vertex float4 alpha_surface_vs(uint id [[vertex_id]]) {
        const float2 p[3] = { float2(-1, -1), float2(3, -1), float2(-1, 3) };
        return float4(p[id], 0, 1);
    }
    fragment float4 alpha_surface_write(float4 p [[position]]) {
        uint2 xy = uint2(p.xy);
        float r = float(64u + ((xy.x * 7u + xy.y * 13u) % 128u)) / 255.0f;
        return float4(r, 15.0f / 255.0f, 231.0f / 255.0f, 213.0f / 255.0f);
    }
    fragment float4 alpha_surface_read(float4 p [[position]],
        texture2d<float> tex [[texture(0)]]) {
        return tex.read(uint2(p.xy));
    }
    fragment float4 alpha_surface_sample(float4 p [[position]],
        texture2d<float> tex [[texture(0)]]) {
        constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest);
        return tex.sample(s, p.xy / float2(tex.get_width(), tex.get_height()));
    }
    fragment float4 alpha_surface_paired_read(float4 p [[position]],
        texture2d<float> first [[texture(0)]], texture2d<float> second [[texture(1)]]) {
        float4 a = first.read(uint2(p.xy)), b = second.read(uint2(p.xy));
        bool red_first = a.a == 1.0f && all(a.gb == float2(0)) &&
            all(b.rgb == float3(0)) && a.r == b.a;
        bool alpha_first = b.a == 1.0f && all(b.gb == float2(0)) &&
            all(a.rgb == float3(0)) && b.r == a.a;
        return red_first || alpha_first ? a : float4(1, 0, 1, 0);
    }
    """
    let shaders: MTLLibrary
    do {
        shaders = try dev.makeLibrary(source: source, options: nil)
    } catch {
        report("a8_iosurface_render_\(w)x\(h)", false, "shader library: \(error)")
        return
    }
    let masks: [(String, MTLColorWriteMask)] = [
        ("all", .all), ("red", .red), ("alpha", .alpha), ("none", []),
    ]
    let rect = MTLScissorRect(x: w / 4, y: h / 4, width: max(1, w / 2), height: max(1, h / 2))

    func pipeline(_ fragment: String, _ format: MTLPixelFormat,
                  _ mask: MTLColorWriteMask = .all) throws -> MTLRenderPipelineState {
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = shaders.makeFunction(name: "alpha_surface_vs")
        descriptor.fragmentFunction = shaders.makeFunction(name: fragment)
        descriptor.colorAttachments[0].pixelFormat = format
        descriptor.colorAttachments[0].writeMask = mask
        descriptor.colorAttachments[0].isBlendingEnabled = false
        return try dev.makeRenderPipelineState(descriptor: descriptor)
    }

    func outputTexture() -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        descriptor.storageMode = .private
        descriptor.usage = [.renderTarget, .shaderRead]
        return dev.makeTexture(descriptor: descriptor)
    }

    func samplePass(_ command: MTLCommandBuffer, _ texture: MTLTexture,
                    _ output: MTLTexture, _ pipe: MTLRenderPipelineState,
                    paired: MTLTexture? = nil) -> Bool {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = output
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        if paired != nil {
            guard let seed = command.makeRenderCommandEncoder(descriptor: pass) else { return false }
            seed.endEncoding()
            pass.colorAttachments[0].loadAction = .load
        }
        guard let encoder = command.makeRenderCommandEncoder(descriptor: pass) else { return false }
        encoder.setRenderPipelineState(pipe)
        encoder.setFragmentTexture(texture, index: 0)
        if let paired { encoder.setFragmentTexture(paired, index: 1) }
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        return true
    }

    for (maskName, mask) in masks {
        for readName in ["read", "sample", "paired_read"] {
            let label = "a8_iosurface_render_mask_\(maskName)_\(readName)_\(w)x\(h)"
            do {
                let write = try pipeline("alpha_surface_write", .r8Unorm, mask)
                let read = try pipeline("alpha_surface_\(readName)", .rgba8Unorm)
                let properties: [IOSurfacePropertyKey: Any] = [
                    .width: w, .height: h, .bytesPerElement: 1,
                    .pixelFormat: UInt32(0x5238_2020), // 'R8  ', not an MTL ordinal
                ]
                guard let surface = IOSurface(properties: properties) else {
                    report(label, false, "IOSurface allocation failed"); continue
                }
                let redDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .r8Unorm, width: w, height: h, mipmapped: false)
                redDescriptor.storageMode = .shared
                redDescriptor.usage = [.renderTarget, .shaderRead]
                let alphaDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .a8Unorm, width: w, height: h, mipmapped: false)
                alphaDescriptor.storageMode = .shared
                alphaDescriptor.usage = .shaderRead
                guard let red = dev.makeTexture(descriptor: redDescriptor, iosurface: surface, plane: 0),
                      let alpha = dev.makeTexture(descriptor: alphaDescriptor, iosurface: surface, plane: 0),
                      let redOutput = outputTexture(), let alphaOutput = outputTexture(),
                      let command = queue.makeCommandBuffer() else {
                    report(label, false, "R8/A8 surface aliases or output allocation refused"); continue
                }
                let pass = MTLRenderPassDescriptor()
                pass.colorAttachments[0].texture = red
                pass.colorAttachments[0].loadAction = .clear
                pass.colorAttachments[0].clearColor = MTLClearColor(
                    red: 37.0 / 255.0, green: 0, blue: 0, alpha: 211.0 / 255.0)
                pass.colorAttachments[0].storeAction = .store
                guard let seed = command.makeRenderCommandEncoder(descriptor: pass) else {
                    report(label, false, "R8 clear encoder refused"); continue
                }
                seed.endEncoding()
                pass.colorAttachments[0].loadAction = .load
                guard let encoder = command.makeRenderCommandEncoder(descriptor: pass) else {
                    report(label, false, "R8 render encoder refused"); continue
                }
                encoder.setRenderPipelineState(write)
                encoder.setScissorRect(rect)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                guard command.status == .completed,
                      let sampleCommand = queue.makeCommandBuffer() else {
                    report(label, false, "producer failed: \(String(describing: command.error))")
                    continue
                }
                // Completed-command ordering isolates channel interpretation
                // from synchronization between separate IOSurface wrappers.
                // Paired mode exercises both snapshot/direct binding orders
                // while LOAD belongs to an unrelated primary attachment.
                let paired = readName == "paired_read"
                guard samplePass(sampleCommand, alpha, alphaOutput, read, paired: paired ? red : nil),
                      samplePass(sampleCommand, red, redOutput, read, paired: paired ? alpha : nil) else {
                    report(label, false, "sample encoder refused"); continue
                }
                sampleCommand.commit()
                sampleCommand.waitUntilCompleted()
                guard sampleCommand.status == .completed,
                      let alphaPixels = readBack(readPipe, alphaOutput, w, h),
                      let redPixels = readBack(readPipe, redOutput, w, h) else {
                    report(label, false, "readback failed: \(String(describing: sampleCommand.error))")
                    continue
                }
                var raw = [UInt8](repeating: 0, count: w * h)
                raw.withUnsafeMutableBytes {
                    red.getBytes($0.baseAddress!, bytesPerRow: w,
                                 from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
                }
                var rawWrong = 0, redWrong = 0, alphaWrong = 0
                var first = ""
                for y in 0..<h {
                    for x in 0..<w {
                        let i = y * w + x
                        let inside = x >= rect.x && x < rect.x + rect.width &&
                            y >= rect.y && y < rect.y + rect.height
                        let value = UInt8(inside && mask.contains(.red)
                            ? 64 + ((x * 7 + y * 13) % 128) : 37)
                        let wantRed = pack(value, 0, 0, 255)
                        let wantAlpha = pack(0, 0, 0, value)
                        if raw[i] != value { rawWrong += 1 }
                        if redPixels[i] != wantRed { redWrong += 1 }
                        if alphaPixels[i] != wantAlpha { alphaWrong += 1 }
                        if first.isEmpty && (raw[i] != value || redPixels[i] != wantRed ||
                                             alphaPixels[i] != wantAlpha) {
                            first = "first=(\(x),\(y)) raw=\(raw[i]) wantByte=\(value)" +
                                " red=\(hex(redPixels[i])) wantRed=\(hex(wantRed))" +
                                " alpha=\(hex(alphaPixels[i])) wantAlpha=\(hex(wantAlpha))"
                        }
                    }
                }
                report(label, rawWrong + redWrong + alphaWrong == 0,
                       "rawWrong=\(rawWrong) redWrong=\(redWrong) alphaWrong=\(alphaWrong) \(first)")
            } catch {
                report(label, false, "pipeline: \(error)")
            }
        }
    }
}
