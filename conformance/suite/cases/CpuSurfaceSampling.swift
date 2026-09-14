import Foundation
import IOSurface
import Metal

// No vertex/constant input buffers: this isolates sampled IOSurface freshness
// from the separately tested set*Bytes snapshot lifecycle.
func cpuSurfaceSamplingCases(gpuSeeded: Bool = false) {
    let source = """
    #include <metal_stdlib>
    using namespace metal;
    vertex float4 cpu_surface_vertex(uint i [[vertex_id]]) {
        const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
        return float4(p[i], 0, 1);
    }
    fragment float4 cpu_surface_fragment(float4 p [[position]],
                                        texture2d<float, access::read> image [[texture(0)]]) {
        return image.read(uint2(p.xy));
    }
    fragment float4 cpu_surface_gpu_seed() { return float4(0.25, 0.5, 0.75, 1.0); }
    """
    for (label, format, fourcc, bpp) in [
        ("bgra8", MTLPixelFormat.bgra8Unorm, UInt32(0x4247_5241), 4),
        ("rgba16float", MTLPixelFormat.rgba16Float, UInt32(0x5247_6841), 8),
    ] {
        let prefix = gpuSeeded ? "gpu_published_cpu_iosurface" : "cpu_iosurface"
        let name = "\(prefix)_sample_after_completed_commands_\(label)"
        let width = 8, height = 4
        guard let library = try? dev.makeLibrary(source: source, options: nil),
              let vertex = library.makeFunction(name: "cpu_surface_vertex"),
              let fragment = library.makeFunction(name: "cpu_surface_fragment") else {
            report(name, false, "sampling shader compilation failed"); continue
        }
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertex
        pipelineDescriptor.fragmentFunction = fragment
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        guard let pipeline = try? dev.makeRenderPipelineState(descriptor: pipelineDescriptor),
              let surface = IOSurface(properties: [
                .width: width, .height: height, .pixelFormat: fourcc, .bytesPerElement: bpp,
              ]) else {
            report(name, false, "pipeline or IOSurface creation failed"); continue
        }
        let inputDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false)
        inputDescriptor.storageMode = .shared
        inputDescriptor.usage = gpuSeeded ? [.shaderRead, .renderTarget] : .shaderRead
        let outputDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        outputDescriptor.storageMode = .shared
        outputDescriptor.usage = .renderTarget
        guard let input = dev.makeTexture(descriptor: inputDescriptor, iosurface: surface, plane: 0),
              let output = dev.makeTexture(descriptor: outputDescriptor) else {
            report(name, false, "texture creation failed"); continue
        }
        var seedPipeline: MTLRenderPipelineState?
        if gpuSeeded {
            let seedDescriptor = MTLRenderPipelineDescriptor()
            seedDescriptor.vertexFunction = vertex
            seedDescriptor.fragmentFunction = library.makeFunction(name: "cpu_surface_gpu_seed")
            seedDescriptor.colorAttachments[0].pixelFormat = format
            seedPipeline = try? dev.makeRenderPipelineState(descriptor: seedDescriptor)
            guard seedPipeline != nil else {
                report(name, false, "GPU seed pipeline unavailable"); continue
            }
        }
        var failure: String?
        for round in 0..<256 {
            autoreleasepool {
                let red = UInt8(round), green = UInt8(255 - round), blue = UInt8((round * 13) & 255)
                if let seedPipeline {
                    // Produce a real prior GPU frame. No publication/generation
                    // stamp is fabricated by the test, and no input buffer can
                    // contaminate the constant seed color.
                    let pass = MTLRenderPassDescriptor()
                    pass.colorAttachments[0].texture = input
                    pass.colorAttachments[0].loadAction = .clear
                    pass.colorAttachments[0].storeAction = .store
                    guard let command = queue.makeCommandBuffer(),
                          let encoder = command.makeRenderCommandEncoder(descriptor: pass) else {
                        failure = "round=\(round) GPU seed encoder unavailable"; return
                    }
                    encoder.setRenderPipelineState(seedPipeline)
                    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                    encoder.endEncoding()
                    command.commit()
                    command.waitUntilCompleted()
                    guard command.status == .completed else {
                        failure = "round=\(round) GPU seed failed: \(String(describing: command.error))"
                        return
                    }
                }
                let locked = surface.lock(options: [], seed: nil)
                guard locked == 0 else {
                    failure = "round=\(round) lock=\(locked)"; return
                }
                for y in 0..<height {
                    let row = surface.baseAddress.advanced(by: y * surface.bytesPerRow)
                    for x in 0..<width {
                        if bpp == 4 {
                            let pixel = row.advanced(by: x * bpp).assumingMemoryBound(to: UInt8.self)
                            for (channel, value) in [blue, green, red, 255].enumerated() {
                                pixel[channel] = value
                            }
                        } else {
                            let pixel = row.advanced(by: x * bpp).assumingMemoryBound(to: Float16.self)
                            for (channel, value) in [red, green, blue, 255].enumerated() {
                                pixel[channel] = Float16(Float(value) / 255)
                            }
                        }
                    }
                }
                let unlocked = surface.unlock(options: [], seed: nil)
                guard unlocked == 0 else {
                    failure = "round=\(round) unlock=\(unlocked)"; return
                }
                let pass = MTLRenderPassDescriptor()
                pass.colorAttachments[0].texture = output
                pass.colorAttachments[0].loadAction = .clear
                pass.colorAttachments[0].storeAction = .store
                guard let command = queue.makeCommandBuffer(),
                      let encoder = command.makeRenderCommandEncoder(descriptor: pass) else {
                    failure = "round=\(round) command/encoder unavailable"; return
                }
                encoder.setRenderPipelineState(pipeline)
                encoder.setFragmentTexture(input, index: 0)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                guard command.status == .completed else {
                    failure = "round=\(round) command=\(command.status.rawValue) \(String(describing: command.error))"
                    return
                }
                var bytes = [UInt8](repeating: 0, count: width * height * 4)
                bytes.withUnsafeMutableBytes {
                    output.getBytes($0.baseAddress!, bytesPerRow: width * 4,
                                    from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
                }
                let expected = [blue, green, red, 255]
                for pixel in 0..<(width * height) {
                    let actual = Array(bytes[pixel * 4..<pixel * 4 + 4])
                    if zip(actual, expected).contains(where: { abs(Int($0) - Int($1)) > 1 }) {
                        failure = "round=\(round) pixel=\(pixel) have=\(actual) want=\(expected)"
                        // Distinguish sampling an old host image from a later
                        // device write clobbering the CPU-authored source.
                        let lock = surface.lock(options: .readOnly, seed: nil)
                        if lock == 0 {
                            let position = (pixel / width) * surface.bytesPerRow + (pixel % width) * bpp
                            let pointer = surface.baseAddress.advanced(by: position)
                            let after: String
                            if bpp == 4 {
                                let cpu = Array(UnsafeBufferPointer(
                                    start: pointer.assumingMemoryBound(to: UInt8.self), count: 4))
                                after = "\(cpu)"
                            } else {
                                let p = pointer.assumingMemoryBound(to: Float16.self)
                                after = "\((0..<4).map { Float(p[$0]) })_RGBA_float"
                            }
                            let unlocked = surface.unlock(options: .readOnly, seed: nil)
                            failure! += " source_cpu_after=\(after) unlock=\(unlocked)"
                        } else {
                            failure! += " source_cpu_after_lock=\(lock)"
                        }
                        return
                    }
                }
            }
            if failure != nil { break }
        }
        if let failure { report(name, false, failure) }
        else {
            report(name, true, gpuSeeded
                ? "256 GPU-produced frames then locked CPU updates sampled after completion"
                : "256 locked CPU updates sampled after separate completed commands")
        }
    }
}
