import Metal
import Foundation

func pipelineTaskIsolationChild() -> Int32 {
    guard let argument = CommandLine.arguments.last,
          let variant = UInt32(argument), variant < 4 else {
        print("invalid pipeline task variant")
        return 2
    }
    let marker = 0x13579B00 + variant
    let red = Float(variant + 1) / 8
    let source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void task_compute(device uint *output [[buffer(0)]],
                             uint tid [[thread_position_in_grid]]) {
        if (tid == 0) output[0] = \(marker)u;
    }
    vertex float4 task_vertex(uint index [[vertex_id]]) {
        const float2 positions[3] = {
            float2(-1, -1), float2(3, -1), float2(-1, 3)
        };
        return float4(positions[index], 0, 1);
    }
    fragment float4 task_fragment() {
        return float4(\(red), 0, 0, 1);
    }
    """
    do {
        let shaders = try dev.makeLibrary(source: source, options: nil)
        let compute = try dev.makeComputePipelineState(
            function: shaders.makeFunction(name: "task_compute")!)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = shaders.makeFunction(name: "task_vertex")
        descriptor.fragmentFunction = shaders.makeFunction(name: "task_fragment")
        descriptor.colorAttachments[0].pixelFormat = .rgba8Unorm
        let render = try dev.makeRenderPipelineState(descriptor: descriptor)
        guard let output = dev.makeBuffer(length: 4, options: .storageModeShared) else {
            print("buffer allocation failed")
            return 2
        }
        output.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 8, height: 8, mipmapped: false)
        textureDescriptor.storageMode = .shared
        textureDescriptor.usage = [.renderTarget]
        guard let target = dev.makeTexture(descriptor: textureDescriptor),
              let commands = queue.makeCommandBuffer(),
              let dispatch = commands.makeComputeCommandEncoder() else {
            print("task fixture allocation failed")
            return 2
        }
        dispatch.setComputePipelineState(compute)
        dispatch.setBuffer(output, offset: 0, index: 0)
        dispatch.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        dispatch.endEncoding()
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        pass.colorAttachments[0].storeAction = .store
        guard let draw = commands.makeRenderCommandEncoder(descriptor: pass) else {
            print("render encoder allocation failed")
            return 2
        }
        draw.setRenderPipelineState(render)
        draw.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        draw.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        guard commands.status == .completed else {
            print("command buffer failed: \(String(describing: commands.error))")
            return 1
        }
        let got = output.contents().load(as: UInt32.self)
        var pixels = [UInt8](repeating: 0, count: 8 * 8 * 4)
        pixels.withUnsafeMutableBytes {
            target.getBytes($0.baseAddress!, bytesPerRow: 8 * 4,
                            from: MTLRegionMake2D(0, 0, 8, 8), mipmapLevel: 0)
        }
        let expected = UInt8((red * 255).rounded())
        let wrong = (0..<64).filter {
            let offset = $0 * 4
            return pixels[offset] != expected || pixels[offset + 1] != 0
                || pixels[offset + 2] != 0 || pixels[offset + 3] != 255
        }.count
        print("compute=\(got)/\(marker) render_wrong=\(wrong)/64")
        return got == marker && wrong == 0 ? 0 : 1
    } catch {
        print("pipeline creation failed: \(error)")
        return 2
    }
}

/// Each fresh task constructs objects in the same order but has different
/// shader constants, so a previous task's readiness cannot satisfy a cold build.
func pipelineTaskIsolationCases() {
    for variant in 0..<4 {
        let child = Process()
        let output = Pipe()
        child.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
        child.arguments = ["--pipeline-task-child", String(variant)]
        child.standardOutput = output
        child.standardError = output
        do {
            try child.run()
        } catch {
            report("pipeline_task_isolation_\(variant)", false, "child launch failed: \(error)")
            continue
        }
        output.fileHandleForWriting.closeFile()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        child.waitUntilExit()
        let detail = String(decoding: data, as: UTF8.self)
            .split(whereSeparator: \.isNewline).joined(separator: " ")
        report("pipeline_task_isolation_\(variant)",
               child.terminationReason == .exit && child.terminationStatus == 0,
               "exit=\(child.terminationStatus) \(detail)")
    }
}
