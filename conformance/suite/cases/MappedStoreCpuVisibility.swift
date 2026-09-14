import Foundation
import IOSurface
import Metal

func mappedStoreCpuVisibilityCase() {
    let name = "mapped_store_cpu_visible_after_completion"
    let width = 16, height = 8
    guard let pipeline = makeRenderPipeline("solid_fs", .bgra8Unorm),
          let vertices = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                                        options: .storageModeShared) else {
        report(name, false, "pipeline or vertex buffer unavailable")
        return
    }
    guard let (texture, surface) = makeIOSurfaceTargetWithBacking(width, height, name) else { return }
    for value in 0..<256 {
        let passed: Bool = autoreleasepool {
            let red = UInt8(value), green = UInt8(255 - value), blue = UInt8((value * 13) & 255)
            var color = SIMD4<Float>(Float(red) / 255, Float(green) / 255, Float(blue) / 255, 1)
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = texture
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
            guard let command = queue.makeCommandBuffer(),
                  let encoder = command.makeRenderCommandEncoder(descriptor: pass) else {
                report(name, false, "round=\(value) command or encoder unavailable")
                return false
            }
            encoder.setRenderPipelineState(pipeline)
            encoder.setVertexBuffer(vertices, offset: 0, index: 0)
            encoder.setFragmentBytes(&color, length: 16, index: 0)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else {
                report(name, false, "round=\(value) command failed: \(String(describing: command.error))")
                return false
            }
            let locked = surface.lock(options: .readOnly, seed: nil)
            guard locked == 0 else {
                report(name, false, "round=\(value) IOSurface lock failed: \(locked)")
                return false
            }
            defer { surface.unlock(options: .readOnly, seed: nil) }
            let bytes = surface.baseAddress.assumingMemoryBound(to: UInt8.self)
            let pitch = surface.bytesPerRow
            let expected = [blue, green, red, 255]
            for y in 0..<height {
                for x in 0..<width {
                    let offset = y * pitch + x * 4
                    let actual = Array(UnsafeBufferPointer(start: bytes.advanced(by: offset), count: 4))
                    if actual != expected {
                        report(name, false,
                               "round=\(value) pixel=(\(x),\(y)) CPU=\(actual) want=\(expected)")
                        return false
                    }
                }
            }
            return true
        }
        if !passed { return }
    }
    report(name, true, "256 rendered colors visible through CPU IOSurface memory after completion")
}
