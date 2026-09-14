import Foundation
import Metal

func mapperRingWrapCase() {
    let name = "iosurface_mapper_ring_wrap"
    let iterations = 2048
    let width = 64, height = 32
    for iteration in 0..<iterations {
        let passed: Bool = autoreleasepool {
            guard let texture = makeIOSurfaceTarget(width, height, name) else { return false }
            guard let command = queue.makeCommandBuffer() else {
                report(name, false, "iteration=\(iteration) command buffer unavailable")
                return false
            }
            let red = UInt8(iteration % 251 + 2)
            let green = UInt8((iteration * 7) % 251 + 2)
            let blue = UInt8((iteration * 13) % 251 + 2)
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = texture
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColor(
                red: Double(red) / 255, green: Double(green) / 255,
                blue: Double(blue) / 255, alpha: 1)
            guard let encoder = command.makeRenderCommandEncoder(descriptor: pass) else {
                report(name, false, "iteration=\(iteration) render encoder unavailable")
                return false
            }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else {
                report(name, false, "iteration=\(iteration) \(String(describing: command.error))")
                return false
            }
            var bytes = [UInt8](repeating: 0xa5, count: width * height * 4)
            bytes.withUnsafeMutableBytes {
                texture.getBytes($0.baseAddress!, bytesPerRow: width * 4,
                                 from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
            }
            let expected = [blue, green, red, 255]
            for offset in stride(from: 0, to: bytes.count, by: 4) {
                let actual = Array(bytes[offset..<offset + 4])
                if actual != expected {
                    report(name, false,
                           "iteration=\(iteration) pixel=\(offset / 4) got=\(actual) want=\(expected)")
                    return false
                }
            }
            return true
        }
        if !passed { return }
    }
    report(name, true, "\(iterations) distinct IOSurfaces cleared and read back exactly")
}
