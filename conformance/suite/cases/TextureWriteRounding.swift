import Foundation
import Metal

private func textureRoundingSource(offset: Int = 0) -> String { """
#include <metal_stdlib>
using namespace metal;
kernel void rounding_write(device const float4 *values [[buffer(0)]],
    texture2d<float, access::write> destination [[texture(0)]],
    uint i [[thread_position_in_grid]]) {
    destination.write(values[i], uint2(i + \(offset), 0));
}
""" }

private func roundedHalfBits(_ value: Float, towardZero: Bool) -> UInt16 {
    let sign = UInt16((value.bitPattern >> 16) & 0x8000)
    if value.isNaN { return sign | 0x7e00 }
    if value.isInfinite { return sign | 0x7c00 }
    let magnitude = Double(abs(value))
    if magnitude > 65504 && towardZero { return sign | 0x7bff }
    if magnitude >= 65520 { return sign | 0x7c00 }
    let rule: FloatingPointRoundingRule = towardZero ? .towardZero : .toNearestOrEven
    if magnitude < pow(2, -14) {
        return sign | UInt16((magnitude * pow(2, 24)).rounded(rule))
    }
    let exponent = Int((value.bitPattern >> 23) & 255) - 127
    let significand = Int((magnitude / pow(2, Double(exponent - 10))).rounded(rule))
    return sign | UInt16((exponent + 15) * 1024 + significand - 1024)
}

/// Alternating modes and storage formats catches executable specialization aliasing. Select the
/// documented compiler mode as well: the native oracle retains that mode when only the private
/// pipeline property changes. Normalized formats are outside MSL §1.6.7's rounding control.
func textureWriteRoundingCases() {
    do {
        let descriptor = MTLComputePipelineDescriptor()
        guard descriptor.responds(to: NSSelectorFromString("setTextureWriteRoundingMode:")) else {
            skip("compute_texture_write_rounding", "pipeline rounding property unavailable")
            return
        }
        var pipelines: [[MTLComputePipelineState]] = []
        for compilerMode in ["native", "rtz", "rte"] {
            let options = MTLCompileOptions()
            if #available(macOS 15.0, *) { options.mathMode = .safe }
            else { options.fastMathEnabled = false }
            guard options.responds(to: NSSelectorFromString("setAdditionalCompilerArguments:")) else {
                skip("compute_texture_write_rounding", "runtime compiler rounding option unavailable")
                return
            }
            options.setValue("-ftexture-write-rounding-mode=\(compilerMode)",
                             forKey: "additionalCompilerArguments")
            let shaders = try dev.makeLibrary(source: textureRoundingSource(), options: options)
            descriptor.computeFunction = shaders.makeFunction(name: "rounding_write")
            var variants: [MTLComputePipelineState] = []
            for mode in 0...2 {
                descriptor.setValue(NSNumber(value: mode), forKey: "textureWriteRoundingMode")
                variants.append(try dev.makeComputePipelineState(
                    descriptor: descriptor, options: [], reflection: nil))
            }
            pipelines.append(variants)
        }
        let values: [Float] = [
            0, -0.0, 1, -1,
            Float(bitPattern: 0x3f801000), Float(bitPattern: 0x3f803000),
            Float(bitPattern: 0x3f801800), -Float(bitPattern: 0x3f801800),
            pow(2, -25), pow(2, -24) * 1.5, pow(2, -24), -pow(2, -25),
            Float(bitPattern: 0x387fe000), Float(bitPattern: 0x387ff000),
            65504, 65520, 70000, -70000, .infinity, -.infinity, .nan,
            0.5, 0.1, 1.5 / 255, 2.5 / 255, 10.25 / 255, 10.75 / 255,
        ]
        let vectors = values.map { SIMD4<Float>(repeating: $0) }
        let input = dev.makeBuffer(bytes: vectors, length: vectors.count * 16)!
        let formats: [(MTLPixelFormat, String, Int, Int)] = [
            (.r16Float, "r16float", 2, 1),
            (.rgba8Unorm, "rgba8unorm", 4, 4),
            (.rgba16Float, "rgba16float", 8, 4),
            (.rgba32Float, "rgba32float", 16, 4),
            (.rg16Float, "rg16float", 4, 2),
            (.bgra8Unorm, "bgra8unorm", 4, 4),
        ]
        for (format, name, bytesPerPixel, channels) in formats {
            let td = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: values.count, height: 1, mipmapped: false)
            td.storageMode = .shared
            td.usage = [.shaderRead, .shaderWrite]
            guard let texture = dev.makeTexture(descriptor: td) else {
                report("compute_texture_rounding_\(name)_setup", false, "texture allocation failed")
                continue
            }
            var baseline: [UInt8] = []
            for compilerMode in 0...2 {
              for (iteration, mode) in [0, 1, 2, 1].enumerated() {
                let command = queue.makeCommandBuffer()!
                let encoder = command.makeComputeCommandEncoder()!
                encoder.setComputePipelineState(pipelines[compilerMode][mode])
                encoder.setBuffer(input, offset: 0, index: 0)
                encoder.setTexture(texture, index: 0)
                encoder.dispatchThreads(MTLSize(width: values.count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                let label = "compute_texture_rounding_\(name)_air\(compilerMode)_mode\(mode)_\(iteration)"
                guard command.status == .completed else {
                    report(label, false, "command failed \(String(describing: command.error))")
                    continue
                }
                var bytes = [UInt8](repeating: 0, count: values.count * bytesPerPixel)
                texture.getBytes(&bytes, bytesPerRow: bytes.count,
                    from: MTLRegionMake2D(0, 0, values.count, 1), mipmapLevel: 0)
                if compilerMode == 0 && mode == 0 { baseline = bytes }
                var wrong = 0
                if format == .rgba8Unorm || format == .bgra8Unorm {
                    wrong = zip(bytes, baseline).filter { $0 != $1 }.count
                } else {
                    for (index, value) in values.enumerated() {
                        for channel in 0..<channels {
                            let offset = index * bytesPerPixel + channel * (bytesPerPixel / channels)
                            if format == .rgba32Float {
                                let bits = (0..<4).reduce(UInt32(0)) {
                                    $0 | (UInt32(bytes[offset + $1]) << ($1 * 8))
                                }
                                if value.isNaN ? !Float(bitPattern: bits).isNaN : bits != value.bitPattern {
                                    wrong += 1
                                }
                            } else {
                                let bits = UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
                                // Native AIR uses RTZ on this calibrated Metal compatibility
                                // profile. Explicit AIR modes survive mismatched descriptors.
                                let expected = roundedHalfBits(value, towardZero: compilerMode != 2)
                                let isNaN = bits & 0x7c00 == 0x7c00 && bits & 0x03ff != 0
                                if value.isNaN ? !isNaN : bits != expected {
                                    wrong += 1
                                }
                            }
                            // A fresh, observable coordinate offset makes each AIR program different. These cases
                            // distinguish per-write semantics from a native PSO cache accidentally retaining the
                            // first descriptor mode compiled for one function.
                            for (compilerMode, mode, offset) in [("native", 2, 509), ("rte", 1, 521), ("rtz", 2, 523)] {
                                let options = MTLCompileOptions()
                                if #available(macOS 15.0, *) { options.mathMode = .safe }
                                else { options.fastMathEnabled = false }
                                options.setValue("-ftexture-write-rounding-mode=\(compilerMode)", forKey: "additionalCompilerArguments")
                                let shaders = try dev.makeLibrary(source: textureRoundingSource(offset: offset), options: options)
                                let descriptor = MTLComputePipelineDescriptor()
                                descriptor.computeFunction = shaders.makeFunction(name: "rounding_write")
                                descriptor.setValue(NSNumber(value: mode), forKey: "textureWriteRoundingMode")
                                let pipeline = try dev.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
                                let td = MTLTextureDescriptor.texture2DDescriptor(
                                    pixelFormat: .rgba16Float, width: values.count + offset, height: 1, mipmapped: false)
                                td.storageMode = .shared
                                td.usage = [.shaderRead, .shaderWrite]
                                let texture = dev.makeTexture(descriptor: td)!
                                let command = queue.makeCommandBuffer()!
                                let encoder = command.makeComputeCommandEncoder()!
                                encoder.setComputePipelineState(pipeline)
                                encoder.setBuffer(input, offset: 0, index: 0)
                                encoder.setTexture(texture, index: 0)
                                encoder.dispatchThreads(MTLSize(width: values.count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                                encoder.endEncoding()
                                command.commit()
                                command.waitUntilCompleted()
                                let label = "compute_texture_rounding_cold_\(compilerMode)_mode\(mode)"
                                guard command.status == .completed else {
                                    report(label, false, "command failed \(String(describing: command.error))")
                                    continue
                                }
                                var bytes = [UInt8](repeating: 0, count: values.count * 8)
                                texture.getBytes(&bytes, bytesPerRow: bytes.count,
                                    from: MTLRegionMake2D(offset, 0, values.count, 1), mipmapLevel: 0)
                                var wrong = 0
                                for (i, value) in values.enumerated() {
                                    for channel in 0..<4 {
                                        let at = i * 8 + channel * 2
                                        let bits = UInt16(bytes[at]) | (UInt16(bytes[at + 1]) << 8)
                                        let isNaN = bits & 0x7c00 == 0x7c00 && bits & 0x03ff != 0
                                        let expected = roundedHalfBits(value, towardZero: compilerMode != "rte")
                                        if value.isNaN ? !isNaN : bits != expected { wrong += 1 }
                                    }
                                }
                                report(label, wrong == 0, "wrong=\(wrong) first_descriptor=\(mode) coordinate_offset=\(offset)")
                            }
                        }
                    }
                }
                report(label, wrong == 0, "wrong=\(wrong) pixels=\(values.count)")
              }
            }
        }
    } catch {
        report("compute_texture_write_rounding_setup", false, "\(error)")
    }
}
