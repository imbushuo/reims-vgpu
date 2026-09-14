#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>

// Retained composite views must observe CPU writes after completed commands.
static BOOL render(id<MTLCommandBuffer> command, id<MTLRenderPipelineState> pipeline,
                   id<MTLTexture> source, id<MTLTexture> target) {
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
    if (!encoder) return NO;
    [encoder setRenderPipelineState:pipeline];
    [encoder setFragmentTexture:source atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    return YES;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            @"#include <metal_stdlib>\nusing namespace metal;\n"
             "vertex float4 v(uint i [[vertex_id]]) { const float2 p[]={"
             "float2(-1,-1),float2(3,-1),float2(-1,3)}; return float4(p[i],0,1); }\n"
             "fragment float4 f(float4 p [[position]],texture2d<float> t [[texture(0)]]) {"
             "constexpr sampler s(coord::normalized,address::clamp_to_edge,filter::nearest);"
             "return t.sample(s,p.xy/float2(t.get_width(),t.get_height())); }\n"
            options:nil error:&error];
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [library newFunctionWithName:@"v"];
        pd.fragmentFunction = [library newFunctionWithName:@"f"];
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:pd error:&error];
        if (!device || !queue || !library || !pipeline) {
            fprintf(stderr, "setup failed: %s\n", error.description.UTF8String);
            return 1;
        }
        for (NSNumber *word in @[@0x1f9, @0x21f]) {
            CVPixelBufferRef pixels = NULL;
            NSDictionary *properties = @{
                (__bridge NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{},
                (__bridge NSString *)kCVPixelBufferMetalCompatibilityKey: @YES,
            };
            if (CVPixelBufferCreate(kCFAllocatorDefault, 16, 8,
                    kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
                    (__bridge CFDictionaryRef)properties, &pixels) != kCVReturnSuccess) {
                fprintf(stderr, "pixel buffer failed\n");
                return 2;
            }
            IOSurfaceRef surface = CVPixelBufferGetIOSurface(pixels);
            MTLTextureDescriptor *sd = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:(MTLPixelFormat)word.unsignedIntegerValue
                width:16 height:8 mipmapped:NO];
            sd.storageMode = MTLStorageModeShared;
            sd.usage = MTLTextureUsageShaderRead;
            id<MTLTexture> retained = [device newTextureWithDescriptor:sd iosurface:surface plane:0];
            MTLTextureDescriptor *td = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                width:16 height:8 mipmapped:NO];
            td.storageMode = MTLStorageModeShared;
            td.usage = MTLTextureUsageRenderTarget;
            id<MTLTexture> a = [device newTextureWithDescriptor:td];
            id<MTLTexture> b = [device newTextureWithDescriptor:td];
            if (!retained || !a || !b) {
                fprintf(stderr, "texture creation failed format=%lx\n", word.unsignedLongValue);
                CFRelease(pixels);
                return 3;
            }
            unsigned changes = 0;
            uint32_t previous = 0;
            for (unsigned round = 0; round < 256; ++round) {
                @autoreleasepool {
                    if (IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) return 4;
                    for (size_t plane = 0; plane < 2; ++plane) {
                        uint8_t *base = IOSurfaceGetBaseAddressOfPlane(surface, plane);
                        size_t pitch = IOSurfaceGetBytesPerRowOfPlane(surface, plane);
                        size_t width = IOSurfaceGetWidthOfPlane(surface, plane);
                        size_t height = IOSurfaceGetHeightOfPlane(surface, plane);
                        size_t channels = plane ? 2 : 1;
                        uint16_t value = (plane ? 512 : (round % 2 ? 940 : 64)) << 6;
                        for (size_t y = 0; y < height; ++y) {
                            uint16_t *row = (uint16_t *)(base + y * pitch);
                            for (size_t x = 0; x < width * channels; ++x) row[x] = value;
                        }
                    }
                    if (IOSurfaceUnlock(surface, 0, NULL) != kIOReturnSuccess) return 5;
                    id<MTLTexture> fresh = [device newTextureWithDescriptor:sd iosurface:surface plane:0];
                    id<MTLCommandBuffer> command = [queue commandBuffer];
                    if (!fresh || !render(command, pipeline, retained, a) ||
                        !render(command, pipeline, fresh, b)) return 6;
                    [command commit];
                    [command waitUntilCompleted];
                    if (command.status != MTLCommandBufferStatusCompleted) {
                        fprintf(stderr, "command failed: %s\n", command.error.description.UTF8String);
                        return 7;
                    }
                    uint32_t oldResult[128], freshResult[128];
                    [a getBytes:oldResult bytesPerRow:64 fromRegion:MTLRegionMake2D(0,0,16,8) mipmapLevel:0];
                    [b getBytes:freshResult bytesPerRow:64 fromRegion:MTLRegionMake2D(0,0,16,8) mipmapLevel:0];
                    if (memcmp(oldResult, freshResult, sizeof(oldResult))) {
                        fprintf(stderr, "FAIL format=%lx round=%u retained=%08x fresh=%08x\n",
                                word.unsignedLongValue, round, oldResult[0], freshResult[0]);
                        return 8;
                    }
                    changes += round && freshResult[0] != previous;
                    previous = freshResult[0];
                }
            }
            if (changes != 255) {
                fprintf(stderr, "source did not visibly change: %u\n", changes);
                return 9;
            }
            printf("CASE current_planar_%lx PASS 256 CPU updates, retained versus fresh texture, 255 visible changes\n",
                   word.unsignedLongValue);
            fflush(stdout);
            CFRelease(pixels);
        }
    }
    return 0;
}
