#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

enum { Width = 16, Height = 12, DestWidth = 24, DestHeight = 20 };
typedef struct { simd_float2 size, offset; } Parameters;

static void source_pixel(unsigned x, unsigned y, BOOL wide, uint16_t out[4]) {
    const uint16_t palette[][4] = {
        {0x1001, 0x3400, 0x3800, 0x3c00},
        {0x4000, 0xb800, 0x3555, 0x3800},
        {0x3800, 0x3a00, 0x3000, 0x3c00},
    };
    BOOL glyph = (x == 3 && y >= 2 && y <= 9)
        || (y == 2 && x >= 3 && x <= 10) || (y == 5 && x >= 3 && x <= 8);
    if (glyph) {
        memcpy(out, (uint16_t[]){0, 0x3c00, 0, 0x3c00}, 8);
    } else if (wide) {
        memcpy(out, palette[(x + 3 * y) % 3], 8);
    } else {
        out[0] = x & 1 ? 0x3c00 : 0;
        out[1] = y & 1 ? 0x3c00 : 0;
        out[2] = (x + y) & 1 ? 0x3c00 : 0;
        out[3] = 0x3c00;
    }
}

static id<MTLTexture> texture(id<MTLDevice> device, MTLPixelFormat format,
                              NSUInteger width, NSUInteger height, BOOL memoryless) {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
    desc.storageMode = memoryless ? MTLStorageModeMemoryless : MTLStorageModeShared;
    desc.usage = memoryless ? MTLTextureUsageRenderTarget :
        MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    return [device newTextureWithDescriptor:desc];
}

static id<MTLRenderPipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                          MTLPixelFormat primary, NSUInteger count, BOOL copy) {
    MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = [library newFunctionWithName:@"vertex_main"];
    desc.fragmentFunction = [library newFunctionWithName:copy ? @"ordered_copy" : @"paint"];
    for (NSUInteger slot = 0; slot < count; slot++) {
        desc.colorAttachments[slot].pixelFormat = slot ? MTLPixelFormatRGBA16Float : primary;
        desc.colorAttachments[slot].writeMask = !copy && slot == 0
            ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
    }
    NSError *error = nil;
    id<MTLRenderPipelineState> result = [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!result) fprintf(stderr, "pipeline: %s\n", error.description.UTF8String);
    return result;
}

static void draw(id<MTLRenderCommandEncoder> encoder, id<MTLRenderPipelineState> pso,
                 id<MTLBuffer> indices, MTLScissorRect rect, simd_float2 offset) {
    Parameters params = {{Width, Height}, offset};
    [encoder setRenderPipelineState:pso];
    [encoder setVertexBytes:&params length:sizeof(params) atIndex:0];
    [encoder setScissorRect:rect];
    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:6
        indexType:MTLIndexTypeUInt16 indexBuffer:indices indexBufferOffset:0];
}

static void copy_expected(uint16_t *destination, const uint16_t *source,
                          MTLScissorRect rect, unsigned dx, unsigned dy) {
    for (NSUInteger y = rect.y; y < rect.y + rect.height; y++) {
        for (NSUInteger x = rect.x; x < rect.x + rect.width; x++) {
            memcpy(destination + ((y + dy) * DestWidth + x + dx) * 4,
                   source + (y * Width + x) * 4, 8);
        }
    }
}

static BOOL compare_half(const uint16_t *actual, const uint16_t *expected,
                         unsigned width, unsigned height, const char *label) {
    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            for (unsigned c = 0; c < 4; c++) {
                NSUInteger index = (y * width + x) * 4 + c;
                if (actual[index] != expected[index]) {
                    fprintf(stderr, "%s (%u,%u) component%u half=%04x expected=%04x\n",
                            label, x, y, c, actual[index], expected[index]);
                    return NO;
                }
            }
        }
    }
    return YES;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!device || !queue) {
            puts("CASE rog_copy_device FAIL no device or queue");
            return 2;
        }
        NSString *source =
            @"#include <metal_stdlib>\nusing namespace metal;\n"
             "struct Params { float2 size; float2 offset; };\n"
             "struct V { float4 position [[position]]; float2 destination; };\n"
             "vertex V vertex_main(uint i [[vertex_id]], constant Params& p [[buffer(0)]]) {\n"
             " const float2 positions[4]={float2(-1,-1),float2(1,-1),float2(-1,1),float2(1,1)};\n"
             " float2 q=positions[i]; V v; v.position=float4(q,0,1);\n"
             " v.destination=float2((q.x+1)*p.size.x*0.5,(1-q.y)*p.size.y*0.5)+p.offset;\n"
             " return v; }\n"
             "fragment void ordered_copy(V v [[stage_in]], half4 prior [[color(0)]],\n"
             " texture2d<half,access::write> destination [[texture(3),raster_order_group(0)]]) {\n"
             " destination.write(prior,ushort2(v.destination)); }\n"
             "fragment half4 paint() { return half4(0,1,0,1); }\n";
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            fprintf(stderr, "CASE rog_copy_shader FAIL %s\n", error.description.UTF8String);
            return 2;
        }
        const uint16_t indexBytes[] = {0, 1, 2, 2, 1, 3};
        id<MTLBuffer> indices = [device newBufferWithBytes:indexBytes length:sizeof(indexBytes)
            options:MTLResourceStorageModeShared];
        if (!indices) {
            puts("CASE rog_copy_indices FAIL allocation");
            return 2;
        }
        const MTLScissorRect first = {1, 1, 11, 8}, painted = {4, 2, 7, 7};
        const MTLScissorRect second = {3, 0, 11, 10}, corner = {0, 0, 3, 3};
        const char *names[] = {"rog_framebuffer_copy_rgba16_two_rt", "rog_framebuffer_copy_bgra8_three_rt"};
        int failures = 0;
        for (unsigned test = 0; test < 2; test++) {
            BOOL wide = test == 0;
            NSUInteger count = wide ? 2 : 3;
            MTLPixelFormat format = wide ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
            id<MTLTexture> primary = texture(device, format, Width, Height, NO);
            id<MTLTexture> destination = texture(device, MTLPixelFormatRGBA16Float, DestWidth, DestHeight, NO);
            id<MTLTexture> secondTarget = texture(device, MTLPixelFormatRGBA16Float, Width, Height, YES);
            id<MTLTexture> thirdTarget = count == 3
                ? texture(device, MTLPixelFormatRGBA16Float, Width, Height, YES) : nil;
            id<MTLRenderPipelineState> copier = pipeline(device, library, format, count, YES);
            id<MTLRenderPipelineState> painter = pipeline(device, library, format, count, NO);
            if (!primary || !destination || !secondTarget || (count == 3 && !thirdTarget)
                || !copier || !painter) {
                printf("CASE %s FAIL allocation or pipeline\n", names[test]);
                failures++;
                continue;
            }
            uint16_t expectedSource[Width * Height * 4], expectedDest[DestWidth * DestHeight * 4];
            uint8_t source8[Width * Height * 4];
            for (unsigned y = 0; y < Height; y++) {
                for (unsigned x = 0; x < Width; x++) {
                    NSUInteger at = (y * Width + x) * 4;
                    source_pixel(x, y, wide, expectedSource + at);
                    const unsigned order[] = {2, 1, 0, 3};
                    for (unsigned c = 0; c < 4; c++) {
                        source8[at + c] = expectedSource[at + order[c]] == 0x3c00 ? 255 : 0;
                    }
                }
            }
            for (unsigned y = 0; y < DestHeight; y++) {
                for (unsigned x = 0; x < DestWidth; x++) {
                    source_pixel(x + 19, y + 13, YES, expectedDest + (y * DestWidth + x) * 4);
                }
            }
            [primary replaceRegion:MTLRegionMake2D(0, 0, Width, Height) mipmapLevel:0
                withBytes:wide ? (const void *)expectedSource : (const void *)source8
                bytesPerRow:Width * (wide ? 8 : 4)];
            [destination replaceRegion:MTLRegionMake2D(0, 0, DestWidth, DestHeight) mipmapLevel:0
                withBytes:expectedDest bytesPerRow:DestWidth * 8];
            MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[0].texture = primary;
            pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            for (NSUInteger slot = 1; slot < count; slot++) {
                pass.colorAttachments[slot].texture = slot == 1 ? secondTarget : thirdTarget;
                pass.colorAttachments[slot].loadAction = MTLLoadActionClear;
                pass.colorAttachments[slot].storeAction = MTLStoreActionDontCare;
            }
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLRenderCommandEncoder> enc = [command renderCommandEncoderWithDescriptor:pass];
            if (!command || !enc) {
                printf("CASE %s FAIL command or encoder allocation\n", names[test]);
                return 2;
            }
            [enc setFragmentTexture:destination atIndex:3];
            draw(enc, copier, indices, first, (simd_float2){3, 2});
            copy_expected(expectedDest, expectedSource, first, 3, 2);
            draw(enc, painter, indices, painted, (simd_float2){0, 0});
            for (NSUInteger y = painted.y; y < painted.y + painted.height; y++) {
                for (NSUInteger x = painted.x; x < painted.x + painted.width; x++) {
                    memcpy(expectedSource + (y * Width + x) * 4,
                           (uint16_t[]){0, 0x3c00, 0, 0x3c00}, 8);
                }
            }
            draw(enc, copier, indices, second, (simd_float2){3, 2});
            copy_expected(expectedDest, expectedSource, second, 3, 2);
            draw(enc, copier, indices, corner, (simd_float2){0, 0});
            copy_expected(expectedDest, expectedSource, corner, 0, 0);
            [enc endEncoding];
            [command commit];
            [command waitUntilCompleted];
            BOOL ok = command.status == MTLCommandBufferStatusCompleted;
            if (!ok) fprintf(stderr, "command: %s\n", command.error.description.UTF8String);
            if (ok) {
                uint16_t actualDest[DestWidth * DestHeight * 4], actualSource[Width * Height * 4];
                [destination getBytes:actualDest bytesPerRow:DestWidth * 8
                    fromRegion:MTLRegionMake2D(0, 0, DestWidth, DestHeight) mipmapLevel:0];
                ok = compare_half(actualDest, expectedDest, DestWidth, DestHeight, "texture3");
                if (wide) {
                    [primary getBytes:actualSource bytesPerRow:Width * 8
                        fromRegion:MTLRegionMake2D(0, 0, Width, Height) mipmapLevel:0];
                } else {
                    [primary getBytes:source8 bytesPerRow:Width * 4
                        fromRegion:MTLRegionMake2D(0, 0, Width, Height) mipmapLevel:0];
                    const unsigned order[] = {2, 1, 0, 3};
                    for (unsigned p = 0; p < Width * Height; p++) {
                        for (unsigned c = 0; c < 4; c++) {
                            uint8_t byte = source8[p * 4 + order[c]];
                            if (byte != 0 && byte != 255) ok = NO;
                            actualSource[p * 4 + c] = byte == 255 ? 0x3c00 : 0;
                        }
                    }
                }
                ok = compare_half(actualSource, expectedSource, Width, Height, "attachment0") && ok;
            }
            printf("CASE %s %s texture3 exact half pixels, overlapping indexed draws, offset and untouched coverage\n",
                   names[test], ok ? "PASS" : "FAIL");
            failures += !ok;
        }
        printf("SUMMARY cases=2 failures=%d skipped=0\n", failures);
        return failures ? 1 : 0;
    }
}
