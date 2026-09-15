#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

enum { Width = 16, Height = 8 };
typedef struct { simd_float4 a, b; } Colours;

static id<MTLTexture> texture(id<MTLDevice> device) {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
        width:Width height:Height mipmapped:NO];
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    return [device newTextureWithDescriptor:desc];
}

static id<MTLRenderPipelineState> pipeline(id<MTLDevice> device,
                                         id<MTLLibrary> library,
                                         NSString *fragment, BOOL mrt) {
    MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = [library newFunctionWithName:@"vertex_main"];
    desc.fragmentFunction = [library newFunctionWithName:fragment];
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    if (mrt) desc.colorAttachments[1].pixelFormat = MTLPixelFormatBGRA8Unorm;
    NSError *error = nil;
    id<MTLRenderPipelineState> result = [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!result) fprintf(stderr, "Pipeline failed: %s\n", error.description.UTF8String);
    return result;
}

static id<MTLRenderCommandEncoder> encoder(id<MTLCommandBuffer> command,
                                          id<MTLTexture> a, id<MTLTexture> b,
                                          MTLLoadAction loadA, MTLLoadAction loadB) {
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = a;
    pass.colorAttachments[0].loadAction = loadA;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    if (b) {
        pass.colorAttachments[1].texture = b;
        pass.colorAttachments[1].loadAction = loadB;
        pass.colorAttachments[1].storeAction = MTLStoreActionStore;
        pass.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, 0, 1);
    }
    return [command renderCommandEncoderWithDescriptor:pass];
}

static void draw(id<MTLRenderCommandEncoder> enc, id<MTLRenderPipelineState> pso,
                 NSUInteger x, NSUInteger width, Colours colours) {
    [enc setRenderPipelineState:pso];
    [enc setScissorRect:(MTLScissorRect){x, 0, width, Height}];
    [enc setFragmentBytes:&colours length:sizeof(colours) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
}

static BOOL pixels(id<MTLTexture> image, const uint8_t left[4],
                   const uint8_t right[4], const char *label) {
    uint8_t bytes[Width * Height * 4];
    [image getBytes:bytes bytesPerRow:Width * 4
        fromRegion:MTLRegionMake2D(0, 0, Width, Height) mipmapLevel:0];
    for (NSUInteger y = 0; y < Height; y++) {
        for (NSUInteger x = 0; x < Width; x++) {
            const uint8_t *want = x < Width / 2 ? left : right;
            const uint8_t *have = bytes + (y * Width + x) * 4;
            if (memcmp(have, want, 4)) {
                fprintf(stderr, "%s pixel=(%lu,%lu) BGRA=[%u,%u,%u,%u] want=[%u,%u,%u,%u]\n",
                        label, x, y, have[0], have[1], have[2], have[3],
                        want[0], want[1], want[2], want[3]);
                return NO;
            }
        }
    }
    return YES;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            puts("CASE mrt_device FAIL no Metal device");
            return 2;
        }
        NSString *source =
            @"#include <metal_stdlib>\nusing namespace metal;\n"
             "struct V { float4 position [[position]]; };\n"
             "struct Colours { float4 a; float4 b; };\n"
             "struct Pair { float4 a [[color(0)]]; float4 b [[color(1)]]; };\n"
             "vertex V vertex_main(uint i [[vertex_id]]) {\n"
             " const float2 p[6] = {float2(-1,-1),float2(1,-1),float2(-1,1),"
             "float2(-1,1),float2(1,-1),float2(1,1)};\n"
             " V v; v.position=float4(p[i],0,1); return v; }\n"
             "fragment Pair pair_main(constant Colours &c [[buffer(0)]]) {\n"
             " Pair p; p.a=c.a; p.b=c.b; return p; }\n"
             "fragment float4 sample_main(V v [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
             " return t.read(uint2(v.position.xy)); }\n";
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            fprintf(stderr, "CASE mrt_shader FAIL %s\n", error.description.UTF8String);
            return 2;
        }
        id<MTLRenderPipelineState> pair = pipeline(device, library, @"pair_main", YES);
        id<MTLRenderPipelineState> sample = pipeline(device, library, @"sample_main", NO);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!pair || !sample || !queue) {
            puts("CASE mrt_pipeline FAIL pipeline or queue allocation");
            return 2;
        }
        const uint8_t red[4] = {0, 0, 255, 255}, green[4] = {0, 255, 0, 255};
        const uint8_t blue[4] = {255, 0, 0, 255}, yellow[4] = {0, 255, 255, 255};
        const uint8_t black[4] = {0, 0, 0, 255};
        Colours first = {{1, 0, 0, 1}, {0, 1, 0, 1}};
        Colours second = {{0, 0, 1, 1}, {1, 1, 0, 1}};
        const char *names[] = {
            "mrt_two_draw_scissors", "mrt_changed_secondary_encoder",
            "mrt_secondary_clear_next_encoder", "mrt_secondary_sample_next_encoder",
        };
        int failures = 0;
        for (NSUInteger test = 0; test < 4; test++) {
            id<MTLTexture> a = texture(device), b = texture(device), c = texture(device);
            if (!a || !b || !c) {
                fprintf(stderr, "CASE %s FAIL texture allocation\n", names[test]);
                failures++;
                continue;
            }
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLRenderCommandEncoder> enc = encoder(command, a, b, MTLLoadActionClear, MTLLoadActionClear);
            if (!command || !enc) {
                printf("CASE %s FAIL command or encoder allocation\n", names[test]);
                return 2;
            }
            draw(enc, pair, 0, test == 0 ? Width / 2 : Width, first);
            if (test == 0) {
                draw(enc, pair, Width / 2, Width / 2, second);
                [enc endEncoding];
            } else {
                [enc endEncoding];
                if (test == 3) {
                    enc = encoder(command, c, nil, MTLLoadActionClear, MTLLoadActionDontCare);
                    if (!enc) {
                        printf("CASE %s FAIL sampled encoder allocation\n", names[test]);
                        return 2;
                    }
                    [enc setRenderPipelineState:sample];
                    [enc setFragmentTexture:b atIndex:0];
                    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
                } else {
                    enc = encoder(command, a, test == 1 ? c : b,
                                  MTLLoadActionLoad, MTLLoadActionClear);
                    if (!enc) {
                        printf("CASE %s FAIL second MRT encoder allocation\n", names[test]);
                        return 2;
                    }
                    draw(enc, pair, Width / 2, Width / 2, second);
                }
                [enc endEncoding];
            }
            [command commit];
            [command waitUntilCompleted];
            BOOL ok = command.status == MTLCommandBufferStatusCompleted;
            if (!ok) fprintf(stderr, "command failed: %s\n", command.error.description.UTF8String);
            if (ok) {
                if (test == 0) {
                    ok = pixels(a, red, blue, "primary");
                    ok = pixels(b, green, yellow, "secondary") && ok;
                } else if (test == 1) {
                    ok = pixels(a, red, blue, "primary");
                    ok = pixels(b, green, green, "old-secondary") && ok;
                    ok = pixels(c, black, yellow, "new-secondary") && ok;
                } else if (test == 2) {
                    ok = pixels(a, red, blue, "primary");
                    ok = pixels(b, black, yellow, "cleared-secondary") && ok;
                } else {
                    ok = pixels(a, red, red, "primary");
                    ok = pixels(b, green, green, "secondary") && ok;
                    ok = pixels(c, green, green, "sampled-secondary") && ok;
                }
            }
            printf("CASE %s %s exact BGRA pixels across all affected targets\n", names[test], ok ? "PASS" : "FAIL");
            failures += !ok;
        }
        printf("SUMMARY cases=4 failures=%d skipped=0\n", failures);
        return failures ? 1 : 0;
    }
}
