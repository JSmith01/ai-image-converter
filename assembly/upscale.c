#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include <stdint.h>
#include <wasm_simd128.h>

// emcc -O3 -Wl,--import-memory -Wl,--export-dynamic -Wl,--allow-undefined -Wl,--no-check-features -msimd128 -s ALLOW_MEMORY_GROWTH=1 -Wl,--no-entry upscale.c -o upscale.wasm

EMSCRIPTEN_KEEPALIVE
void upscale_channel_bilinear(const uint8_t* src, uint8_t* dst, int srcWidth, int srcHeight) {
    int dstWidth = srcWidth * 2;

    for (int y = 0; y < srcHeight; y++) {
        const uint8_t* srcRow0 = src + y * srcWidth;
        const uint8_t* srcRow1 = (y + 1 < srcHeight) ? srcRow0 + srcWidth : srcRow0;
        uint8_t* dstRow0 = dst + (2 * y) * dstWidth;
        uint8_t* dstRow1 = dst + (2 * y + 1) * dstWidth;
        
        // Process 16 pixels at a time using SIMD
        int x = 0;
        for (; x <= srcWidth - 16; x += 16) {
            // Load 16 pixels from the current position
            v128_t topLeft = wasm_v128_load(&srcRow0[x]);
            
            // For the right pixels, we need to handle potential boundary
            v128_t topRight, bottomRight;
            if (x + 16 < srcWidth) {
                // Not at the right edge, safe to load shifted
                topRight = wasm_v128_load(&srcRow0[x + 1]);
                bottomRight = wasm_v128_load(&srcRow1[x + 1]);
            } else {
                // At the right edge, manually handle last pixel
                uint8_t buffer[16], bufferBottom[16];
                for (int i = 0; i < 15; i++) {
                    buffer[i] = srcRow0[x + i + 1];
                    bufferBottom[i] = srcRow1[x + i + 1];
                }
                // Repeat the last pixel for the boundary
                buffer[15] = srcRow0[srcWidth - 1];
                bufferBottom[15] = srcRow1[srcWidth - 1];
                
                topRight = wasm_v128_load(buffer);
                bottomRight = wasm_v128_load(bufferBottom);
            }
            
            // Load bottom row
            v128_t bottomLeft = wasm_v128_load(&srcRow1[x]);
            
            // Calculate all four interpolated values for each pixel using SIMD
            v128_t topAvg = wasm_u8x16_avgr(topLeft, topRight);    // Horizontal average (TL+TR+1)>>1
            v128_t bottomAvg = wasm_u8x16_avgr(bottomLeft, bottomRight);
            v128_t vAvg = wasm_u8x16_avgr(topLeft, bottomLeft);    // Vertical average (TL+BL+1)>>1
            v128_t diagAvg = wasm_u8x16_avgr(topAvg, bottomAvg);   // Approximates (TL+TR+BL+BR+2)>>2
            
            // Use SIMD shuffle to interleave pixels for the first 8 pixels
            // Interleave topLeft and topAvg for the top row
            v128_t row0_interleaved1 = wasm_i8x16_shuffle(
                topLeft, topAvg,
                0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23
            );
            
            v128_t row0_interleaved2 = wasm_i8x16_shuffle(
                topLeft, topAvg,
                8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
            );
            
            // Interleave vAvg and diagAvg for the bottom row
            v128_t row1_interleaved1 = wasm_i8x16_shuffle(
                vAvg, diagAvg,
                0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23
            );
            
            v128_t row1_interleaved2 = wasm_i8x16_shuffle(
                vAvg, diagAvg,
                8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
            );
            
            // Store the interleaved pixels directly to the destination
            wasm_v128_store(&dstRow0[2 * x], row0_interleaved1);
            wasm_v128_store(&dstRow0[2 * x + 16], row0_interleaved2);
            wasm_v128_store(&dstRow1[2 * x], row1_interleaved1);
            wasm_v128_store(&dstRow1[2 * x + 16], row1_interleaved2);
        }
        
        // Handle remaining pixels with scalar code
        for (; x < srcWidth; x++) {
            uint8_t topLeft = srcRow0[x];
            uint8_t topRight = (x + 1 < srcWidth) ? srcRow0[x + 1] : topLeft;
            uint8_t bottomLeft = srcRow1[x];
            uint8_t bottomRight = (x + 1 < srcWidth) ? srcRow1[x + 1] : bottomLeft;
            
            dstRow0[2 * x] = topLeft;
            dstRow0[2 * x + 1] = (uint8_t)((topLeft + topRight + 1) >> 1);
            dstRow1[2 * x] = (uint8_t)((topLeft + bottomLeft + 1) >> 1);
            dstRow1[2 * x + 1] = (uint8_t)((topLeft + topRight + bottomLeft + bottomRight + 2) >> 2);
        }
    }
}
