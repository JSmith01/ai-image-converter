const NORMALIZE_VECTOR: v128 = v128.splat<f32>(255.0);

// Nv12ToHWTensorOpt, size should be divisible by 4
export function Nv12ToHW(inputNv12$: i32, outputBuffer$: i32, size: i32): void {
    // copy Y channel to the input buffer with normalization
    let input$: i32 = inputNv12$;
    let Y$: i32 = outputBuffer$;
    const yEnd: i32 = inputNv12$ + size;
    let v: v128;
    while (input$ < yEnd) {
        v = v128.load32_splat(input$);
        v = v128.extend_low<u8>(v);
        v = v128.extend_low<u16>(v);
        v = v128.convert<u32>(v);
        store<v128>(Y$, f32x4.div(v, NORMALIZE_VECTOR)); // 4 x u8 -> 4 x f32
        input$ += 4;
        Y$ += 16;
    }
}

const U_PARTS: v128 = v128(0, 1, 2, 3, 0, 1, 2, 3, 8, 9, 10, 11, 8, 9, 10, 11);
const V_PARTS: v128 = v128(4, 5, 6, 7, 4, 5, 6, 7, 12, 13, 14, 15, 12, 13, 14, 15);

//Nv12ToCHWTensorOpt, width should be divisible by 4
export function Nv12ToCHW(inputNv12$: i32, outputBuffer$: i32, width: i32, height: i32): void {
    const size: i32 = width * height;
    // copy Y channel to the input buffer with normalization
    let input$: i32 = inputNv12$;
    let Y$: i32 = outputBuffer$;
    const yEnd$: i32 = inputNv12$ + size;
    let v: v128;
    while (input$ < yEnd$) {
        // copy-paste to avoid stack use
        v = v128.load32_splat(input$);
        v = v128.extend_low<u8>(v);
        v = v128.extend_low<u16>(v);
        v = v128.convert<u32>(v);
        store<v128>(Y$, f32x4.div(v, NORMALIZE_VECTOR)); // 4 x u8 -> 4 x f32
        input$ += 4;
        Y$ += 16;
    }

    let U$: i32 = Y$;
    let V$: i32 = Y$ + size * 4;
    // copy U and V channels to the input buffer with normalization
    const yuvArrayLength$: i32 = yEnd$ + size / 2; // 4:2:0 Y + 2 * (U/4) + 2 * (V/4)
    const outputLineLength: i32 = width * 4;
    const lineWidthUV = width / 2;
    let linePixels: i32 = lineWidthUV; // U & V are half resolution horizontally (and vertically, but that's up to doubling the line)

    while (input$ < yuvArrayLength$) {
        v = v128.load32_splat(input$);
        input$ += 4;
        v = v128.extend_low<u8>(v);
        v = v128.extend_low<u16>(v);
        v = v128.convert<u32>(v);
        // four values U1 V1 U2 V2 normalized to f32 (u8 / 255 -> f32)
        v = f32x4.div(v, NORMALIZE_VECTOR);
        // sets four U outputs - U1 U1 U2 U2
        v128.store(U$, v128.swizzle(v, U_PARTS));
        U$ += 16;
        // sets four V outputs - V1 V1 V2 V2
        v128.store(V$, v128.swizzle(v, V_PARTS));
        V$ += 16;
        linePixels -= 2;

        if (linePixels <= 0) {
            memory.copy(U$, U$ - outputLineLength, outputLineLength);
            memory.copy(V$, V$ - outputLineLength, outputLineLength);
            U$ += outputLineLength;
            V$ += outputLineLength;
            linePixels = lineWidthUV;
        }
    }
}

const I1_PARTS: v128 = v128(0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7);
const I2_PARTS: v128 = v128(8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15);

// I420ToCHWTensorOpt, width should be divisible by 8
export function I420ToCHW(inputI420$: i32, outputBuffer$: i32, width: i32, height: i32): void {
    const size: i32 = width * height;
    // copy Y channel to the input buffer with normalization
    let input$: i32 = inputI420$;
    let output$: i32 = outputBuffer$;
    const yEnd$: i32 = inputI420$ + size;
    let v: v128;
    while (input$ < yEnd$) {
        // copy-paste to avoid stack use
        v = v128.load32_splat(input$);
        v = v128.extend_low<u8>(v);
        v = v128.extend_low<u16>(v);
        v = v128.convert<u32>(v);
        store<v128>(output$, f32x4.div(v, NORMALIZE_VECTOR)); // 4 x u8 -> 4 x f32
        input$ += 4;
        output$ += 16;
    }

    const uvEnd$ = input$ + size / 2;
    const outputLineLength: i32 = width * 4;
    const lineWidthUV = width / 2;
    let linePixels: i32 = lineWidthUV;
    while(input$ < uvEnd$) {
        v = v128.load32_splat(input$); // assumes width is divisible by 8
        v = v128.extend_low<u8>(v);
        v = v128.extend_low<u16>(v);
        v = v128.convert<u32>(v);
        v = f32x4.div(v, NORMALIZE_VECTOR);
        input$ += 4;
        linePixels -= 4;
        // four 4 from the input become 8 pixels in the output - ABCD -> AABBCCDD
        // each with conversion (u8 / 255) -> f32
        v128.store(output$, v128.swizzle(v, I1_PARTS), 0);
        v128.store(output$, v128.swizzle(v, I2_PARTS), 16);
        output$ += 32;
        if (linePixels <= 0) {
            memory.copy(output$, output$ - outputLineLength, outputLineLength);
            output$ += outputLineLength;
            linePixels = lineWidthUV;
        }
    }
}

export function I420TileToCHW(inputI420$: i32, outputBuffer$: i32, x: i32, y: i32, width: i32, height: i32, frameWidth: i32, frameHeight: i32): void {
    const planeYSize = frameWidth * frameHeight;
    const planeUOffset = inputI420$ + planeYSize + (x >> 1) + (y >> 1) * (frameWidth >> 1);
    const planeVOffset = planeUOffset + planeYSize / 4;
    const uvLines = (height >> 1);
    let input$: i32 = inputI420$ + x + y * frameWidth;
    let output$: i32 = outputBuffer$;
    let yLines: i32 = height;
    const widthOffset = width * 4;
    let v: v128;
    let hCounter: i32;
    while (yLines > 0) {
        hCounter = width;
        while (hCounter > 0) {
            // copy-paste to avoid stack use
            v = v128.load32_splat(input$);
            v = v128.extend_low<u8>(v);
            v = v128.extend_low<u16>(v);
            v = v128.convert<u32>(v);
            store<v128>(output$, f32x4.div(v, NORMALIZE_VECTOR)); // 4 x u8 -> 4 x f32
            input$ += 4;
            output$ += 16;
            hCounter -= 4;
        }
        input$ += frameWidth - width;
        yLines--;
    }

    input$ = planeUOffset;
    let lines = uvLines;
    const uvWidth = width / 2;
    let planeCounter: i32 = 1;

    while (lines > 0) {
        hCounter = uvWidth;
        while (hCounter > 0) {
            // copy-paste to avoid stack use
            v = v128.load32_splat(input$); // assumes width is divisible by 8
            v = v128.extend_low<u8>(v);
            v = v128.extend_low<u16>(v);
            v = v128.convert<u32>(v);
            v = f32x4.div(v, NORMALIZE_VECTOR);
            // four 4 from the input become 8 pixels in the output - ABCD -> AABBCCDD
            // each with conversion (u8 / 255) -> f32
            v128.store(output$, v128.swizzle(v, I1_PARTS), 0);
            v128.store(output$, v128.swizzle(v, I2_PARTS), 16);
            input$ += 4;
            hCounter -= 4;
            output$ += 32;
        }
        // copy U/V line to double vertical resolution
        memory.copy(output$, output$ - widthOffset, widthOffset);
        output$ += widthOffset;
        input$ += frameWidth / 2 - uvWidth;
        lines--;

        // V plane start
        if (lines <= 0 && planeCounter > 0) {
            planeCounter--;
            lines = uvLines;
            input$ = planeVOffset;
        }
    }
}

/**
 * Bilinear upscale of a single channel image (2x in each dimension)
 * @param src Source image data pointer
 * @param dst Destination image buffer pointer (must be 4x the size of source)
 * @param srcWidth Width of source image
 * @param srcHeight Height of source image
 */
export function bilinearUpscaleChannel(
  src: usize, 
  dst: usize, 
  srcWidth: i32, 
  srcHeight: i32
): void {
  const dstWidth = srcWidth * 2;

  for (let y = 0; y < srcHeight; y++) {
    const srcRow0 = src + y * srcWidth;
    const srcRow1 = (y + 1 < srcHeight) ? srcRow0 + srcWidth : srcRow0;
    const dstRow0 = dst + (2 * y) * dstWidth;
    const dstRow1 = dst + (2 * y + 1) * dstWidth;
    
    // Process 16 pixels at a time using SIMD
    let x: i32 = 0;
    for (; x <= srcWidth - 16; x += 16) {
      // Load 16 pixels from the current position
      const topLeft = v128.load(srcRow0 + x);
      
      // For the right pixels, we need to handle potential boundary
      let topRight: v128, bottomRight: v128;
      if (x + 16 < srcWidth) {
        // Not at the right edge, safe to load shifted
        topRight = v128.load(srcRow0 + x + 1);
        bottomRight = v128.load(srcRow1 + x + 1);
      } else {
        // At the right edge, handle boundary with SIMD masks
        const remainingPixels = srcWidth - x - 1;
        
        if (remainingPixels > 0) {
          // Load as much as we can directly (unaligned is fine)
          topRight = v128.load(srcRow0 + x + 1);
          bottomRight = v128.load(srcRow1 + x + 1);
          
          // For any pixels that would read past the end, replace with the last valid pixel
          if (remainingPixels < 16) {
            const lastPixelTR = load<u8>(srcRow0 + srcWidth - 1);
            const lastPixelBR = load<u8>(srcRow1 + srcWidth - 1);
            
            // Create vectors with replicated last valid pixel
            const lastTRVector = v128.splat<u8>(lastPixelTR);
            const lastBRVector = v128.splat<u8>(lastPixelBR);
            
            // Create a mask based on remainingPixels
            // We need all 1s for valid pixels, all 0s for invalid ones
            let byteMask: v128;
            
            // Select appropriate mask based on remainingPixels
            if (remainingPixels == 1) {
              byteMask = v128(-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            } else if (remainingPixels == 2) {
              byteMask = v128(-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            } else if (remainingPixels == 3) {
              byteMask = v128(-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            } else if (remainingPixels == 4) {
              byteMask = v128(-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            } else if (remainingPixels == 5) {
              byteMask = v128(-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            } else if (remainingPixels == 6) {
              byteMask = v128(-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            } else if (remainingPixels == 7) {
              byteMask = v128(-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            } else if (remainingPixels == 8) {
              byteMask = v128(-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0);
            } else if (remainingPixels == 9) {
              byteMask = v128(-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0);
            } else if (remainingPixels == 10) {
              byteMask = v128(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0);
            } else if (remainingPixels == 11) {
              byteMask = v128(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0);
            } else if (remainingPixels == 12) {
              byteMask = v128(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0);
            } else if (remainingPixels == 13) {
              byteMask = v128(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0);
            } else if (remainingPixels == 14) {
              byteMask = v128(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0);
            } else /* remainingPixels == 15 */ {
              byteMask = v128(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0);
            }
            
            // Use bitwise operations to blend the loaded value and the last valid pixel
            // Format: (loaded & mask) | (lastPixel & ~mask)
            // This keeps loaded values where mask is -1 and uses lastPixel where mask is 0
            topRight = v128.or(
              v128.and(topRight, byteMask),
              v128.and(lastTRVector, v128.not(byteMask))
            );
            
            bottomRight = v128.or(
              v128.and(bottomRight, byteMask),
              v128.and(lastBRVector, v128.not(byteMask))
            );
          }
        } else {
          // Edge case: x + 1 is already past srcWidth, create vectors with the last pixel replicated
          const lastPixelTR = load<u8>(srcRow0 + srcWidth - 1);
          const lastPixelBR = load<u8>(srcRow1 + srcWidth - 1);
          
          topRight = v128.splat<u8>(lastPixelTR);
          bottomRight = v128.splat<u8>(lastPixelBR);
        }
      }
      
      // Load bottom row
      const bottomLeft = v128.load(srcRow1 + x);
      
      // Calculate all four interpolated values for each pixel using SIMD
      const topAvg = i8x16.avgr_u(topLeft, topRight);    // Horizontal average (TL+TR+1)>>1
      const bottomAvg = i8x16.avgr_u(bottomLeft, bottomRight);
      const vAvg = i8x16.avgr_u(topLeft, bottomLeft);    // Vertical average (TL+BL+1)>>1
      const diagAvg = i8x16.avgr_u(topAvg, bottomAvg);   // Approximates (TL+TR+BL+BR+2)>>2
      
      // Use SIMD shuffle to interleave pixels for the first 8 pixels
      // Interleave topLeft and topAvg for the top row
      const row0_interleaved1 = i8x16.shuffle(
        topLeft, topAvg,
        0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23
      );
      
      const row0_interleaved2 = i8x16.shuffle(
        topLeft, topAvg,
        8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
      );
      
      // Interleave vAvg and diagAvg for the bottom row
      const row1_interleaved1 = i8x16.shuffle(
        vAvg, diagAvg,
        0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23
      );
      
      const row1_interleaved2 = i8x16.shuffle(
        vAvg, diagAvg,
        8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
      );
      
      // Store the interleaved pixels directly to the destination
      v128.store(dstRow0 + 2 * x, row0_interleaved1);
      v128.store(dstRow0 + 2 * x + 16, row0_interleaved2);
      v128.store(dstRow1 + 2 * x, row1_interleaved1);
      v128.store(dstRow1 + 2 * x + 16, row1_interleaved2);
    }
    
    // Handle remaining pixels with scalar code
    for (; x < srcWidth; x++) {
      const topLeft = load<u8>(srcRow0 + x);
      const topRight = (x + 1 < srcWidth) ? load<u8>(srcRow0 + x + 1) : topLeft;
      const bottomLeft = load<u8>(srcRow1 + x);
      const bottomRight = (x + 1 < srcWidth) ? load<u8>(srcRow1 + x + 1) : bottomLeft;
      
      store<u8>(dstRow0 + 2 * x, topLeft);
      store<u8>(dstRow0 + 2 * x + 1, <u8>((topLeft + topRight + 1) >> 1));
      store<u8>(dstRow1 + 2 * x, <u8>((topLeft + bottomLeft + 1) >> 1));
      store<u8>(dstRow1 + 2 * x + 1, <u8>((topLeft + topRight + bottomLeft + bottomRight + 2) >> 2));
    }
  }
}
