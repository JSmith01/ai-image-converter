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
