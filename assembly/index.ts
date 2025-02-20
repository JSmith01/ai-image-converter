const NORMALIZE_VALUE: f32 = 1.0 / 255.0;
const NORMALIZE_VECTOR: v128 = v128.splat<f32>(255.0);

export function Nv12ToHWTensorOpt(inputNv12: i32, outputBuffer: i32, size: i32): void {
    // copy Y channel to the input buffer with normalization
    let ptrInput: i32 = inputNv12;
    let ptrY: i32 = outputBuffer;
    const yEnd: i32 = inputNv12 + size;
    let v: v128;
    while (ptrInput < yEnd) {
        v = v128.load32_splat(ptrInput);
        v = v128.extend_low<u8>(v);
        v = v128.extend_low<u16>(v);
        v = v128.convert<u32>(v);
        store<v128>(ptrY, f32x4.div(v, NORMALIZE_VECTOR));
        ptrInput += 4;
        ptrY += 16;
    }
}

export function Nv12ToHWTensor(inputNv12: i32, outputBuffer: i32, size: i32): void {
    // copy Y channel to the input buffer with normalization
    let ptrInput: i32 = inputNv12;
    let ptrY: i32 = outputBuffer;
    const yEnd: i32 = inputNv12 + size;
    while (ptrInput < yEnd) {
        store<v128>(
            ptrY,
            f32x4.div(
                f32x4((load<u8>(ptrInput, 0) as f32), (load<u8>(ptrInput, 1) as f32), (load<u8>(ptrInput, 2) as f32), (load<u8>(ptrInput, 3) as f32)),
                NORMALIZE_VECTOR
            )
        );
        ptrInput += 4;
        ptrY += 16;
    }
}

export function Nv12ToHWTensorDeopt(inputNv12: i32, outputBuffer: i32, size: i32): void {
    let ptrInput: i32 = inputNv12;
    let ptrY: i32 = outputBuffer;
    const yEnd: i32 = inputNv12 + size;
    while (ptrInput < yEnd) {
        store<f32>(ptrY, (load<u8>(ptrInput++) as f32) * NORMALIZE_VALUE);
        ptrY += 4;
    }
}

const U_PARTS: v128 = v128(0, 1, 2, 3, 0, 1, 2, 3, 8, 9, 10, 11, 8, 9, 10, 11);
const V_PARTS: v128 = v128(4, 5, 6, 7, 4, 5, 6, 7, 12, 13, 14, 15, 12, 13, 14, 15);

export function Nv12ToCHWTensorOpt(inputNv12: i32, outputBuffer: i32, width: i32, height: i32): void {
    const size: i32 = width * height;
    // copy Y channel to the input buffer with normalization
    let ptrInput: i32 = inputNv12;
    let ptrY: i32 = outputBuffer;
    const yEnd: i32 = inputNv12 + size;
    let v: v128;
    while (ptrInput < yEnd) {
        v = v128.load32_splat(ptrInput);
        v = v128.extend_low<u8>(v);
        v = v128.extend_low<u16>(v);
        v = v128.convert<u32>(v);
        store<v128>(ptrY, f32x4.div(v, NORMALIZE_VECTOR));
        ptrInput += 4;
        ptrY += 16;
    }

    let uIdx: i32 = ptrY;
    let vIdx: i32 = ptrY + size * 4;
    // copy U and V channels to the input buffer with normalization
    const yuvArrayLength: i32 = yEnd + size / 2; // 4:2:0 Y + 2 * (U/4) + 2 * (V/4)
    const outputLineLength: i32 = width * 4;
    const lineWidthUV = width / 2;
    let linePixels: i32 = lineWidthUV; // U & V are half resolution horizontally (and vertically, but that's up to doubling the line)

    while (ptrInput < yuvArrayLength) {
        v = v128.load32_splat(ptrInput);
        ptrInput += 4;
        v = v128.extend_low<u8>(v);
        v = v128.extend_low<u16>(v);
        v = v128.convert<u32>(v);
        v = f32x4.div(v, NORMALIZE_VECTOR);
        v128.store(uIdx, v128.swizzle(v, U_PARTS));
        uIdx += 16;
        v128.store(vIdx, v128.swizzle(v, V_PARTS));
        vIdx += 16;
        linePixels -= 2;

        if (linePixels <= 0) {
            memory.copy(uIdx, uIdx - outputLineLength, outputLineLength);
            memory.copy(vIdx, vIdx - outputLineLength, outputLineLength);
            uIdx += outputLineLength;
            vIdx += outputLineLength;
            linePixels = lineWidthUV;
        }
    }
}

export function Nv12ToCHWTensor(inputNv12: i32, outputBuffer: i32, width: i32, height: i32): void {
    const size: i32 = width * height;
    // copy Y channel to the input buffer with normalization
    let ptrInput: i32 = inputNv12;
    let ptrY: i32 = outputBuffer;
    const yEnd: i32 = inputNv12 + size;
    while (ptrInput < yEnd) {
        store<f32>(ptrY, (load<u8>(ptrInput++) as f32) * NORMALIZE_VALUE);
        ptrY += 4;
    }

    let uIdx: i32 = ptrY;
    let vIdx: i32 = ptrY + size * 4;

    // copy U and V channels to the input buffer with normalization
    const yuvArrayLength: i32 = yEnd + size / 2; // 4:2:0 Y + 2 * (U/4) + 2 * (V/4)
    const outputLineLength: i32 = width * 4;
    let linePixels: i32 = width / 2; // U & V are half resolution horizontally (and vertically, but that's up to doubling the line)
    while (ptrInput < yuvArrayLength) {
        const uVal: f32 = (load<u8>(ptrInput, 0) as f32) * NORMALIZE_VALUE;
        const vVal: f32 = (load<u8>(ptrInput, 1) as f32) * NORMALIZE_VALUE;
        ptrInput += 2;
        linePixels--;

        store<f32>(uIdx, uVal);
        store<f32>(uIdx, uVal, 4);
        uIdx += 8;
        store<f32>(vIdx, vVal);
        store<f32>(vIdx, vVal, 4);
        vIdx += 8;

        // handle doubling the line
        if (linePixels === 0) {
            memory.copy(uIdx, uIdx - outputLineLength, outputLineLength);
            memory.copy(vIdx, vIdx - outputLineLength, outputLineLength);
            uIdx += outputLineLength;
            vIdx += outputLineLength;
            linePixels = width / 2;
        }
    }
}


const I1_PARTS: v128 = v128(0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7);
const I2_PARTS: v128 = v128(8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15);
export function I420ToCHWTensorOpt(inputI420: i32, outputBuffer: i32, width: i32, height: i32): void {
    const size: i32 = width * height;
    // copy Y channel to the input buffer with normalization
    let ptrInput: i32 = inputI420;
    let ptrOutput: i32 = outputBuffer;
    const yEnd: i32 = inputI420 + size;
    let v: v128;
    while (ptrInput < yEnd) {
        v = v128.load32_splat(ptrInput);
        v = v128.extend_low<u8>(v);
        v = v128.extend_low<u16>(v);
        v = v128.convert<u32>(v);
        store<v128>(ptrOutput, f32x4.div(v, NORMALIZE_VECTOR));
        ptrInput += 4;
        ptrOutput += 16;
    }

    const uvEnd = ptrInput + size / 2;
    const outputLineLength: i32 = width * 4;
    const lineWidthUV = width / 2;
    let linePixels: i32 = lineWidthUV;
    while(ptrInput < uvEnd) {
        v = v128.load32_splat(ptrInput); // assumes width is divisible by 8
        v = v128.extend_low<u8>(v);
        v = v128.extend_low<u16>(v);
        v = v128.convert<u32>(v);
        v = f32x4.div(v, NORMALIZE_VECTOR);
        ptrInput += 4;
        linePixels -= 4;
        v128.store(ptrOutput, v128.swizzle(v, I1_PARTS), 0);
        v128.store(ptrOutput, v128.swizzle(v, I2_PARTS), 16);
        ptrOutput += 32;
        if (linePixels <= 0) {
            memory.copy(ptrOutput, ptrOutput - outputLineLength, outputLineLength);
            ptrOutput += outputLineLength;
            linePixels = lineWidthUV;
        }
    }
}
