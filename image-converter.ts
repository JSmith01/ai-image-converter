const MODULE_SOURCE = 'BASE64';

let module: WebAssembly.Module | undefined;

function loadWasm(imports?: WebAssembly.Imports) {
    if (!module) {
        const s = atob(MODULE_SOURCE);
        const bytes = new Uint8Array(s.length);
        for (let i = 0; i < s.length; i++) {
            bytes[i] = s.charCodeAt(i);
        }

        module = new WebAssembly.Module(bytes);
    }

    return new WebAssembly.Instance(module, imports);
}

type i32 = number;

interface WasmFunctions {
    Nv12ToHW(inputNv12: i32, outputBuffer: i32, size: i32): void;
    Nv12ToCHW(inputNv12: i32, outputBuffer: i32, width: i32, height: i32): void;
    I420ToCHW(inputNv12: i32, outputBuffer: i32, width: i32, height: i32): void;
    I420TileToCHW(inputI420$: i32, outputBuffer$: i32, x: i32, y: i32, width: i32, height: i32, frameWidth: i32, frameHeight: i32): void;
    bilinearUpscaleChannel(src: i32, dst: i32, width: i32, height: i32): void;
}

const alignSizeTo16 = (n: number) => Math.ceil(n / 16) * 16;

class ImageConverter {
    static maxWidth = 1280;
    static maxHeight = 720;
    currentMaxSize: number;
    memory: WebAssembly.Memory;
    outputPtr: number;
    instance: WebAssembly.Instance;
    wasmHelpers: WasmFunctions;

    constructor(width = ImageConverter.maxWidth, height = ImageConverter.maxHeight) {
        this.adjustMemory(width, height);
        this.instance = loadWasm({ env: { memory: this.memory } });
        this.wasmHelpers = this.instance.exports as unknown as WasmFunctions;
    }

    _getAlignedYUVBufferSize() {
        // considering input is I420 or NV12, so U and V have quarter of Y (half in both vertical and horizontal)
        return alignSizeTo16(this.currentMaxSize * 3);
    }

    _getMemorySize() {
        const alignedYUVBufferSize = this._getAlignedYUVBufferSize();
        // 4 bytes per fp32 value, 3 full-size channels
        const alignedOutputBufferSize = alignSizeTo16(this.currentMaxSize * 12);

        return Math.ceil((alignedYUVBufferSize + alignedOutputBufferSize) / 65536);
    }

    adjustMemory(width: number, height: number) {
        if (this.currentMaxSize > 0 && width * height <= this.currentMaxSize) return;

        this.currentMaxSize = width * height;
        const pagesNeeded = this._getMemorySize();
        this.outputPtr = this._getAlignedYUVBufferSize();
        if (this.memory) {
            const pagesReserved = this.memory.buffer.byteLength / 65536;
            this.memory.grow(pagesNeeded - pagesReserved);
        } else {
            this.memory = new WebAssembly.Memory({ initial: pagesNeeded });
        }
    }

    getInputBufferView(width: number, height: number, channels = 3) {
        return new Uint8Array(this.memory.buffer, 0, width * height * (channels === 3 ? 1.5 : 1));
    }

    getOutputBufferView(width: number, height: number, channels = 3) {
        return new Float32Array(this.memory.buffer, this.outputPtr, width * height * channels);
    }

    // Y component only
    convertI420ToHW(width: number, height: number) {
        return this.convertNv12ToHW(width, height);
    }

    convertI420ToCHWBilinear(width: number, height: number) {
        const mem = new Uint8Array(this.memory.buffer);
        const fullPlaneSize = width * height;
        this.wasmHelpers.bilinearUpscaleChannel(fullPlaneSize, this.outputPtr, width / 2, height / 2); // U to *outputPtr
        this.wasmHelpers.bilinearUpscaleChannel(fullPlaneSize * 1.25, fullPlaneSize * 2, width / 2, height / 2); // V to *(fullPlaneSize * 2)
        mem.copyWithin(fullPlaneSize, this.outputPtr, this.outputPtr + fullPlaneSize);
        
        this.wasmHelpers.Nv12ToHW(0, this.outputPtr, fullPlaneSize);
        this.wasmHelpers.Nv12ToHW(fullPlaneSize, this.outputPtr + fullPlaneSize * 4, fullPlaneSize);
        this.wasmHelpers.Nv12ToHW(fullPlaneSize * 2, this.outputPtr + fullPlaneSize * 8, fullPlaneSize);
    }

    // Y component only
    convertNv12ToHW(width: number, height: number) {
        if (!this.wasmHelpers) throw new Error(`initialization hasn't finished yet`);

        console.assert((width * height) % 4 === 0, `Incorrect (width x height) = ${width}x${height} not divisible by 4`);

        this.wasmHelpers.Nv12ToHW(0, this.outputPtr, width * height);
    }

    convertNv12ToCHW(width: number, height: number) {
        if (!this.wasmHelpers) throw new Error(`initialization hasn't finished yet`);

        console.assert(width % 4 === 0, `Incorrect width = ${width} not divisible by 4`);

        this.wasmHelpers.Nv12ToCHW(0, this.outputPtr, width, height);
    }

    convertI420ToCHW(width: number, height: number) {
        if (!this.wasmHelpers) throw new Error(`initialization hasn't finished yet`);

        console.assert(width % 8 === 0, `Incorrect width = ${width} not divisible by 8`);

        this.wasmHelpers.I420ToCHW(0, this.outputPtr, width, height);
    }

    convertI420TileToCHW(x: number, y: number, width: number, height: number, frameWidth: number, frameHeight: number) {
        if (!this.wasmHelpers) throw new Error(`initialization hasn't finished yet`);

        console.assert(x % 4 === 0, `Incorrect x = ${x} not divisible by 8`);
        console.assert(width % 8 === 0, `Incorrect width = ${width} not divisible by 8`);

        this.wasmHelpers.I420TileToCHW(0, this.outputPtr, x, y, width, height, frameWidth, frameHeight);
    }

    convertNv12ToHWC(width: number, height: number) {
        throw new Error('Not implemented yet');
    }

    convertI420ToHWC(width: number, height: number) {
        throw new Error('Not implemented yet');
    }
}

export default ImageConverter;
