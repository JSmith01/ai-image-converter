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
}

class ImageConverter {
    static maxDimension = 1920;

    static getPages() {
        return Math.ceil(ImageConverter.maxDimension * ImageConverter.maxDimension / 8192);
    }

    memory = new WebAssembly.Memory({ initial: ImageConverter.getPages() });
    outputPtr = this.memory.buffer.byteLength / 2;

    instance = loadWasm({ env: { memory: this.memory } });

    wasmHelpers: WasmFunctions = this.instance.exports as unknown as WasmFunctions;

    adjustMemory(width: number, height: number) {
        if (width * height <= ImageConverter.maxDimension * ImageConverter.maxDimension) return;

        const pagesReserved = ImageConverter.maxDimension;
        ImageConverter.maxDimension = Math.ceil(Math.max(width, height) / 4) * 4;
        const pagesNeeded = ImageConverter.getPages();
        this.memory.grow(pagesNeeded - pagesReserved);
        this.outputPtr = this.memory.buffer.byteLength / 2;
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
