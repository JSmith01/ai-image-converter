/*
import assert from "assert";
import { add } from "../build/debug.js";
assert.strictEqual(add(1, 2), 3);
console.log("ok");

*/
import { readFile } from 'node:fs/promises';
import { hrtime } from 'node:process';

const imports = {
    env: { memory: new WebAssembly.Memory({ initial: 10000 }) },
  };

(async () => {
    const moduleBinary = await readFile('build/debug.wasm');
    const module = await WebAssembly.compile(moduleBinary);
    const instance = await WebAssembly.instantiate(module, imports);

    const { buffer } = imports.env.memory;
    const wasm = instance.exports;

    const w = 1920;
    const h = 1080;
    const inputPtr = 0;
    const size = w * h;
    const outputPtr = size * 1.5;
    const input = new Uint8Array(buffer, inputPtr, size);
    for(let i = 0; i < size * 1.5; i++) {
        input[i] = i % 256;
    }

    const output = new Float32Array(buffer, outputPtr, size);

    const NS_PER_SEC = 1e9;

    const iterations = 300;

    function iterate(descr, fnName, ...args) {
        output.fill(0);
        const t = hrtime();
        for (let i = 0; i < iterations; i++) {
            wasm[fnName](inputPtr, outputPtr, ...args);
        }
        const dt = hrtime(t);
        console.log(`${descr} x${iterations} took ${(dt[0] * NS_PER_SEC + dt[1]) / 1_000_000} ms`);
    }

    iterate('Naive SIMD', 'Nv12ToHWTensor', size);
    iterate('SIMD', 'Nv12ToHWTensorOpt', size);
    iterate('Deopt', 'Nv12ToHWTensorDeopt', size);
    iterate('Nv12ToCHW SIMD', 'Nv12ToCHWTensorOpt', w, h);
    // console.log(Array.from(input.slice(0, size * 1.5)));
    // console.log(Array.from(output.slice(0, size * 3)).map(v => v * 255));
    iterate('Nv12ToCHW non-SIMD', 'Nv12ToCHWTensor', w, h);
    iterate('I420ToCHW SIMD', 'I420ToCHWTensorOpt', w, h);
})();
