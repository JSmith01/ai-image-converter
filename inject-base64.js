import { readFileSync, writeFileSync } from 'node:fs';

const wasmFile = readFileSync('build/optimized.wasm');
const base64String = wasmFile.toString('base64');
const tsFile = readFileSync('image-converter.ts');
const tsFileContent = tsFile.toString();
const updatedTsFileContent = tsFileContent.replace(
    /MODULE_SOURCE = '.*';/,
    () => `MODULE_SOURCE = '${base64String}';`
);
writeFileSync('build/image-converter-injected.ts', updatedTsFileContent);
