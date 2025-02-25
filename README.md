# Image converter for use in AI (WASM SIMD optimized)

The ImageConverter class provides with following methods:

* `convertNv12ToHW` - converts Y channel of a given frame to HW array of f32
* `convertNv12ToCHW` - converts full YUV frame in NV12 to CHW f32 representation
* `convertI420ToCHW` - converts full YUV frame in I420 to CHW f32 representation
* `convertI420TileToCHW` - takes a tile from I420 frame and puts it as YUV f32 normalized CHW

All methods assume the input frame will be at 0 offset of the internal `memory` buffer,
output appears at the offset `imageConverterInstance.outputPtr`.

For convenience the class provides two methods `getInputBufferView` and
`getOutputBufferView` for input and output accordingly.

Initial assumption for the class is the max image size of **1920 x 1920**. If bigger image
is required, it needs to either changing `ImageConverter.maxDimension` before instantiating
the class, or in runtime `imageConverterInstance.adjustMemory(width, height)` method can be used.


# Implementation details

It uses AssemblyScript (TS subset) to compile to WASM functions.
It intentionally does not use function calls, to prevent stack use,
so every function is independent. All of those operate on the same memory buffer,
and it's possible to extract data from `VideoFrame` objects to it directly using
[MediaStreamTrackProcessor](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrackProcessor) API.


## Methods to implement

* `convertHWToI444`
* `convertCHWToI444`
* `convertNV12TileToCHW`

Potentially it'd be nice to have a basic ONNX wrapper to make usual and tiled inference.
