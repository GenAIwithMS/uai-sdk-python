# Vision

> Stub — Phase 2 content.

## Overview

Multi-modal vision operations: image captioning, OCR, object detection. Currently supported by: Qwen, StepFun, Doubao, MiniMax, Hunyuan.

## Usage

```python
result = client.vision(
    image="assets/photo.jpg",  # path or base64
    task="caption",  # or "ocr", "detect"
    model="qwen-vl-max",
)

print(result.caption)
```