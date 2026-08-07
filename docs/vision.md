# Vision

## Overview

Vision capabilities are handled **through the chat API** using multimodal
content blocks, not via a dedicated `vision()` method. A `ChatMessage` can
contain a list of content blocks mixing text and images
(`ImageContent` → `ImageURL`), which the provider adapter translates into its
native multimodal schema.

Supported providers/models: Qwen (`qwen-vl-max`), StepFun (`stepfun-vision`),
Doubao (`doubao-vision`), MiniMax (`minimax-m2.5`), Hunyuan (`hunyuan-vision`).

## Usage

Pass an image via an `ImageURL` — either a data URI (base64) or an HTTP(S) URL:

```python
from uai import UniversalAI
from uai.models import ChatMessage, ImageContent, ImageURL, Role

client = UniversalAI(api_key="...", provider="qwen")

response = client.chat(
    messages=[
        ChatMessage(
            role=Role.USER,
            content=[
                ImageContent(image_url=ImageURL(url="https://example.com/photo.jpg")),
            ],
        ),
        ChatMessage(role=Role.USER, content="Describe this image."),
    ],
    model="qwen-vl-max",
)
print(response.content)
```

## Capability gating

Providers/models that do not advertise `vision: True` raise
`FeatureNotSupportedError`. Use `check_capability` to verify before calling:

```python
from uai.registry import check_capability
check_capability("qwen", "qwen-vl-max", "vision")  # raises if unsupported
```

## Unsupported providers

DeepSeek, GLM, and Kimi do not expose vision models; calling them with an
image content block raises `FeatureNotSupportedError`.