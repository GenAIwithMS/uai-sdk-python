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

The registry tracks `vision` per model, and `check_capability` raises
`FeatureNotSupportedError` for models that do not advertise `vision: True`.
Use it as a pre-flight check before sending image content:

```python
from uai.registry import check_capability
check_capability("qwen", "qwen-vl-max", "vision")  # raises if unsupported
```

> **Note:** `client.chat()` currently gates on the `chat`/`streaming`
> capabilities only — it does not inspect message content for images. Use
> `check_capability` to guard vision calls explicitly.

## Unsupported providers

DeepSeek, GLM, and Kimi do not expose vision models — `check_capability`
raises `FeatureNotSupportedError` for them. Sending image content to a
text-only model is not gated by the client, so verify with
`check_capability` first.