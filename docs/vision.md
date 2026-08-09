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

`client.chat()` enforces the capability matrix **before** any network or
middleware work (Module 1.3.1): if any message carries an `ImageContent`
block and the target model does not advertise `vision`, the client raises
`FeatureNotSupportedError` instantly.

Pre-flight checks are also available if you want to choose a model before
building the request:

```python
client = UniversalAI(api_key="...", provider="qwen")
client.supports("vision", model="qwen-vl-max")   # -> True

from uai.registry import check_capability
check_capability("qwen", "qwen-vl-max", "vision")  # raises if unsupported
```

## Unsupported providers

DeepSeek, GLM, and Kimi do not expose vision models. Sending image content
to a text-only model raises `FeatureNotSupportedError` immediately — the
request never reaches the network. To switch providers on the fly:

```python
client.chat(messages=..., provider="qwen", model="qwen-vl-max")
```