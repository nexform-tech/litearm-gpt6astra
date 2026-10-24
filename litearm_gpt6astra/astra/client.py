"""OpenAI-compatible Responses API 客户端，只使用标准库 + OpenCV。

一次观测一次请求，不重试、不切换模型。失败抛 ``Halt``，由上层锁存。
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

import numpy as np

from .schema import ACTION_SCHEMA


class Halt(RuntimeError):
    """锁存失败：不得用另一个动作提案重试。"""


class AstraClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        key_env: str = "LITEARM_ASTRA_API_KEY",
        *,
        timeout_s: float = 30.0,
        proxy: str = "",
        instructions: str | None = None,
        schema: dict | None = None,
        schema_name: str = "litearm_astra_action",
        max_output_tokens: int = 2500,
    ):
        self.key = os.environ.get(key_env)
        if not self.key:
            raise Halt(f"missing API credential in environment variable {key_env}")
        if not model:
            raise Halt("an explicit Astra model ID is required")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/responses"
        self.timeout_s = float(timeout_s)
        self.instructions = instructions
        self.schema = ACTION_SCHEMA if schema is None else schema
        self.schema_name = schema_name
        self.max_output_tokens = int(max_output_tokens)
        self.opener = None
        if proxy:
            self.opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            )

    @property
    def key_present(self) -> bool:
        return bool(self.key)

    def decide(self, observation: dict, views: list[tuple[str, np.ndarray]]) -> dict:
        import cv2

        content = [
            {
                "type": "input_text",
                "text": json.dumps(observation, ensure_ascii=False, allow_nan=False),
            }
        ]
        for name, frame in views:
            if frame is None or frame.size == 0:
                raise Halt(f"empty camera frame: {name}")
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise Halt("JPEG encoding failed")
            content.append({"type": "input_text", "text": f"Camera: {name}"})
            content.append(
                {
                    "type": "input_image",
                    "detail": "high",
                    "image_url": "data:image/jpeg;base64,"
                    + base64.b64encode(encoded).decode("ascii"),
                }
            )
        body = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": self.schema_name,
                    "strict": True,
                    "schema": self.schema,
                }
            },
            "store": False,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.instructions:
            body["instructions"] = self.instructions
        request = urllib.request.Request(
            self.url,
            json.dumps(body).encode(),
            {
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
        )
        try:
            opener = self.opener.open if self.opener else urllib.request.urlopen
            with opener(request, timeout=self.timeout_s) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise Halt(
                f"Astra HTTP {exc.code}; no retry (check endpoint/model/credential)"
            ) from None
        except Exception as exc:
            raise Halt(f"Astra request failed ({type(exc).__name__}); no retry") from None

        if result.get("status") != "completed":
            raise Halt("Astra response incomplete")
        blocks = [
            block
            for item in result.get("output", [])
            if item.get("type") == "message"
            for block in item.get("content", [])
        ]
        if any(block.get("type") == "refusal" for block in blocks):
            raise Halt("Astra refused the request")
        texts = [block["text"] for block in blocks if block.get("type") == "output_text"]
        if len(texts) != 1:
            raise Halt("expected exactly one policy JSON response")
        try:
            return json.loads(texts[0])
        except (ValueError, TypeError):
            raise Halt("Astra did not return JSON") from None


class ChatCompletionsClient:
    """OpenAI-compatible Chat Completions 传输（/v1/chat/completions）。

    支持 ``image_url`` 内容块与 ``response_format=json_schema``。一次观测一次
    请求，不重试。
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        key_env: str = "LITEARM_ASTRA_API_KEY",
        *,
        timeout_s: float = 30.0,
        proxy: str = "",
        instructions: str | None = None,
        schema: dict | None = None,
        schema_name: str = "litearm_astra_action",
        max_output_tokens: int = 2500,
    ):
        self.key = os.environ.get(key_env)
        if not self.key:
            raise Halt(f"missing API credential in environment variable {key_env}")
        if not model:
            raise Halt("an explicit model ID is required")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/chat/completions"
        self.timeout_s = float(timeout_s)
        self.instructions = instructions
        self.schema = ACTION_SCHEMA if schema is None else schema
        self.schema_name = schema_name
        self.max_output_tokens = int(max_output_tokens)
        self.opener = None
        if proxy:
            self.opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            )

    @property
    def key_present(self) -> bool:
        return bool(self.key)

    def decide(self, observation: dict, views: list[tuple[str, np.ndarray]]) -> dict:
        import cv2

        content: list[dict] = [
            {
                "type": "text",
                "text": json.dumps(observation, ensure_ascii=False, allow_nan=False),
            }
        ]
        for name, frame in views:
            if frame is None or frame.size == 0:
                raise Halt(f"empty camera frame: {name}")
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise Halt("JPEG encoding failed")
            content.append({"type": "text", "text": f"Camera: {name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                        + base64.b64encode(encoded).decode("ascii"),
                        "detail": "high",
                    },
                }
            )
        messages: list[dict] = []
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        messages.append({"role": "user", "content": content})
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": self.schema_name,
                    "strict": True,
                    "schema": self.schema,
                },
            },
            "max_completion_tokens": self.max_output_tokens,
        }
        request = urllib.request.Request(
            self.url,
            json.dumps(body).encode(),
            {
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
        )
        try:
            opener = self.opener.open if self.opener else urllib.request.urlopen
            with opener(request, timeout=self.timeout_s) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise Halt(
                f"chat HTTP {exc.code}; no retry (check endpoint/model/credential): {detail}"
            ) from None
        except Exception as exc:
            raise Halt(f"chat request failed ({type(exc).__name__}); no retry") from None

        choices = result.get("choices") or []
        if not choices:
            raise Halt("chat response has no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        if message.get("refusal"):
            raise Halt("Astra refused the request")
        text = message.get("content")
        if isinstance(text, list):
            text = "".join(part.get("text", "") for part in text if part.get("type") == "text")
        if not isinstance(text, str) or not text.strip():
            raise Halt("chat response has no text content")
        if choice.get("finish_reason") in ("length", "content_filter"):
            raise Halt(f"chat response incomplete ({choice.get('finish_reason')})")
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            raise Halt("chat model did not return JSON") from None
