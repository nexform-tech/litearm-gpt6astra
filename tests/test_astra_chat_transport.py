import json
from unittest.mock import patch

import numpy as np
import pytest

from litearm_gpt6astra.astra.client import ChatCompletionsClient, Halt


class _Response:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self._body).encode()


def _chat(content):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
    }


def test_chat_request_shape():
    with (
        patch.dict("os.environ", {"LITEARM_ASTRA_API_KEY": "k"}),
        patch(
            "urllib.request.urlopen", return_value=_Response(_chat('{"status":"move"}'))
        ) as urlopen,
    ):
        client = ChatCompletionsClient(
            "gpt-6.1-sol",
            "https://mmu.vod-qcloud.com/v1",
            "LITEARM_ASTRA_API_KEY",
            instructions="system rules",
        )
        result = client.decide({"task": "x"}, [("head", np.zeros((8, 8, 3), dtype=np.uint8))])
        assert result == {"status": "move"}
        request = urlopen.call_args.args[0]
        assert request.full_url == "https://mmu.vod-qcloud.com/v1/chat/completions"
        payload = json.loads(request.data)
        assert payload["model"] == "gpt-6.1-sol"
        assert payload["messages"][0] == {"role": "system", "content": "system rules"}
        content = payload["messages"][1]["content"]
        assert payload["messages"][1]["role"] == "user"
        images = [item for item in content if item["type"] == "image_url"]
        assert len(images) == 1
        assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["max_completion_tokens"] == 2500


def test_chat_code_fence_parsed():
    with (
        patch.dict("os.environ", {"LITEARM_ASTRA_API_KEY": "k"}),
        patch(
            "urllib.request.urlopen",
            return_value=_Response(_chat('```json\n{"status":"hold"}\n```')),
        ),
    ):
        client = ChatCompletionsClient("gpt-6.1-sol", "https://example/v1", "LITEARM_ASTRA_API_KEY")
        assert client.decide({}, []) == {"status": "hold"}


def test_chat_http_error_not_retried():
    from urllib.error import HTTPError

    with (
        patch.dict("os.environ", {"LITEARM_ASTRA_API_KEY": "k"}),
        patch("urllib.request.urlopen", side_effect=HTTPError("u", 401, "e", {}, None)) as urlopen,
    ):
        client = ChatCompletionsClient("gpt-6.1-sol", "https://example/v1", "LITEARM_ASTRA_API_KEY")
        with pytest.raises(Halt, match="401"):
            client.decide({}, [])
        assert urlopen.call_count == 1


def test_chat_missing_key_refused():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(Halt, match="credential"):
            ChatCompletionsClient("gpt-6.1-sol", "https://example/v1", "LITEARM_ASTRA_API_KEY")


def test_chat_empty_content_refused():
    with (
        patch.dict("os.environ", {"LITEARM_ASTRA_API_KEY": "k"}),
        patch("urllib.request.urlopen", return_value=_Response(_chat(None))),
    ):
        client = ChatCompletionsClient("gpt-6.1-sol", "https://example/v1", "LITEARM_ASTRA_API_KEY")
        with pytest.raises(Halt, match="no text"):
            client.decide({}, [])
