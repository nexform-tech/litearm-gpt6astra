import json
from unittest.mock import patch

import numpy as np
import pytest

from litearm_gpt6astra.astra.client import AstraClient, Halt


class _Response:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self._body).encode()


def _completed(text):
    return {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
    }


def test_request_shape():
    with (
        patch.dict("os.environ", {"LITEARM_ASTRA_API_KEY": "k"}),
        patch("urllib.request.urlopen", return_value=_Response(_completed("{}"))) as urlopen,
    ):
        client = AstraClient("gpt-6-astra", "https://example/v1", "LITEARM_ASTRA_API_KEY")
        client.decide(
            {"task": "x"},
            [("head", np.zeros((8, 8, 3), dtype=np.uint8))],
        )
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        assert payload["model"] == "gpt-6-astra"
        assert payload["store"] is False
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["text"]["format"]["strict"] is True
        images = [item for item in payload["input"][0]["content"] if item["type"] == "input_image"]
        assert len(images) == 1


def test_http_error_not_retried():
    from urllib.error import HTTPError

    with (
        patch.dict("os.environ", {"LITEARM_ASTRA_API_KEY": "k"}),
        patch(
            "urllib.request.urlopen",
            side_effect=HTTPError("u", 500, "e", {}, None),
        ) as urlopen,
    ):
        client = AstraClient("gpt-6-astra", "https://example/v1", "LITEARM_ASTRA_API_KEY")
        with pytest.raises(Halt, match="HTTP"):
            client.decide({}, [])
        assert urlopen.call_count == 1


def test_missing_key_refused():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(Halt, match="credential"):
            AstraClient("gpt-6-astra", "https://example/v1", "LITEARM_ASTRA_API_KEY")


def test_refusal_is_halt():
    body = {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "refusal"}]}],
    }
    with (
        patch.dict("os.environ", {"LITEARM_ASTRA_API_KEY": "k"}),
        patch("urllib.request.urlopen", return_value=_Response(body)),
    ):
        client = AstraClient("gpt-6-astra", "https://example/v1", "LITEARM_ASTRA_API_KEY")
        with pytest.raises(Halt, match="refused"):
            client.decide({}, [])
