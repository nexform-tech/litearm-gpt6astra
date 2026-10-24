from litearm_gpt6astra.astra.schema import ACTION_SCHEMA, REQUIRED_FIELDS


def test_schema_is_strict():
    assert ACTION_SCHEMA["additionalProperties"] is False
    assert set(ACTION_SCHEMA["required"]) == set(REQUIRED_FIELDS)


def test_schema_title_and_status():
    assert ACTION_SCHEMA["title"] == "litearm_astra_action"
    assert ACTION_SCHEMA["properties"]["status"]["enum"] == [
        "move",
        "grasp",
        "release",
        "hold",
        "done",
        "stop",
    ]


def test_schema_has_tcp_target():
    target = ACTION_SCHEMA["properties"]["tcp_target"]
    assert target["additionalProperties"] is False
    assert set(target["required"]) == {"position", "quaternion_wxyz"}
