"""GPT-6 Astra 单步 EEF 目标的严格 JSON schema。"""

VECTOR3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
VECTOR4 = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}

REQUIRED_FIELDS = (
    "status",
    "tcp_target",
    "gripper",
    "confidence",
    "target_visible",
    "hand_visible",
    "path_clear",
    "near_grasp",
    "object_between_fingers",
    "tips_at_grasp_height",
    "object_in_target_zone",
    "uncertain",
    "reason",
)

ACTION_SCHEMA = {
    "title": "litearm_astra_action",
    "type": "object",
    "additionalProperties": False,
    "required": list(REQUIRED_FIELDS),
    "properties": {
        "status": {
            "type": "string",
            "enum": ["move", "grasp", "release", "hold", "done", "stop"],
        },
        "tcp_target": {
            "type": "object",
            "additionalProperties": False,
            "required": ["position", "quaternion_wxyz"],
            "properties": {
                "position": VECTOR3,
                "quaternion_wxyz": VECTOR4,
            },
        },
        "gripper": {"type": "integer", "enum": [0, 1]},
        "confidence": {"type": "number"},
        "target_visible": {"type": "boolean"},
        "hand_visible": {"type": "boolean"},
        "path_clear": {"type": "boolean"},
        "near_grasp": {"type": "boolean"},
        "object_between_fingers": {"type": "boolean"},
        "tips_at_grasp_height": {"type": "boolean"},
        "object_in_target_zone": {"type": "boolean"},
        "uncertain": {"type": "boolean"},
        "reason": {"type": "string"},
    },
}

STATUSES = tuple(ACTION_SCHEMA["properties"]["status"]["enum"])
EVIDENCE_FIELDS = (
    "target_visible",
    "hand_visible",
    "path_clear",
    "near_grasp",
    "object_between_fingers",
    "tips_at_grasp_height",
    "object_in_target_zone",
    "uncertain",
)
