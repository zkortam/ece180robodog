"""Tier-0 local perception tools: the functions the LLM may call, each bound to a
VisionService instance. This module is the single source of truth for the vision
tool schemas (also reused by vision_mcp_server.py, DRY).

Eight are read-only or camera on/off. Two -- enroll_face and train_emotion -- WRITE
biometric data to the local databases; tool_bus.py registers exactly those two as
RISK_WRITE, and vision_mcp_server.py deliberately does not expose them at all."""


def _schema(name, description, properties=None, required=None):
    props = properties or {}
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props,
                       "required": required or [], "additionalProperties": False}}}


# OpenAI-shaped tool definitions Cerebras sees. Grounded in VisionService (§5 of the
# arch doc). Order matters only for readability; the last two are the writers.
VISION_TOOL_SCHEMAS = [
    _schema("who_is_in_view",
            "Who is in front of the camera right now: enrolled names + how many unknown faces. Cheap, latest frame.",
            {"min_identity_score": {"type": "number",
             "description": "Only report known people at/above this SFace cosine confidence (0-1). Default 0."}}),
    _schema("describe_scene",
            "Full snapshot of everyone visible now: name, emotion, sentiment, rough position and size in frame.",
            {"include_probs": {"type": "boolean",
             "description": "Include the full 7-way emotion probability vector per person. Default false."}}),
    _schema("get_person_emotion",
            "Current/most-recent emotion and sentiment for one named, enrolled person.",
            {"name": {"type": "string", "description": "Enrolled identity name."}},
            ["name"]),
    _schema("emotion_timeline",
            "How a named person has been feeling over a recent window: dominant emotion + fractions + a short series. Use for anything about the past.",
            {"name": {"type": "string", "description": "Enrolled identity name."},
             "since_seconds": {"type": "number", "description": "Look-back window in seconds. Default 60."}},
            ["name"]),
    _schema("presence_events",
            "Who is present now and who entered or left recently. Use for 'did anyone walk in / leave'.",
            {"since_seconds": {"type": "number", "description": "Look-back window in seconds. Default 120."}}),
    _schema("list_enrolled",
            "The people registered in the system (face enrolled + which personal expressions they trained). Works even if the camera is off."),
    _schema("start_watching",
            "Turn the camera/perception loop on. Idempotent. Takes no arguments in normal use.",
            {"camera": {"type": "integer",
             "description": "Camera index. OMIT THIS unless the user names a specific camera: "
                            "the running index was auto-discovered and overriding it blinds the robot."},
             "fps": {"type": "integer",
             "description": "Frames per second. Omit to keep the configured rate."}}),
    _schema("stop_watching",
            "Turn the camera/perception loop off (history is kept)."),
    _schema("enroll_face",
            "Register a person: capture their face and save it under their name. Call as soon as they "
            "give their name and are looking at the camera. Do NOT ask them to confirm or spell it. "
            "Takes a few seconds.",
            {"name": {"type": "string", "description": "The person's name as you heard it."},
             "samples": {"type": "integer", "description": "How many shots to capture. Default 16."}},
            ["name"]),
    _schema("train_emotion",
            "Teach the system what one expression looks like for a registered person: capture their face "
            "while they hold that expression. Call once you have told them which face to make. Common "
            "expressions: neutral, happy, sad, surprise.",
            {"name": {"type": "string", "description": "An enrolled person's name."},
             "expression": {"type": "string", "description": "neutral | happy | sad | surprise | ..."},
             "samples": {"type": "integer", "description": "How many shots to capture. Default 16."}},
            ["name", "expression"]),
]


class VisionTools:
    """Binds the schemas above to a live VisionService and dispatches calls."""

    def __init__(self, vision_service):
        self.vs = vision_service
        self._impl = {
            "who_is_in_view": self.vs.who_is_in_view,
            "describe_scene": self.vs.describe_scene,
            "get_person_emotion": self.vs.get_person_emotion,
            "emotion_timeline": self.vs.emotion_timeline,
            "presence_events": self.vs.presence_events,
            "list_enrolled": self.vs.list_enrolled,
            "start_watching": self.vs.start_watching,
            "stop_watching": self.vs.stop_watching,
            "enroll_face": self.vs.enroll_face,
            "train_emotion": self.vs.train_emotion,
        }

    def schemas(self):
        return list(VISION_TOOL_SCHEMAS)

    def names(self):
        return set(self._impl.keys())

    def call(self, name, args):
        fn = self._impl.get(name)
        if fn is None:
            return {"error": f"unknown vision tool: {name}"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:  # never let a tool crash the turn
            return {"error": f"{name} failed: {e}"}
