import json

def parse_json_response(raw_text, expected_keys=None):
    """
    Strip markdown fences if present, parse JSON, and optionally verify
    expected keys exist. Raises on any failure — callers catch and fail-safe.
    """
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    parsed = json.loads(cleaned)

    if expected_keys:
        for key in expected_keys:
            if key not in parsed:
                raise KeyError(f"missing expected key: {key}")

    return parsed