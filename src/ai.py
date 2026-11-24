"""
Optional Gemini client. Everything in the app works without it; when a key is set, AI writes the memo text
or reads documents and images. Set GEMINI_API_KEY (and optionally GEMINI_MODEL) in Streamlit secrets or the environment.

The AI never decides payment: it drafts, extracts and explains. A person reviews every output.
"""
import base64
import json
import os

import requests

DEFAULT_MODEL = "gemini-2.5-flash"
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _secret(name):
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def available():
    return bool(_secret("GEMINI_API_KEY"))


def model_name():
    return _secret("GEMINI_MODEL") or DEFAULT_MODEL


def generate(prompt, images=None, pdf=None, json_mode=False, timeout=60):
    """Return (text, error). images: list of (mime, bytes); pdf: bytes."""
    key = _secret("GEMINI_API_KEY")
    if not key:
        return None, "No GEMINI_API_KEY set"
    parts = [{"text": prompt}]
    for mime, data in images or []:
        parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}})
    if pdf:
        parts.append({"inline_data": {"mime_type": "application/pdf", "data": base64.b64encode(pdf).decode()}})
    body = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.2}}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    try:
        r = requests.post(URL.format(model=model_name()), params={"key": key}, json=body, timeout=timeout)
        if r.status_code != 200:
            return None, f"Gemini error {r.status_code}: {r.text[:300]}"
        cand = r.json().get("candidates", [])
        text = "".join(p.get("text", "") for p in cand[0]["content"]["parts"]) if cand else ""
        return text, None
    except Exception as e:  # network, timeout, parsing
        return None, f"Gemini call failed: {e}"


def generate_json(prompt, **kw):
    text, err = generate(prompt, json_mode=True, **kw)
    if err:
        return None, err
    try:
        return json.loads(text.strip().removeprefix("```json").removesuffix("```")), None
    except Exception:
        return None, "Gemini returned text that is not valid JSON"
