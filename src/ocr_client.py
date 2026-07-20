"""Thin wrapper around Umi-OCR's local HTTP API.

Umi-OCR must have its HTTP service started manually in the desktop app
(Toolbox tab -> HTTP OCR service -> Start service). Default port 1224.
"""

import base64
import os

import requests

UMI_OCR_HTTP_URL = os.environ.get("UMI_OCR_HTTP_URL", "http://127.0.0.1:1224")


class UmiOcrUnavailable(RuntimeError):
    """Raised when the local Umi-OCR HTTP service cannot be reached."""


def is_service_running(timeout: float = 1.5) -> bool:
    try:
        requests.get(UMI_OCR_HTTP_URL, timeout=timeout)
        return True
    except requests.RequestException:
        return False


def ocr_image_bytes(image_bytes: bytes, timeout: float = 30.0) -> str:
    """Send a single page image to Umi-OCR and return recognized text."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    try:
        resp = requests.post(
            f"{UMI_OCR_HTTP_URL}/api/ocr",
            json={"base64": b64},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise UmiOcrUnavailable(
            f"Could not reach Umi-OCR HTTP service at {UMI_OCR_HTTP_URL}. "
            "Make sure Umi-OCR is running and its HTTP service is started "
            "(Toolbox tab -> HTTP OCR service -> Start service)."
        ) from exc

    payload = resp.json()
    code = payload.get("code")

    if code == 100:  # success, text found
        lines = [item.get("text", "") for item in payload.get("data", [])]
        return "\n".join(lines)
    if code == 101:  # success, no text found on this page
        return ""

    raise UmiOcrUnavailable(f"Umi-OCR returned an error (code={code}): {payload.get('data')}")
