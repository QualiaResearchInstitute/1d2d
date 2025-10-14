from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib import request as urllib_request


@dataclass
class IndraClientOptions:
    base_url: str = "http://127.0.0.1:8787"
    timeout: float = 60


class IndraClient:
    def __init__(self, options: Optional[IndraClientOptions] = None):
        opts = options or IndraClientOptions()
        self._base_url = opts.base_url.rstrip("/")
        self._timeout = opts.timeout

    def health(self) -> bool:
        try:
            response = self._request("GET", "/health")
        except Exception:
            return False
        return response.get("status") == "ok"

    def render(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._require_keys(payload, ["input", "output"], "render")
        payload.setdefault("bitDepth", 8)
        response = self._request("POST", "/render", payload)
        return self._expect_ok(response)

    def simulate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._require_keys(payload, ["input"], "simulate")
        response = self._request("POST", "/simulate", payload)
        return self._expect_ok(response)

    def capture(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._require_keys(payload, ["input", "output"], "capture")
        response = self._request("POST", "/capture", payload)
        return self._expect_ok(response)

    def validate_manifest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "manifest" not in payload and "manifestPath" not in payload:
            raise ValueError('manifest validation requires "manifest" or "manifestPath".')
        response = self._request("POST", "/manifest/validate", payload)
        return self._expect_ok(response)

    def _expect_ok(self, response: Dict[str, Any]) -> Dict[str, Any]:
        status = response.get("status")
        if status != "ok":
            message = response.get("message", "Unknown error")
            raise RuntimeError(str(message))
        return {k: v for k, v in response.items() if k != "status"}

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        data: Optional[bytes] = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib_request.Request(url, data=data, method=method, headers=headers)
        with urllib_request.urlopen(req, timeout=self._timeout) as response:  # type: ignore[arg-type]
            payload = response.read().decode("utf-8")
            return json.loads(payload)

    @staticmethod
    def _require_keys(payload: Dict[str, Any], keys: list[str], operation: str) -> None:
        missing = [key for key in keys if key not in payload]
        if missing:
            raise ValueError(f"{operation} requires keys: {', '.join(missing)}")
