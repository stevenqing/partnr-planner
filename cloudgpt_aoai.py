from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Optional

from openai import AzureOpenAI

_CLOUDGPT_ENDPOINT = "https://cloudgpt-openai.azure-api.net/"
_CLOUDGPT_SCOPE = "api://feb7b661-cac7-44a8-8dc1-163b63c23df2/.default"
_CLOUDGPT_API_VERSION = "2024-06-01"
_TOKEN_REFRESH_MARGIN_SECONDS = 300

_token_lock = threading.Lock()
_cached_token: Optional[str] = None
_cached_expiry = 0


def _azure_cli_enabled() -> bool:
    return os.environ.get("CLOUDGPT_USE_AZURE_CLI", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_azure_cli_token() -> str:
    global _cached_expiry, _cached_token

    if not _azure_cli_enabled():
        raise RuntimeError("Set CLOUDGPT_USE_AZURE_CLI=1 to use CloudGPT")
    if shutil.which("az") is None:
        raise RuntimeError("Azure CLI is not available on PATH")
    now = int(time.time())
    with _token_lock:
        if (
            _cached_token is not None
            and now + _TOKEN_REFRESH_MARGIN_SECONDS < _cached_expiry
        ):
            return _cached_token
        completed = subprocess.run(
            [
                "az",
                "account",
                "get-access-token",
                "--scope",
                _CLOUDGPT_SCOPE,
                "--output",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        token = payload.get("accessToken")
        expiry = payload.get("expires_on")
        if not isinstance(token, str) or not token:
            raise RuntimeError("Azure CLI returned no CloudGPT access token")
        if not isinstance(expiry, int):
            raise RuntimeError("Azure CLI returned no numeric token expiry")
        _cached_token = token
        _cached_expiry = expiry
        return token


def get_openai_client(
    *,
    timeout: Any = 3600,
    max_retries: int = 5,
) -> AzureOpenAI:
    return AzureOpenAI(
        api_version=_CLOUDGPT_API_VERSION,
        azure_endpoint=_CLOUDGPT_ENDPOINT,
        azure_ad_token_provider=_get_azure_cli_token,
        timeout=timeout,
        max_retries=max_retries,
    )
