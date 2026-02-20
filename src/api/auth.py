"""
API Key authentication middleware.
"""

import os
from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key: str = Security(_api_key_header)) -> str:
    expected = os.getenv("API_KEY", "")
    if not expected:
        # No key configured → open access (dev mode)
        return "dev"
    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )
    return api_key
