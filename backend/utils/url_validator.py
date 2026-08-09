"""SSRF-safe URL validation utilities."""

from __future__ import annotations

import ipaddress
from typing import Optional
from urllib.parse import urlparse


class SSRFViolation(Exception):
    """Raised when a URL is deemed unsafe for SSRF."""


def validate_url(url: str, *, allow_private: bool = False, allow_localhost: bool = False) -> str:
    """Validate a URL for SSRF safety.

    Args:
        url: The URL to validate.
        allow_private: Allow private/internal IP ranges.
        allow_localhost: Allow localhost/loopback addresses.

    Returns:
        The validated URL string.

    Raises:
        SSRFViolation: If the URL is unsafe.
    """
    if not isinstance(url, str) or not url.strip():
        raise SSRFViolation("URL must be a non-empty string")

    parsed = urlparse(url.strip())

    if parsed.scheme not in ("http", "https"):
        raise SSRFViolation(f"URL scheme '{parsed.scheme}' is not allowed. Only http and https are permitted.")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFViolation("URL must contain a valid hostname")

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return url

    if ip.is_loopback and not allow_localhost:
        raise SSRFViolation("Loopback addresses are not allowed")

    if ip.is_private and not allow_private:
        raise SSRFViolation("Private IP ranges are not allowed")

    if ip.is_link_local and not allow_localhost:
        raise SSRFViolation("Link-local addresses are not allowed")

    if ip.is_multicast:
        raise SSRFViolation("Multicast addresses are not allowed")

    if ip.is_reserved:
        raise SSRFViolation("Reserved IP ranges are not allowed")

    return url
