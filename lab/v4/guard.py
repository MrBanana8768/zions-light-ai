"""Local-only URL guard for the V4 lab.

Hard safety rule (owner's spec): nothing in the lab may talk to the live pod
or the live Synapse. Every configured URL (compactor, model, Synapse) must
resolve to localhost, a docker-compose service name, or a private address.
The bot, the importer and every lab script call `assert_local_only` on every
URL they are configured with, before doing anything else, and refuse to
start (non-zero exit, clear message) if the check fails.

This is deliberately dependency-free (stdlib only) so it can be imported by
every component (bot, importer, shim, one-off scripts) without pulling in
whatever HTTP/async stack that component happens to use.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import sys
from urllib.parse import urlparse

# Explicit denylist -- checked by substring/regex against the hostname
# *before* any DNS resolution, so a hosts-file trick or a resolver that lies
# can't get a live host waved through. These are named in the owner's spec.
_DENYLIST_PATTERNS = [
    re.compile(r"(^|\.)proxy\.runpod\.net$", re.IGNORECASE),
    re.compile(r"(^|\.)revelationsaints\.com$", re.IGNORECASE),
]

# Hostnames that are always fine without a DNS lookup at all -- pure
# loopback names. Anything else (including docker-compose service names)
# still has to resolve to a private/loopback address.
_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}


class LocalOnlyViolation(RuntimeError):
    """Raised when a configured URL is not provably local/private."""


def _is_private_or_loopback(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved  # keeps 0.0.0.0-style bind addresses from being fatal to check
    )


def check_url(name: str, url: str) -> tuple[bool, str]:
    """Return (ok, reason). Never raises -- callers decide how to fail."""
    if not url:
        return False, f"{name} is not set"
    try:
        parsed = urlparse(url if "://" in url else f"//{url}")
    except Exception as e:
        return False, f"{name}={url!r} could not be parsed ({e})"
    host = parsed.hostname
    if not host:
        return False, f"{name}={url!r} has no hostname"

    for pat in _DENYLIST_PATTERNS:
        if pat.search(host):
            return False, (
                f"{name}={url!r} host {host!r} matches the explicit denylist "
                f"(live pod / live Synapse are forbidden in this lab)"
            )

    if host in _LOOPBACK_NAMES:
        return True, "loopback name"

    # Resolve and require every returned address to be private/loopback.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        # A docker-compose service name that isn't resolvable *yet* (e.g. this
        # check runs before the network is up) is the one legitimate reason
        # this can fail even for a genuinely local target. Callers that need
        # to check before the network exists should retry; we do not treat
        # "can't resolve at all" as automatically safe.
        return False, f"{name}={url!r} host {host!r} did not resolve ({e})"

    ips = {info[4][0] for info in infos}
    if not ips:
        return False, f"{name}={url!r} host {host!r} resolved to no addresses"
    bad = {ip for ip in ips if not _is_private_or_loopback(ip)}
    if bad:
        return False, (
            f"{name}={url!r} host {host!r} resolved to public address(es) "
            f"{sorted(bad)} -- refusing (only localhost/private/compose-service "
            f"addresses are allowed in this lab)"
        )
    return True, f"resolved to {sorted(ips)}"


def assert_local_only(urls: dict[str, str], *, component: str) -> None:
    """Check every named URL; exit(1) with a clear message on any failure.

    `urls` maps a human label (e.g. "SYNAPSE_URL") to the URL string.
    """
    failures = []
    for name, url in urls.items():
        ok, reason = check_url(name, url)
        if not ok:
            failures.append(f"  - {name}={url!r}: {reason}")
    if failures:
        msg = (
            f"[{component}] REFUSING TO START: this is the LOCAL V4 lab and "
            f"it may never talk to the live pod or the live Synapse.\n"
            + "\n".join(failures)
            + "\nEvery URL must resolve to localhost, a docker-compose service "
            "name, or a private address. Fix the configuration and retry."
        )
        print(msg, file=sys.stderr)
        raise SystemExit(1)


def assert_local_only_or_raise(urls: dict[str, str], *, component: str) -> None:
    """Like assert_local_only, but raises LocalOnlyViolation instead of exiting.

    Used by code that wants to catch this (tests, or a caller that wants to
    log-and-alert-in-room rather than hard-exit the process).
    """
    failures = []
    for name, url in urls.items():
        ok, reason = check_url(name, url)
        if not ok:
            failures.append(f"{name}={url!r}: {reason}")
    if failures:
        raise LocalOnlyViolation(
            f"[{component}] refusing to start: " + "; ".join(failures)
        )


if __name__ == "__main__":
    # CLI use from shell scripts: guard.py NAME=URL [NAME=URL ...]
    component = sys.argv[1] if len(sys.argv) > 1 else "guard"
    pairs = {}
    for arg in sys.argv[2:]:
        if "=" not in arg:
            print(f"bad argument (want NAME=URL): {arg}", file=sys.stderr)
            raise SystemExit(2)
        k, v = arg.split("=", 1)
        pairs[k] = v
    assert_local_only(pairs, component=component)
    print("OK: all URLs are local/private.")
