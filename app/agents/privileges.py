"""Least-privilege tool registry + audit log (ported from estimator Session 14).

Every external side-effect (Cliniweb API calls, booking-URL generation) is a
*tool* registered with the set of nodes allowed to call it. A call from any
other node returns a denial envelope (teaching default) or raises
``PrivilegeViolation`` when ``strict=True``.

Every call — allowed or denied — is appended to an audit log with a SHA-256
digest of the full arguments, so a call's identity is provable without
dumping payloads into the log.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()

ARGS_PREVIEW_CHARS = 200


class PrivilegeViolation(Exception):
    """Raised in strict mode when a node calls a tool it is not allowed to."""


@dataclass
class AuditEntry:
    timestamp: float
    tool: str
    caller_node: str
    allowed: bool
    args_digest: str
    args_preview: str


@dataclass
class ToolRegistry:
    """Maps tool name -> set of node names allowed to invoke it."""

    permissions: dict[str, set[str]] = field(default_factory=dict)
    strict: bool = False
    audit_log: list[AuditEntry] = field(default_factory=list)

    def register(self, tool_name: str, allowed_nodes: set[str]) -> None:
        self.permissions[tool_name] = allowed_nodes

    def check(self, tool_name: str, caller_node: str, args: dict) -> bool:
        """Record the call attempt and return whether it is allowed."""
        allowed_nodes = self.permissions.get(tool_name, set())
        allowed = caller_node in allowed_nodes

        args_json = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(args_json.encode("utf-8")).hexdigest()

        entry = AuditEntry(
            timestamp=time.time(),
            tool=tool_name,
            caller_node=caller_node,
            allowed=allowed,
            args_digest=digest,
            args_preview=args_json[:ARGS_PREVIEW_CHARS],
        )
        self.audit_log.append(entry)

        log.info(
            "tool_call_audited",
            tool=tool_name,
            caller=caller_node,
            allowed=allowed,
            args_digest=digest[:16],
        )

        if not allowed and self.strict:
            raise PrivilegeViolation(
                f"Node '{caller_node}' is not allowed to call tool '{tool_name}'"
            )
        return allowed

    def denials(self) -> list[AuditEntry]:
        return [e for e in self.audit_log if not e.allowed]
