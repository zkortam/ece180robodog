"""The tool bus: merges Tier-0 local vision tools with Tier-1 MCP tools into one
flat tools[] list the LLM sees, and dispatches each call back to its owner.

For the MVP, MCP_SERVERS is empty, so this is a thin, well-tested wrapper around
VisionTools. The MCP plumbing (list/convert/route) is stubbed with clear seams so
Section 13 of the architecture can be filled in without touching the orchestrator.
"""
from . import config


class Policy:
    """Per-tool risk + allowlist. Vision tools are read-only and always allowed.
    Sensitive/destructive MCP tools require identity-gating (owner in view)."""

    def __init__(self, tau_high=config.IDENTITY_TAU_HIGH):
        self.tau_high = tau_high
        self.risk = {}          # name -> RISK_*
        self.allow = set()      # explicit allowlist for MCP tools

    def register(self, name, risk, allowed=True):
        self.risk[name] = risk
        if allowed:
            self.allow.add(name)

    def check(self, name, args, identity):
        """identity = {'owner_present':bool,'identity_score':float,'liveness':bool} or None."""
        risk = self.risk.get(name, config.RISK_READONLY)
        if risk in (config.RISK_READONLY, config.RISK_WRITE):
            # RISK_WRITE = local first-party enrollment; allowed today. Once sensitive MCP
            # actions exist, tighten enroll/train to require owner-in-view here.
            return True, None
        if name not in self.allow:
            return False, f"tool '{name}' is not on the allowlist"
        if identity is None or not identity.get("owner_present"):
            return False, "sensitive action requires the enrolled owner to be in view"
        if identity.get("identity_score", 0.0) < self.tau_high:
            return False, "owner identity confidence too low to authorize this action"
        if risk == config.RISK_DESTRUCTIVE and not identity.get("liveness"):
            return False, "destructive action requires a liveness / confirmation check"
        return True, None


class ToolBus:
    def __init__(self, vision_tools, mcp_sessions=None, policy=None):
        self.vision = vision_tools
        self.mcp_sessions = mcp_sessions or {}     # {server_id: client session}
        self.policy = policy or Policy()
        self._mcp_tools = {}                        # qualified_name -> (server_id, raw_name)
        self._schemas = []

    def build(self):
        """Assemble the merged schema list once (call again on MCP (re)connect)."""
        self._schemas = self.vision.schemas()      # Tier 0: local, in-process
        writers = {"enroll_face", "train_emotion"}
        for tool_name in self.vision.names():
            self.policy.register(tool_name,
                                 config.RISK_WRITE if tool_name in writers else config.RISK_READONLY)
        # Tier 1: MCP servers -> convert inputSchema ~1:1 -> namespaced OpenAI tool defs.
        for sid, sess in self.mcp_sessions.items():
            try:
                listed = sess.list_tools()          # official mcp client call
            except Exception:
                continue
            for t in getattr(listed, "tools", []):
                qn = f"{sid}__{t.name}"
                if qn not in self.policy.allow and self.policy.risk.get(qn) != config.RISK_READONLY:
                    # unknown/new tools are NOT auto-exposed (defeats rug-pull)
                    continue
                self._mcp_tools[qn] = (sid, t.name)
                self._schemas.append({"type": "function", "function": {
                    "name": qn, "description": t.description or "",
                    "parameters": t.inputSchema}})
        return self._schemas

    def schemas(self):
        return self._schemas or self.build()

    def dispatch(self, name, args, identity=None):
        ok, reason = self.policy.check(name, args, identity)
        if not ok:
            return {"error": "not_authorized", "reason": reason}
        if name in self.vision.names():             # Tier 0
            return self.vision.call(name, args)
        if name in self._mcp_tools:                 # Tier 1
            sid, raw = self._mcp_tools[name]
            try:
                out = self.mcp_sessions[sid].call_tool(raw, args or {})
                content = getattr(out, "content", None)
                if content and hasattr(content[0], "text"):
                    return content[0].text
                return out
            except Exception as e:
                return {"error": f"mcp tool {name} failed: {e}"}
        return {"error": f"unknown tool: {name}"}
