# Phase 4A: Tool Registry and Policy

Phase 4A adds a definition and authorization boundary only. It has no execution API and does not call a tool, provider, shell, browser, or computer.

## Modules

- `kripa/agent_manager/tools/contracts.py` defines typed `ALLOW`, `DENY`, and `REQUIRES_APPROVAL` decisions, risk levels, reason codes, and a redacted audit event.
- `schema.py` validates a deliberately limited JSON Schema subset. The root must be an object, object schemas must set `additionalProperties: false`, and unknown schema keywords are rejected. Argument keys/order are normalized using deterministic JSON serialization; non-JSON, non-finite, circular, oversized, or deeply nested values are denied.
- `registry.py` registers and inspects only existing `ToolDefinition` records. It accepts no callable field and rejects subclasses that could add executable behavior. Returned definitions are detached copies.
- `policy.py` applies agent enabled state, tool enabled state, explicit grant, schema validation, risk, approval binding and expiry. It makes decisions only; an `ALLOW` is not a tool invocation.
- `repositories.py` defines repository protocols and adapters. Tool definitions and approvals use the existing `AgentManagerStore`; explicit grants use `AgentDefinition.tool_ids` through `AgentRegistry`; audit is appended to `.kripa/agents/tool_audit.jsonl`.

## Authorization flow

`ToolPolicyService.authorize(agent_id, run_id, tool_id, arguments)` denies unknown/disabled agents or tools, missing grants, and invalid arguments. Low/medium-risk tools without an explicit approval requirement may return `ALLOW` after those checks. High/critical tools and any tool marked `requires_approval` create a pending approval and return `REQUIRES_APPROVAL`.

Approval is explicit through `decide_approval(approval_id, approved, decided_by)`. The caller must authenticate `decided_by` as an authorized human; this service is not an identity provider. Approvals bind agent ID, run ID, tool ID, normalized argument digest, a canonical action digest, and an expiry. The action digest includes the expiry. Reusing an approval for changed arguments, another run/agent/tool, or altered expiry is denied. Rejected and expired approvals cannot authorize an action.

The audit repository records policy outcome/reason, risk level, digests, and hashed agent/run/tool references. It never records raw arguments or prompts. If audit persistence fails, a would-be allow or approval request fails closed. The local JSONL audit file is created with restrictive permissions and symlink following is disabled where the platform supports it.

## Phase 4A limits and Phase 6 integration

- `ALLOW` means the policy checks passed; there is no executor in this phase.
- The approval transition method trusts its application caller to authenticate the human principal.
- The approval state model has no consumed state. Phase 6 adds a separate exclusive-create marker ledger for one-time consumption; it does not rely on concurrent JSON snapshot updates.
- The declared tool risk level remains the registry's classification. Operation-sensitive risk rules are still required for future tools whose risk depends on arguments or target state.
- Repository interfaces are intentionally small. The current JSON snapshot is suitable for this single-process phase, not concurrent workers or durable workflow execution.

See [TOOL_EXECUTION_PHASE_6.md](TOOL_EXECUTION_PHASE_6.md) for the executor boundary, approval consumption ledger, runtime integration, and remaining limitations.

No provider adapter, `AdaptiveProvider`, Model Router, provider setting, environment file, or fallback behavior is changed.
