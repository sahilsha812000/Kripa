# Phase 6: Safe Tool Execution

Phase 6 connects the existing definition-only Tool Registry and Tool Policy to
the Phase 5 single-agent runtime. The executor accepts a model proposal only
after policy authorization; registry records never contain or load callables.

## Request and execution path

1. The runtime advertises only enabled, explicitly granted tools that have a
   matching fixed application handler.
2. The provider may return one exact JSON object with a
   `kripa_tool_request` field. Duplicate JSON object keys and additional
   wrapper fields are rejected. This text is a proposal, not permission.
3. `ToolExecutor` asks `ToolPolicyService` to verify agent/tool state, explicit
   grant, strict argument schema, risk, and any approval. It normalizes
   arguments and uses the policy action digest.
4. For an allowed call, the executor confirms that the registered definition
   matches the fixed handler, writes a durable preflight audit event, and
   repeats policy validation immediately before calling that handler.
5. The result is limited to JSON values, size/depth bounded, and credential
   patterns and credential-shaped fields are redacted before it returns to the
   runtime. The runtime sends at most one sanitized result back to the provider
   for a final answer; a second tool proposal ends the run safely.

The first and only built-in handler is `utc_time`. It has no arguments and
reads the local system clock. It has no network, file, shell, Python, browser,
or computer access. Its definition is registered when the default tool
executor is composed, but this does not grant it to any agent. An agent must
explicitly list `utc_time` in `AgentDefinition.tool_ids`.

## Approval and one-time consumption

High-risk, critical, and explicitly approval-required actions use the existing
Phase 4 approval lifecycle. The runtime persists a `waiting_approval` run with
the approval ID, tool ID, and digests, not the raw arguments. The pending
`ToolRequest` is returned to the in-process caller. After an authenticated
human decision, the caller resumes using that same request. The runtime
compares its canonical argument digest with the paused run, and the executor
compares the expected action digest before execution. The policy then
revalidates the approved state, exact agent/run/tool/arguments, and expiry
again immediately before invoking the handler.

`AgentManagerStore` replaces its JSON snapshot atomically but has no
transaction or compare-and-swap operation, so it cannot guarantee single-use
approval consumption under concurrent processes. Phase 6 therefore adds an
independent immutable marker ledger under
`.kripa/agents/consumed_approvals/`. A claim uses exclusive file creation
(`O_CREAT | O_EXCL`) through a no-follow directory descriptor. Only one process
can claim an approval ID. The marker contains a hashed approval reference, the
approved action digest, and a timestamp; it is created with user-only
permissions, synced, and retained so deletion cannot permit replay. Any
storage or audit failure denies execution. A claimed marker is not removed if
later validation fails, which favors safety over retry availability.

## Audits and errors

The existing Phase 4 policy audit remains in use. Execution events are appended
to `.kripa/agents/tool_execution_audit.jsonl` with user-only file permissions.
They contain hashed agent/run/tool/approval references, action/result digests,
fixed reason codes, and fixed error codes. They never contain prompts, raw
arguments, raw tool results, or SDK/provider exceptions. A durable preflight
event is required before execution. Handler exceptions are replaced with a
safe failure code and message. If the handler completes but the completion
audit write fails, the executor reports `EXECUTED` with an
`execution_completed_audit_unavailable` reason and returns the sanitized result;
it does not mislabel an already completed action as retryable failure. The
durable preflight event remains as evidence that execution was entered. The
run record retains the executed tool ID if the follow-up provider response
fails, and the CLI warns before a caller retries that task.

## Runtime and approval UI boundary

The Phase 5 provider factory and `AdaptiveProvider` are unchanged. Agent runs
continue through the existing configured provider path, including
Gemini-first fallback. The Model Router remains in shadow mode. The runtime
supports `decide_tool_approval` and `resume_run`, but `decide_tool_approval`
trusts the application caller to authenticate `decided_by`. The current CLI
prints the run, approval, and tool IDs when a run pauses; it does not yet offer
an approval/resume command or durable retrieval of the in-memory pending
request. There are no sensitive built-in handlers, and sensitive tools should
not be enabled until an authenticated approval UI and restart-safe resume
design are provided.

## Scope and remaining boundaries

- One explicitly selected agent and at most one tool request per run.
- No tool chaining, delegation, persistent long-term memory, or Model Router
  activation.
- No general-purpose shell or Python execution, arbitrary filesystem access,
  network browser, or computer control.
- The JSON snapshot store still does not support concurrent definition or run
  updates. The approval marker ledger only closes the one-time-consumption
  race; it does not turn the full Agent Manager store into a transactional
  repository.
- The result sanitizer is a conservative credential-pattern filter, not a
  general data-loss-prevention system. New handlers need tool-specific
  argument/result policy and tests before registration.

## Tests

Offline tests cover policy denials, the granted UTC read-only tool, sensitive
approval pause/rejection/expiry, changed arguments, action digest validation,
approval replay, handler failure, audit/result redaction, failed preflight
audit, concurrent marker consumption, and fake-executor runtime integration.
No test makes a provider or live tool call.
