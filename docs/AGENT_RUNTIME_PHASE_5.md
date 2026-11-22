# Phase 5: Single-Agent Runtime

Phase 5 adds a synchronous runtime for one explicitly selected agent task. It
uses the current Agent Manager records, provider factory and run collection.
It does not execute tools, delegate to other agents, activate Model Router, or
load previous runs as model context.

## Integration contracts verified

- `AgentDefinition` is retrieved through `AgentRegistry`; `enabled` is checked
  both before creating a run and immediately before execution.
- `SkillDefinition` is retrieved from the existing skill collection. Only
  enabled, local skills with the Phase 3 `human_digest_confirmation` marker and
  a SHA-256 package digest are included. Skill instructions are serialized as
  explicitly untrusted context.
- `AgentRun` and `AgentManagerStore` already provide persisted execution
  records. Phase 5 adds `pending` and `completed` states plus an optional
  `created_at`; existing legacy status strings and schema version 1 remain
  readable.
- `create_provider(get_settings())` is the same composition path used by chat.
  Its `reply(messages)` interface is wrapped by a small provider protocol. The
  provider instance is kept for the runtime lifetime so Gemini's existing
  `AdaptiveProvider` recovery and fallback state is preserved.
- The runtime has no imports from tool policy or Model Router modules and
  exposes no tool-execution method.

## Run lifecycle

`AgentRuntime.create_run(agent_id, task)` creates a unique pending record for
one active agent. `execute_run(run_id)` transitions it through `running` to
`completed` or `failed`. `cancel_run(run_id)` can cancel a pending run before
the provider request starts. The CLI combines create and execute:

```bash
.venv/bin/python -m kripa agents run <agent-id> "<task>"
```

Timestamps are timezone-aware ISO 8601 values. `created_at` is set while a run
is pending; `started_at` is set when execution begins; `completed_at` marks a
completed, failed, or cancelled run. Errors persisted in the run are fixed safe
summaries; SDK exception text is not stored or printed. The response is
returned to the caller but is not copied into run history. The selected task
is stored by `AgentRun` as execution history, and past tasks/results are not
retrieved for later runs.

## Context and boundaries

The runtime sends a bounded JSON task envelope through the provider's existing
`reply(messages)` method. It contains only the selected agent definition, its
currently eligible skill guidance, and this task. Agent model preferences are
not used for selection. The configured provider remains authoritative, so the
Gemini-first fallback path is unchanged.

Skill text is marked as untrusted reference input and cannot provide runtime
authorization. The runtime does not parse model output for actions and has no
tool executor, delegation facility, or memory retrieval path.

## Limitations

- Provider requests are synchronous. Cancellation can stop a pending run, but
  cannot interrupt a request already in progress because adapters expose no
  cancellation contract.
- The existing JSON store is atomic for each replacement but is not safe for
  concurrent writers or background workers.
- `AgentRun.task` is retained as local execution history. Use the existing
  restrictive local file permissions and do not put secrets in task text.
- The runtime trusts the stored Phase 3 approval marker and digest metadata;
  it does not independently reopen the original skill package at each run.
- Agent `model_preferences` are retained as metadata only; model selection
  remains the existing configured provider path.
