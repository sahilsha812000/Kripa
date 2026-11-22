# Phase 6 End-to-End Verification Report

**Audit date:** 2026-11-22

**Scope:** Phase 6 tool execution and its boundaries with the existing runtime, provider fallback, and shadow router. No implementation code, provider configuration, `.env`, API key, or real user data was changed. No provider/API calls were made.

## Results

| # | Scenario | Status | What was tested and evidence | Weakness or limitation |
|---|---|---|---|---|
| 1 | Agent without a tool grant cannot execute | **PASS** | `tests/test_agent_manager_tool_executor.py::ToolExecutorTests::test_missing_agent_grant_is_denied` verifies a denied result and zero handler calls. | None found in this path. |
| 2 | Explicitly granted agent can execute `utc_time` | **PASS** | `test_authorized_read_only_utc_execution` executes the fixed UTC handler. `tests/test_agent_manager_runtime.py::AgentRuntimeRepositoryTests::test_default_executor_integrates_registered_read_only_tool_with_runtime` verifies runtime integration using a temporary store. | `utc_time` is the only built-in handler and reads the local clock. |
| 3 | Disabled tools cannot execute | **PASS** | `test_disabled_tool_is_denied` verifies a denied result and zero handler calls. | None found in this path. |
| 4 | Invalid arguments are rejected | **PASS** | `test_invalid_arguments_are_denied` submits an extra schema field and verifies denial before handler invocation. | None found in this path. |
| 5 | Policy denial never reaches the handler | **PASS** | `test_policy_deny_never_invokes_handler` verifies a missing-grant denial and zero handler calls. Unknown and disabled tool tests also verify zero calls. | None found in tested denial paths. |
| 6 | Approval-required action pauses without executing | **PASS** | `test_sensitive_action_pauses_until_explicit_approval` verifies a pending approval and no handler calls. Runtime test `test_sensitive_agent_tool_request_pauses_without_auto_approval` verifies a `waiting_approval` run and no second provider turn. | Sensitive built-in actions are intentionally not available. |
| 7 | Valid approval allows the intended action | **PASS** | In `test_approval_replay_is_rejected_after_one_execution`, the first call after approval executes the same bound action; the handler is called once. | The sensitive handler is a test fake; no sensitive production handler is enabled. |
| 8 | Approval cannot be reused | **PASS** | `test_approval_replay_is_rejected_after_one_execution` verifies the replay is denied. It uses `ExclusiveFileApprovalConsumptionRepository` in a temporary directory. | Consumption markers must be retained; manually deleting a marker could permit replay. |
| 9 | Concurrent attempts cannot execute the same approved action twice | **PASS** | An isolated temporary-storage exercise made 12 simultaneous calls through `ToolExecutor` using one real exclusive-file consumption ledger: 1 execution, 11 denials, 1 handler call. `ApprovalConsumptionRepositoryTests::test_exclusive_file_claim_allows_only_one_concurrent_consumer` separately verifies atomic claims across 24 contenders. | Atomicity applies to approval consumption on the local filesystem. The broader Agent Manager JSON snapshot store remains single-writer and non-transactional. |
| 10 | Changed arguments invalidate approval | **PASS** | `test_changed_arguments_are_denied_before_consumption` verifies changed arguments are denied, the approval is not consumed, and the handler is not called. | None found in this path. |
| 11 | Expired approvals are rejected | **PASS** | `test_expired_approval_is_rejected` advances a fixed test clock beyond expiry, verifies denial, expired status, and zero handler calls. | None found in this path. |
| 12 | Execution and audit records contain no raw arguments, secrets, or sensitive results | **FAIL** | `test_execution_results_and_audit_redact_secrets` verifies a credential-shaped argument/result is redacted from the returned result and JSONL audit; the audit does not include raw arguments or results. Pending run metadata stores argument/action digests rather than the argument object. | The guarantee is broader than the implementation: `_sanitize_result` uses a limited credential-pattern and sensitive-key filter, not general data-loss prevention, so arbitrary sensitive result text can pass through. Runtime also persists the user-supplied task text as `AgentRun.task`; that text is not automatically redacted. Thus this is not a general guarantee that no secrets or sensitive content are ever persisted or returned. |
| 13 | Failed execution is recorded safely | **PASS** | `test_handler_failure_is_sanitized_and_audited` makes a fake handler raise an error containing a credential-shaped value. The result is the fixed `handler_failed` reason and audit uses fixed `handler_error`; the value is absent. | The caller receives a generic failure and cannot inspect the underlying exception, by design. |
| 14 | Pending approval behavior after application restart | **PASS** | In an isolated temporary `AgentManagerStore`, a sensitive action was paused, then the store, repositories, executor, and runtime were reconstructed. The run remained `waiting_approval`, approval remained `pending`, the raw tool argument was absent, and the handler was not called. | Restart-safe **resumption is not implemented**: the pending `ToolRequest` is returned only in memory, there is no pending-request retrieval API, and the CLI has no approval/resume command. A restarted process cannot resume the pending action using persisted data alone. |
| 15 | Gemini-first provider fallback and Model Router shadow mode remain unchanged | **PASS** | No provider or router implementation was changed by this audit. `tests/test_engine.py::ProviderAdapterTests::test_factory_keeps_gemini_default_and_wraps_with_ollama_fallback` verifies Gemini remains the configured primary wrapped in `AdaptiveProvider`; the adaptive-provider tests cover fallback/recovery. `tests/test_model_router_engine.py::ShadowRoutingTests::test_shadow_router_reports_no_benchmark_evidence_without_changing_execution` verifies the current provider still answers while the router only reports an advisory decision. | These are offline tests with fake providers; no live Gemini request was made. |

## Verification commands

- Python runtime: `.venv/bin/python --version` → **Python 3.9.6**.
- Full offline suite: `.venv/bin/python -m unittest discover -s tests` → **161 tests passed**.
- `git diff --check` → passed for tracked changes; the new report also passed a no-index whitespace check.

## Overall assessment

The tested authorization, approval binding, expiry, replay prevention, concurrent one-time consumption, and fixed `utc_time` execution paths passed. No Phase 7 work was performed. The audit identified two limits: sensitive-result protection is pattern-based rather than comprehensive, and a pending approval cannot be resumed after process restart from persisted state alone. See scenarios 12 and 14.
