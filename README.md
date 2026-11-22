# Kripa

Kripa is a personal AI assistant project. Its command-line chat supports
multi-turn conversations through Gemini, Groq, Ollama, OpenAI, and Anthropic.

## Getting started

Use Python 3.9 or newer. On macOS, create a project virtual environment with
`python3` (the system command `python` may not be installed), then install the
dependencies and run Kripa through that environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m kripa
```

Alternatively, activate the environment with `source .venv/bin/activate` and
use `python` for the commands below while it remains active.

The CLI starts a conversation with the configured provider. Type `/quit` to
exit.

## Local API key configuration

Kripa reads `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`,
`OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` from environment variables. For local development, copy
`.env.example` to `.env`
and replace the placeholders for the providers you plan to use:

```bash
cp .env.example .env
```

Keep `.env` on your machine and never commit it or share its contents. Git
ignores `.env`; `.env.example` contains placeholders only. Install the
dependencies with `.venv/bin/python -m pip install -r requirements.txt`. The settings
loader in `kripa/config.py` reads the variables (and loads `.env` when present).

## Choose a provider

Set `KRIPA_PROVIDER` in `.env` to `gemini`, `groq`, `openrouter`, `ollama`, `openai`, or `anthropic`
(`claude` is also accepted). Gemini is the default. The default models are
`gemini-3.5-flash-lite` for Gemini, `gpt-6.2-astra` for OpenAI, and
`claude-sonnet-5` for Anthropic. Groq uses `openai/gpt-oss-120b`. Ollama uses `llama3.2:3b` at
`http://localhost:11434`. OpenRouter uses `openai/gpt-oss-120b` by default.
Override cloud models with their corresponding `*_MODEL` variables. Override
Ollama with `OLLAMA_MODEL` or `OLLAMA_BASE_URL`. Install and start Ollama, then
download the model:

```bash
ollama pull llama3.2:3b
```

With Gemini selected, Kripa checks connectivity to Gemini on each turn. If
Gemini cannot be reached or its request fails, Kripa tries Groq, then
OpenRouter, then local Ollama. It retries Gemini on the next turn, so responses
return to Gemini automatically when it recovers. Add `GROQ_API_KEY` and/or
`OPENROUTER_API_KEY` to `.env` to enable those cloud fallbacks. Set
`KRIPA_PROVIDER=groq` or `KRIPA_PROVIDER=openrouter` to select either directly.
Select `ollama` to use the local provider directly. Ollama requires no API key.

Run the CLI:

```bash
.venv/bin/python -m kripa
```

To switch without editing `.env`, set the provider for one command:

```bash
KRIPA_PROVIDER=gemini .venv/bin/python -m kripa
KRIPA_PROVIDER=groq .venv/bin/python -m kripa
KRIPA_PROVIDER=openrouter .venv/bin/python -m kripa
KRIPA_PROVIDER=ollama .venv/bin/python -m kripa
KRIPA_PROVIDER=anthropic .venv/bin/python -m kripa
KRIPA_PROVIDER=openai .venv/bin/python -m kripa
```

The CLI and conversation interface are unchanged; the provider factory routes
each turn to the selected adapter. Gemini uses the Interactions API on Python
3.10 and newer. Python 3.9 uses the SDK's supported `generate_content` API and
resends local chat history. API calls happen when you send a message. The app
displays safe error messages for provider and connection failures and does not
log or print API keys or raw SDK exception details.

## Read-only model update report

Generate a model catalog and compatibility report without changing provider
settings or downloading models:

```bash
.venv/bin/python -m kripa models report
```

The report compares the configured Gemini, Groq, OpenRouter, and Ollama model
IDs with their available catalogs. Gemini and Groq use their account-scoped
model APIs when their keys are configured. OpenRouter uses its public model
catalog. Ollama lists locally installed models and compares their digests with
the registry manifest; this reads model metadata only and does not pull model
files. API keys are used internally for catalog requests and are never
included in the report.

Providers expose different lifecycle metadata. When a current ID is missing
from an account catalog, Kripa reports it as **not listed** because it may be
retired or unavailable to that account. Explicit inactive or expiration
metadata is reported as retired or deprecated when provided. Newer models are
suggestions based on catalog metadata and compatibility details, not automatic
configuration changes. Installing or selecting any model requires Nitin's
approval; this report does neither.

## Model Arena

Compare configured models on the same general conversation, reasoning, coding,
research, summarization, writing, and instruction-following prompts:

```bash
.venv/bin/python -m kripa arena
```

The arena runs each task three times against Gemini, Groq, OpenRouter, and
Ollama directly, without adaptive fallback. Select a subset by listing provider
names, for example `.venv/bin/python -m kripa arena gemini groq`. Every selected
provider gets the same task prompt on each run.

General conversation uses a concise practical-suggestion check. Research asks
for an answer grounded in a fixed source excerpt and checks its facts and
citation. Summarization checks required source facts in exactly two bullets.
These fixed checks are repeatable and do not make live web research calls. The
report separates correctness accuracy, average and median response time,
exact-response consistency, and provider-reported usage/cost. Reasoning uses a
fixed expected answer; instruction following checks the exact expected text.
Coding responses are parsed and checked against predefined cases by a small
allowlisted interpreter; generated code is never executed. Writing responses
are included for manual review and are not assigned an automatic quality score.
Failures are listed individually and included in success/failure counts. Token
counts and cost are shown only when the SDK returns explicit usage data; Kripa
does not infer prices. The arena does not change provider defaults, models, or
fallback settings. Running the arena makes provider requests and may incur
usage charges; the offline test suite uses mocks and does not make live calls.

Arena v2 per-task aggregates are saved locally in `.kripa/model_router.json`
with a 30-day expiry. The directory is ignored by Git and the data file is
restricted to the current user. It contains benchmark aggregates, provider
availability timestamps, and only capabilities explicitly registered with
verification evidence; it does not contain API keys. Malformed or unsupported
data is reported and is not silently overwritten.

Inspect the shadow router's local evidence and refresh stale provider status:

```bash
.venv/bin/python -m kripa router status
```

Availability checks are cached for five minutes. Use
`.venv/bin/python -m kripa router status --refresh` to force a refresh. This
explicit command may make read-only status requests to configured cloud APIs
and the local Ollama server; ordinary chat only reads cached data and never
probes providers. A capability is unknown until explicit verification evidence
is registered in code; benchmark performance alone does not claim capabilities.
This phase remains shadow-only and does not change conversation provider choice
or the existing fallback behavior.

## Agent Manager

Create an agent definition, then explicitly run one task with that agent:

```bash
.venv/bin/python -m kripa agents create editor --name "Editor" \
  --instructions "Edit the supplied text for clarity and correctness."
.venv/bin/python -m kripa agents run editor "Improve this paragraph: ..."
```

Each run is stored under `.kripa/agents/store.json` with its lifecycle state,
timestamps, selected agent, and task. Run history is not loaded into later
prompts. The runtime uses the configured provider factory; with Gemini selected,
it retains the current Gemini-first `AdaptiveProvider` fallback and recovery
behavior. Agent model preferences do not activate Model Router selection.

Only skills assigned to the selected agent that are enabled, locally sourced,
and carry the Phase 3 human digest approval marker are included. Their content
is labelled untrusted guidance. Agent tool IDs are an explicit allowlist. For
example, grant the built-in read-only UTC clock when creating an agent:

```bash
.venv/bin/python -m kripa agents create clock --name "Clock" \
  --instructions "Answer time questions." --tool utc_time
.venv/bin/python -m kripa agents run clock "What time is it?"
```

Phase 6 routes a model's single strict JSON tool proposal through the Tool
Registry and Tool Policy before the Tool Executor can call a fixed application
handler. The only built-in executable tool is `utc_time`; it reads the system
clock and has no network or filesystem access. Registering a tool definition
does not install a handler or grant an agent permission. Unknown, disabled,
ungranted, or invalid requests fail closed. A task accepts at most one tool
request, and tool results are sanitized before the provider sees them. Tool
execution does not run shell commands or Python code.

Sensitive requests pause with `waiting_approval` and an approval ID. An
application embedding `AgentRuntime` must authenticate the human approver,
call `decide_tool_approval`, then resume with the exact pending request. The
CLI reports the run, approval, and tool IDs; it does not yet provide an
approval/resume command, and there are currently no sensitive built-in tools.
Do not enable sensitive handlers until an authenticated approval UI is wired
to this flow. Approval use is claimed through exclusive file creation in
`.kripa/agents/consumed_approvals/`, so concurrent or replayed approvals fail
closed even though the existing JSON snapshot store has no transaction lock.
Markers are intentionally retained. Execution audits in
`.kripa/agents/tool_execution_audit.jsonl` contain only opaque references,
digests, outcomes, and safe error codes; raw arguments and results are not
logged. Tool definitions and policy approvals continue to use the existing
Agent Manager store and policy audit.
See [the Phase 6 design notes](docs/TOOL_EXECUTION_PHASE_6.md) for the approval
and storage guarantees and remaining CLI limitation.

Sub-agent delegation, Model Router activation, and persistent long-term memory
remain unavailable. Cancellation is supported before a pending run starts;
the synchronous provider interface cannot interrupt an in-flight request.

## Tests

Run the offline unit tests with:

```bash
.venv/bin/python -m unittest discover -s tests
```

To exercise a live provider, install dependencies, put its valid key in `.env`,
choose the provider as above, and run `.venv/bin/python -m kripa`. Send a short prompt and
type `/quit` when Kripa responds. Live checks require API access and may incur
provider usage charges.

## Project layout

- `kripa/` contains the application package.
- `kripa/app.py` holds the top-level application function.
- `kripa/config.py` loads API key settings from environment variables.
- `kripa/system_instruction.py` defines Kripa's shared identity and communication
  style for every provider.
- `kripa/engine/` contains conversation state, provider selection, and provider
  adapters.
- `kripa/engine/providers/gemini_provider.py` connects the conversation engine
  to Gemini's Interactions API.
- `kripa/engine/adaptive_provider.py` checks Gemini availability, tries Groq,
  OpenRouter, and then Ollama when needed, and retries Gemini on each new turn.
- `kripa/engine/providers/ollama_provider.py` connects to the local Ollama chat
  API without adding a third-party dependency.
- `kripa/engine/providers/groq_provider.py` connects to Groq's Chat Completions
  API through its official Python SDK.
- `kripa/engine/providers/openrouter_provider.py` connects to OpenRouter's
  Chat Completions API through the existing OpenAI SDK dependency.
- `kripa/model_updates/` discovers provider catalogs, compares configured
  models, and formats a read-only report.
- `kripa/model_arena/` runs repeated provider comparisons with answer checks and reports.
- `kripa/model_router/` classifies tasks and evaluates candidate models in shadow mode; its recommendations do not select the active provider.
- `kripa/agent_manager/tools/` contains the definition-only tool registry,
  deny-by-default policy, fixed-handler executor, one-time approval ledger,
  and redacted execution audit.
- `kripa/engine/usage.py` normalizes explicit token and cost metadata returned by providers.
- `tests/` contains offline unit tests for conversation flow and SDK adapters.
- `kripa/__main__.py` makes the package runnable with `.venv/bin/python -m kripa`.
- `kripa/__init__.py` identifies the directory as a package and exposes its
  version.
- `requirements.txt` is the place to list third-party runtime dependencies
  when they are needed.
- `.gitignore` excludes Python-generated files and common local environments.
- `.env.example` documents the required local environment variable names with
  placeholder values.

## Development

Keep modules focused on one responsibility and provider-specific API behavior
inside its adapter. Each provider has an independent adapter selected through
the factory and common reply interface. The task-aware router currently runs in
shadow mode: it classifies requests and evaluates only candidates with confirmed
availability, declared capabilities, and benchmark evidence. It does not change
the provider used for the response. The project uses the official Gemini, Groq,
OpenAI, and Anthropic SDKs plus `python-dotenv`; the Ollama adapter uses Python's
standard library.
