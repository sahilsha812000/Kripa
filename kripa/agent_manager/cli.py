"""Command-line management and explicit single-agent task execution."""

import argparse
import json
import sys

from kripa.agent_manager.contracts import AgentDefinition
from kripa.agent_manager.registry import (
    AgentAlreadyExistsError, AgentNotFoundError, AgentRegistry,
)


def _parser():
    parser = argparse.ArgumentParser(prog="python -m kripa agents")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="List agent definitions")

    create = commands.add_parser("create", help="Create an agent definition")
    create.add_argument("agent_id")
    create.add_argument("--name", required=True)
    create.add_argument("--instructions", required=True)
    create.add_argument("--description", default="")
    create.add_argument("--model", action="append", default=[], dest="model_preferences")
    create.add_argument("--skill", action="append", default=[], dest="skill_ids")
    create.add_argument("--tool", action="append", default=[], dest="tool_ids")
    create.add_argument("--parent")
    create.add_argument("--max-delegation-depth", type=int, default=0)

    show = commands.add_parser("show", help="View an agent definition")
    show.add_argument("agent_id")

    update = commands.add_parser("update", help="Update an agent definition")
    update.add_argument("agent_id")
    update.add_argument("--name")
    update.add_argument("--instructions")
    update.add_argument("--description")
    update.add_argument("--model", action="append", dest="model_preferences")
    update.add_argument("--skill", action="append", dest="skill_ids")
    update.add_argument("--tool", action="append", dest="tool_ids")
    update.add_argument("--parent")
    update.add_argument("--max-delegation-depth", type=int)

    retire = commands.add_parser("retire", help="Disable an agent definition")
    retire.add_argument("agent_id")

    run = commands.add_parser("run", help="Run one task with an explicitly selected agent")
    run.add_argument("agent_id")
    run.add_argument("task")
    return parser


def run_agents(arguments=None, registry=None, stdout=None, stderr=None, runtime=None):
    if arguments and arguments[0] == "skills":
        from kripa.agent_manager.skills.cli import run_skills

        return run_skills(arguments[1:], stdout=stdout, stderr=stderr)
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    parser = _parser()
    try:
        args = parser.parse_args(list(arguments or []))
        registry = registry or AgentRegistry()
        if args.command == "create":
            record = AgentDefinition(
                agent_id=args.agent_id, name=args.name, description=args.description,
                instructions=args.instructions, model_preferences=tuple(args.model_preferences),
                skill_ids=tuple(args.skill_ids), tool_ids=tuple(args.tool_ids),
                parent_agent_id=args.parent, max_delegation_depth=args.max_delegation_depth,
            )
            registry.create_agent(record)
            print("Created agent {}.".format(record.agent_id), file=output)
        elif args.command == "list":
            agents = registry.list_agents()
            if not agents:
                print("No agents are registered.", file=output)
            for agent in agents:
                state = "active" if agent.enabled else "retired"
                print("{}\t{}\t{}".format(agent.agent_id, state, agent.name), file=output)
        elif args.command == "show":
            print(json.dumps(_safe_agent_view(registry.get_agent(args.agent_id)), indent=2, sort_keys=True), file=output)
        elif args.command == "update":
            changes = {}
            for name in ("name", "instructions", "description", "model_preferences", "skill_ids", "tool_ids", "max_delegation_depth"):
                value = getattr(args, name, None)
                if value is not None:
                    changes[name] = tuple(value) if name in {"model_preferences", "skill_ids", "tool_ids"} else value
            if args.parent is not None:
                changes["parent_agent_id"] = args.parent
            record = registry.update_agent(args.agent_id, **changes)
            print("Updated agent {}.".format(record.agent_id), file=output)
        elif args.command == "retire":
            record = registry.retire_agent(args.agent_id)
            print("Retired agent {}.".format(record.agent_id), file=output)
        elif args.command == "run":
            from kripa.agent_manager.runtime import AgentRuntime
            from kripa.agent_manager.tools.executor import build_default_tool_executor

            runtime = runtime or AgentRuntime(
                tool_executor=build_default_tool_executor(),
            )
            result = runtime.run(args.agent_id, args.task)
            if result.run.status == "failed":
                print("Error: {}".format(result.run.error_summary), file=errors)
                if result.run.metadata.get("executed_tool_id"):
                    print(
                        "Tool {} completed before the run failed; review before retrying.".format(
                            result.run.metadata["executed_tool_id"],
                        ),
                        file=errors,
                    )
                print("Run ID: {}".format(result.run.run_id), file=errors)
                return 1
            if result.run.status == "cancelled":
                print("Agent task was cancelled.", file=errors)
                print("Run ID: {}".format(result.run.run_id), file=errors)
                return 1
            if result.run.status == "waiting_approval":
                print("Approval required; no tool was executed.", file=output)
                print("Run ID: {}".format(result.run.run_id), file=output)
                print("Approval ID: {}".format(result.approval_id or ""), file=output)
                if result.pending_tool_request is not None:
                    print("Requested tool: {}".format(result.pending_tool_request.tool_id), file=output)
                return 0
            print(result.response or "", file=output)
            if result.run.metadata.get("tool_execution_audit_warning"):
                print(
                    "Warning: the tool completed, but its completion audit could not be recorded.",
                    file=errors,
                )
            print("Run ID: {}".format(result.run.run_id), file=output)
        return 0
    except AgentAlreadyExistsError:
        print("Error: an agent with that ID already exists.", file=errors)
        return 2
    except AgentNotFoundError:
        print("Error: the requested agent was not found.", file=errors)
        return 2
    except LookupError:
        print("Error: the requested agent or run was not found.", file=errors)
        return 2
    except (TypeError, ValueError):
        # Do not echo arbitrary arguments or stored values, which could contain secrets.
        print("Error: agent definition or command arguments are invalid.", file=errors)
        return 2


def _safe_agent_view(agent):
    """Avoid printing instructions or arbitrary metadata that may hold secrets."""
    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "description": agent.description,
        "instructions_configured": bool(agent.instructions),
        "model_preferences": list(agent.model_preferences),
        "skill_ids": list(agent.skill_ids),
        "tool_ids": list(agent.tool_ids),
        "parent_agent_id": agent.parent_agent_id,
        "max_delegation_depth": agent.max_delegation_depth,
        "enabled": agent.enabled,
    }
