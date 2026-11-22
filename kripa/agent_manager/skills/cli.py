"""CLI for inspecting and managing local, non-executable skill packages."""

import argparse
import json
import sys

from kripa.agent_manager.skills.manager import SkillManager


def _parser():
    parser = argparse.ArgumentParser(prog="python -m kripa agents skills")
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover", help="Discover skill packages in a local directory")
    discover.add_argument("directory")
    inspect = commands.add_parser("inspect", help="Inspect a local skill before installation")
    inspect.add_argument("directory")
    install = commands.add_parser("install", help="Install a skill after explicit digest approval")
    install.add_argument("directory")
    install.add_argument("--approve-sha256", required=True, help="Reviewed digest shown by the inspect command")
    commands.add_parser("list", help="List registered skills")
    for command in ("enable", "disable", "remove"):
        action = commands.add_parser(command, help="{} a registered skill".format(command.capitalize()))
        action.add_argument("skill_id")
    return parser


def run_skills(arguments=None, manager=None, stdout=None, stderr=None):
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    try:
        args = _parser().parse_args(list(arguments or []))
        manager = manager or SkillManager()
        if args.command == "discover":
            inspections = manager.discover(args.directory)
            if not inspections:
                print("No valid local skill packages found.", file=output)
            for item in inspections:
                print("{}\t{}\t{}\t{}".format(
                    item.skill_id, item.version,
                    "safe to review" if item.safe_to_consider else "blocked by checks",
                    item.sha256,
                ), file=output)
        elif args.command == "inspect":
            item = manager.inspect(args.directory)
            report = {
                "skill_id": item.skill_id,
                "name": item.name,
                "description": item.description,
                "version": item.version,
                "entrypoint": item.entrypoint,
                "compatibility": item.compatibility,
                "compatibility_issues": list(item.compatibility_issues),
                "security_issues": list(item.safety_issues),
                "sha256": item.sha256,
                "instructions": item.instructions if not item.safety_issues else "[hidden: possible credential material detected]",
                "installation_requires_explicit_approval": True,
            }
            print(json.dumps(report, indent=2, sort_keys=True), file=output)
        elif args.command == "install":
            skill = manager.install(args.directory, approval_digest=args.approve_sha256)
            print("Installed skill {} version {}.".format(skill.skill_id, skill.version), file=output)
        elif args.command == "list":
            skills = manager.list_skills()
            if not skills:
                print("No skills are registered.", file=output)
            for skill in skills:
                print("{}\t{}\t{}\t{}".format(
                    skill.skill_id, skill.version, "enabled" if skill.enabled else "disabled", skill.source,
                ), file=output)
        elif args.command == "enable":
            manager.enable(args.skill_id)
            print("Enabled skill.", file=output)
        elif args.command == "disable":
            manager.disable(args.skill_id)
            print("Disabled skill.", file=output)
        elif args.command == "remove":
            manager.remove(args.skill_id)
            print("Removed skill.", file=output)
        return 0
    except LookupError:
        print("Error: skill was not found.", file=errors)
        return 2
    except (OSError, TypeError, ValueError):
        # Never echo file contents, source paths, or arbitrary metadata on errors.
        print("Error: skill operation failed validation or approval checks.", file=errors)
        return 2
