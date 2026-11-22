"""Agent Manager JSON-store adapter for single-agent execution."""

from kripa.agent_manager.registry import AgentRegistry
from kripa.agent_manager.skills.manager import SkillManager
from kripa.agent_manager.store import AgentManagerStore


class AgentManagerRuntimeRepository:
    """Adapt existing agent, skill, and run collections to runtime operations."""

    def __init__(self, store=None):
        self.store = store or AgentManagerStore()
        self.agents = AgentRegistry(self.store)
        self.skills = SkillManager(self.store)

    def get_agent(self, agent_id):
        return self.agents.get_agent(agent_id)

    def get_skill(self, skill_id):
        return self.skills.get_skill(skill_id)

    def create_run(self, run):
        snapshot = self._snapshot()
        if any(item.run_id == run.run_id for item in snapshot.runs):
            raise ValueError("An agent run with that ID already exists.")
        self.store.upsert("runs", run)
        return run

    def get_run(self, run_id):
        for run in self._snapshot().runs:
            if run.run_id == run_id:
                return run
        raise LookupError("Agent run was not found.")

    def save_run(self, run):
        self.store.upsert("runs", run)
        return run

    def _snapshot(self):
        snapshot = self.store.load()
        if snapshot.issues:
            raise ValueError("Agent Manager storage is incomplete or unsupported.")
        run_ids = [run.run_id for run in snapshot.runs]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("Agent Manager storage contains duplicate run IDs.")
        return snapshot
