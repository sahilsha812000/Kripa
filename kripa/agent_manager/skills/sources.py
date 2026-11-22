"""Discovery source boundary for future external skill catalogs."""

from abc import ABC, abstractmethod
import os


class SkillDiscoverySource(ABC):
    """A source enumerates candidates; it does not install or execute them."""

    source_id = "abstract"

    @abstractmethod
    def discover(self, manager, location):
        """Return reviewed-candidate inspections from a source location."""
        raise NotImplementedError


class LocalDirectorySkillSource(SkillDiscoverySource):
    """Discover candidate package directories under a local filesystem path."""

    source_id = "local_directory"

    def discover(self, manager, location):
        root = os.path.abspath(location)
        if not os.path.isdir(root) or os.path.islink(root):
            raise ValueError("Skill discovery requires a real local directory.")
        results = []
        with os.scandir(root) as entries:
            candidates = sorted(entries, key=lambda item: item.name.lower())
            for entry in candidates:
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    continue
                try:
                    results.append(manager.inspect(entry.path))
                except (OSError, ValueError, TypeError):
                    continue
        return tuple(results)
