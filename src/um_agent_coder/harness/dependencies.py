"""
Dependency resolution for harness coordination.

Parses dependencies from roadmaps and provides dependency graph management
for determining execution order and detecting cycles.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HarnessDependency:
    """Dependency definition for a harness."""

    harness_id: str
    depends_on: List[str] = field(default_factory=list)
    artifacts_required: List[str] = field(default_factory=list)
    context_required: List[str] = field(default_factory=list)


class DependencyGraph:
    """
    Dependency graph for harness coordination.

    Manages dependencies between harnesses, detects cycles,
    and determines execution order.

    Example:
        graph = DependencyGraph()

        # Add dependencies
        graph.add_dependency("backend", ["database"])
        graph.add_dependency("frontend", ["backend", "auth"])
        graph.add_dependency("auth", ["database"])

        # Check for cycles
        if graph.has_cycle():
            print("Cycle detected!")

        # Get execution order
        order = graph.topological_sort()
        # ['database', 'auth', 'backend', 'frontend']

        # Get harnesses ready to run
        ready = graph.get_ready_harnesses(completed={"database"})
        # ['auth']
    """

    def __init__(self):
        """Initialize empty dependency graph."""
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_edges: Dict[str, Set[str]] = defaultdict(set)
        self.metadata: Dict[str, HarnessDependency] = {}

    def add_harness(
        self,
        harness_id: str,
        depends_on: Optional[List[str]] = None,
        artifacts_required: Optional[List[str]] = None,
        context_required: Optional[List[str]] = None,
    ) -> None:
        """Add a harness to the graph.

        Args:
            harness_id: Unique harness identifier
            depends_on: List of harness IDs this depends on
            artifacts_required: List of artifact names required
            context_required: List of context keys required
        """
        self.nodes.add(harness_id)
        depends_on = depends_on or []

        for dep in depends_on:
            self.nodes.add(dep)
            self.edges[harness_id].add(dep)
            self.reverse_edges[dep].add(harness_id)

        self.metadata[harness_id] = HarnessDependency(
            harness_id=harness_id,
            depends_on=depends_on,
            artifacts_required=artifacts_required or [],
            context_required=context_required or [],
        )

    def add_dependency(self, harness_id: str, depends_on: List[str]) -> None:
        """Add dependencies for a harness.

        Args:
            harness_id: Harness that has dependencies
            depends_on: List of harness IDs it depends on
        """
        self.add_harness(harness_id, depends_on=depends_on)

    def remove_harness(self, harness_id: str) -> None:
        """Remove a harness from the graph.

        Args:
            harness_id: Harness to remove
        """
        if harness_id in self.nodes:
            self.nodes.remove(harness_id)

        # Remove edges
        if harness_id in self.edges:
            for dep in self.edges[harness_id]:
                self.reverse_edges[dep].discard(harness_id)
            del self.edges[harness_id]

        if harness_id in self.reverse_edges:
            for dependent in self.reverse_edges[harness_id]:
                self.edges[dependent].discard(harness_id)
            del self.reverse_edges[harness_id]

        if harness_id in self.metadata:
            del self.metadata[harness_id]

    def get_dependencies(self, harness_id: str) -> Set[str]:
        """Get direct dependencies of a harness.

        Args:
            harness_id: Harness to check

        Returns:
            Set of harness IDs this depends on
        """
        return self.edges.get(harness_id, set())

    def get_dependents(self, harness_id: str) -> Set[str]:
        """Get harnesses that depend on this one.

        Args:
            harness_id: Harness to check

        Returns:
            Set of harness IDs that depend on this
        """
        return self.reverse_edges.get(harness_id, set())

    def get_all_dependencies(self, harness_id: str) -> Set[str]:
        """Get all transitive dependencies of a harness.

        Args:
            harness_id: Harness to check

        Returns:
            Set of all harness IDs this transitively depends on
        """
        visited = set()
        stack = list(self.edges.get(harness_id, set()))

        while stack:
            dep = stack.pop()
            if dep not in visited:
                visited.add(dep)
                stack.extend(self.edges.get(dep, set()) - visited)

        return visited

    def has_cycle(self) -> bool:
        """Check if graph has a cycle.

        Returns:
            True if cycle exists
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {node: WHITE for node in self.nodes}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor in self.edges.get(node, set()):
                if color.get(neighbor, WHITE) == GRAY:
                    return True
                if color.get(neighbor, WHITE) == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        for node in self.nodes:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False

    def find_cycles(self) -> List[List[str]]:
        """Find all cycles in the graph.

        Returns:
            List of cycles (each cycle is a list of node IDs)
        """
        cycles = []
        visited = set()
        rec_stack = []
        rec_set = set()

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.append(node)
            rec_set.add(node)

            for neighbor in self.edges.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_set:
                    # Found cycle
                    cycle_start = rec_stack.index(neighbor)
                    cycles.append(rec_stack[cycle_start:] + [neighbor])

            rec_stack.pop()
            rec_set.remove(node)

        for node in self.nodes:
            if node not in visited:
                dfs(node)

        return cycles

    def topological_sort(self) -> List[str]:
        """Get topological order of harnesses.

        Returns:
            List of harness IDs in dependency order

        Raises:
            ValueError: If graph has a cycle
        """
        if self.has_cycle():
            raise ValueError("Graph has a cycle, cannot topologically sort")

        in_degree = {node: 0 for node in self.nodes}
        for node in self.nodes:
            for dep in self.edges.get(node, set()):
                in_degree[node] = in_degree.get(node, 0) + 1

        # Nodes with no dependencies
        queue = [node for node in self.nodes if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for dependent in self.reverse_edges.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return result

    def get_ready_harnesses(
        self,
        completed: Optional[Set[str]] = None,
        running: Optional[Set[str]] = None,
    ) -> List[str]:
        """Get harnesses that are ready to run.

        A harness is ready if all its dependencies are completed.

        Args:
            completed: Set of completed harness IDs
            running: Set of currently running harness IDs

        Returns:
            List of harness IDs ready to run
        """
        completed = completed or set()
        running = running or set()
        ready = []

        for node in self.nodes:
            if node in completed or node in running:
                continue

            deps = self.edges.get(node, set())
            if deps.issubset(completed):
                ready.append(node)

        return ready

    def get_blocked_harnesses(
        self,
        completed: Optional[Set[str]] = None,
        failed: Optional[Set[str]] = None,
    ) -> List[str]:
        """Get harnesses blocked by failed dependencies.

        Args:
            completed: Set of completed harness IDs
            failed: Set of failed harness IDs

        Returns:
            List of blocked harness IDs
        """
        completed = completed or set()
        failed = failed or set()
        blocked = []

        for node in self.nodes:
            if node in completed or node in failed:
                continue

            deps = self.edges.get(node, set())
            if deps.intersection(failed):
                blocked.append(node)

        return blocked


class DependencyParser:
    """
    Parse dependencies from roadmap files.

    Extracts dependency information from roadmap.md files that
    use the `depends:` directive.

    Example roadmap format:
        ## Sub-Harnesses
        - **auth-harness**: Handle authentication
          - depends: database-harness
          - artifacts: auth-config
        - **api-harness**: Handle API
          - depends: auth-harness, database-harness
    """

    # Regex patterns for parsing
    HARNESS_PATTERN = re.compile(
        r"^\s*-\s+\*\*([^*]+)\*\*:\s*(.+)$", re.MULTILINE
    )
    DEPENDS_PATTERN = re.compile(
        r"^\s*-\s+depends:\s*(.+)$", re.MULTILINE | re.IGNORECASE
    )
    ARTIFACTS_PATTERN = re.compile(
        r"^\s*-\s+artifacts:\s*(.+)$", re.MULTILINE | re.IGNORECASE
    )
    CONTEXT_PATTERN = re.compile(
        r"^\s*-\s+context:\s*(.+)$", re.MULTILINE | re.IGNORECASE
    )

    def parse_roadmap(self, roadmap_path: Path) -> DependencyGraph:
        """Parse a roadmap file and build dependency graph.

        Args:
            roadmap_path: Path to roadmap.md file

        Returns:
            DependencyGraph with parsed dependencies
        """
        graph = DependencyGraph()
        content = Path(roadmap_path).read_text()

        # Find all harness definitions
        lines = content.split("\n")
        current_harness = None
        current_deps: List[str] = []
        current_artifacts: List[str] = []
        current_context: List[str] = []

        for line in lines:
            # Check for harness definition
            harness_match = self.HARNESS_PATTERN.match(line)
            if harness_match:
                # Save previous harness
                if current_harness:
                    graph.add_harness(
                        harness_id=current_harness,
                        depends_on=current_deps,
                        artifacts_required=current_artifacts,
                        context_required=current_context,
                    )

                current_harness = harness_match.group(1).strip()
                current_deps = []
                current_artifacts = []
                current_context = []
                continue

            # Check for depends directive
            depends_match = self.DEPENDS_PATTERN.match(line)
            if depends_match and current_harness:
                deps = depends_match.group(1).strip()
                current_deps = [d.strip() for d in deps.split(",") if d.strip()]
                continue

            # Check for artifacts directive
            artifacts_match = self.ARTIFACTS_PATTERN.match(line)
            if artifacts_match and current_harness:
                artifacts = artifacts_match.group(1).strip()
                current_artifacts = [a.strip() for a in artifacts.split(",") if a.strip()]
                continue

            # Check for context directive
            context_match = self.CONTEXT_PATTERN.match(line)
            if context_match and current_harness:
                context = context_match.group(1).strip()
                current_context = [c.strip() for c in context.split(",") if c.strip()]
                continue

        # Save last harness
        if current_harness:
            graph.add_harness(
                harness_id=current_harness,
                depends_on=current_deps,
                artifacts_required=current_artifacts,
                context_required=current_context,
            )

        return graph

    def parse_multiple(self, roadmap_paths: List[Path]) -> DependencyGraph:
        """Parse multiple roadmap files and merge dependency graphs.

        Args:
            roadmap_paths: List of roadmap file paths

        Returns:
            Merged DependencyGraph
        """
        graph = DependencyGraph()

        for path in roadmap_paths:
            sub_graph = self.parse_roadmap(path)
            for harness_id, meta in sub_graph.metadata.items():
                graph.add_harness(
                    harness_id=harness_id,
                    depends_on=meta.depends_on,
                    artifacts_required=meta.artifacts_required,
                    context_required=meta.context_required,
                )

        return graph
