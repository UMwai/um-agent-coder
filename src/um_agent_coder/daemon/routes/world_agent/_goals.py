"""Goal store: load from YAML files, CRUD via Firestore."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from um_agent_coder.daemon.routes.world_agent import _firestore as store
from um_agent_coder.daemon.routes.world_agent.models import Goal

logger = logging.getLogger(__name__)

# In-memory cache (Firestore is source of truth, this avoids repeated reads)
_goals_cache: Dict[str, Goal] = {}


def load_goals_from_yaml(goals_dir: str = "goals/") -> List[Goal]:
    """Load goal definitions from YAML files in the given directory.

    Each YAML file should have a top-level `goal:` key with the goal definition.
    Returns list of parsed Goal objects.
    """
    goals_path = Path(goals_dir)
    if not goals_path.is_dir():
        logger.warning("Goals directory not found: %s", goals_dir)
        return []

    goals: List[Goal] = []
    for yaml_file in sorted(goals_path.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if not data or "goal" not in data:
                logger.warning("No 'goal' key in %s, skipping", yaml_file.name)
                continue
            goal = Goal(**data["goal"])
            goals.append(goal)
            logger.info("Loaded goal '%s' from %s", goal.id, yaml_file.name)
        except Exception as e:
            logger.error("Failed to parse goal from %s: %s", yaml_file.name, e)

    return goals


async def sync_goals_from_yaml(goals_dir: str = "goals/") -> List[Goal]:
    """Load goals from YAML and save to Firestore. Returns loaded goals."""
    goals = load_goals_from_yaml(goals_dir)
    for goal in goals:
        await store.save_goal(goal.model_dump())
        _goals_cache[goal.id] = goal
    logger.info("Synced %d goals from YAML to Firestore", len(goals))
    return goals


async def get_all_goals(status: Optional[str] = None) -> List[Goal]:
    """Get all goals from Firestore."""
    docs = await store.list_goals(status=status)
    goals = []
    for doc in docs:
        try:
            goal = Goal(**doc)
            goals.append(goal)
            _goals_cache[goal.id] = goal
        except Exception as e:
            logger.warning("Failed to parse goal doc: %s", e)
    return goals


async def get_goal(goal_id: str) -> Optional[Goal]:
    """Get a single goal by ID."""
    if goal_id in _goals_cache:
        return _goals_cache[goal_id]
    doc = await store.get_goal(goal_id)
    if not doc:
        return None
    try:
        goal = Goal(**doc)
        _goals_cache[goal.id] = goal
        return goal
    except Exception as e:
        logger.warning("Failed to parse goal %s: %s", goal_id, e)
        return None


async def create_goal(goal: Goal) -> bool:
    """Create a new goal in Firestore."""
    success = await store.save_goal(goal.model_dump())
    if success:
        _goals_cache[goal.id] = goal
    return success


async def update_goal(goal_id: str, updates: Dict) -> Optional[Goal]:
    """Update an existing goal. Merges updates into existing doc."""
    existing = await store.get_goal(goal_id)
    if not existing:
        return None
    existing.update(updates)
    success = await store.save_goal(existing)
    if not success:
        return None
    try:
        goal = Goal(**existing)
        _goals_cache[goal.id] = goal
        return goal
    except Exception as e:
        logger.warning("Failed to parse updated goal %s: %s", goal_id, e)
        return None
