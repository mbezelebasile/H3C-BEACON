"""
SMAClite Environment Wrapper
=============================
StarCraft Multi-Agent Challenge — lite (pure Python reimplementation).

Wraps `uoe-agents/smaclite` (Gymnasium API) to fit the H3C-BEACON BaseEnv interface.
SMAClite does NOT require StarCraft II — it's a NumPy-based reimplementation.

Reference:
    Michalski et al., 2023 - "SMAClite: A Lightweight Environment for
    Multi-Agent Reinforcement Learning"
    https://github.com/uoe-agents/smaclite

Author: H3C-BEACON Research Team
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_env import BaseEnv

logger = logging.getLogger(__name__)


# Scenarios shipped with SMAClite (matches scenario JSON files)
SMACLITE_SCENARIOS = (
    "10m_vs_11m",
    "27m_vs_30m",
    "2c_vs_64zg",
    "2s3z",
    "2s_vs_1sc",
    "3s5z",
    "3s5z_vs_3s6z",
    "3s_vs_5z",
    "bane_vs_bane",
    "corridor",
    "mmm",
    "mmm2",
)


class SMACliteWrapper(BaseEnv):
    """
    Wrapper for SMAClite (Gymnasium API).

    SMAClite reward is a *shared scalar* (cooperative): we broadcast it to all
    agents so it matches the per-agent reward list expected by BaseTrainer.

    Win condition uses `info['battle_won']` (SMAC convention).
    Action masking is available via `get_avail_actions()`.
    """

    def __init__(self, env, scenario: str = "unknown"):
        """
        Args:
            env: Gymnasium env returned by `gym.make("smaclite/<scenario>-v0")`
            scenario: Scenario name (for logging / introspection)
        """
        self.env = env
        self._scenario = scenario
        unwrapped = env.unwrapped

        # SMAClite exposes these on the unwrapped env
        n_agents = int(unwrapped.n_agents)
        # n_actions is the same for all agents in SMAClite (homogeneous action space)
        n_actions = int(getattr(unwrapped, "n_actions", 0))
        # episode_limit defines max_steps in the SMAC convention
        max_steps = int(getattr(unwrapped, "episode_limit", 200))

        # Probe observation dimension from a reset (SMAClite stores it but
        # field names have changed across versions, so we infer empirically).
        obs0, _ = env.reset()
        obs_arr = self._to_per_agent_array(obs0, n_agents)
        obs_dim = int(obs_arr.shape[-1])

        super().__init__(n_agents, obs_dim, n_actions, max_steps)

        # Cache last reset's observation so the caller's first reset() call is cheap
        self._cached_first_obs: Optional[List[np.ndarray]] = [
            obs_arr[i].astype(np.float32) for i in range(n_agents)
        ]

    # ---- API helpers --------------------------------------------------------

    @staticmethod
    def _to_per_agent_array(obs, n_agents: int) -> np.ndarray:
        """
        Normalize SMAClite obs into a (n_agents, obs_dim) np.ndarray.

        SMAClite returns a stacked array of shape (n_agents, D), but we stay
        defensive for tuples/lists/dicts in case of future API changes.
        """
        if isinstance(obs, np.ndarray):
            arr = obs
        elif isinstance(obs, (list, tuple)):
            arr = np.asarray(obs)
        elif isinstance(obs, dict):
            # Sort keys to keep agent order stable
            keys = sorted(obs.keys())
            arr = np.asarray([obs[k] for k in keys])
        else:
            raise TypeError(f"Unexpected SMAClite observation type: {type(obs)}")

        if arr.ndim == 1:
            # Flat array → split equally among agents
            if arr.size % n_agents != 0:
                raise ValueError(
                    f"Flat obs of size {arr.size} not divisible by n_agents={n_agents}"
                )
            arr = arr.reshape(n_agents, -1)

        if arr.shape[0] != n_agents:
            raise ValueError(
                f"Obs first dim {arr.shape[0]} != n_agents {n_agents}"
            )
        return arr.astype(np.float32)

    # ---- Core gym-style API -------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> List[np.ndarray]:
        # Reuse the obs captured during __init__ for the first call only
        if self._cached_first_obs is not None and seed is None:
            obs = self._cached_first_obs
            self._cached_first_obs = None
            self.current_step = 0
            return obs

        if seed is not None:
            try:
                obs, _ = self.env.reset(seed=seed)
            except TypeError:
                # Older gym/gymnasium signatures
                obs, _ = self.env.reset()
        else:
            obs, _ = self.env.reset()

        self.current_step = 0
        arr = self._to_per_agent_array(obs, self.n_agents)
        return [arr[i] for i in range(self.n_agents)]

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        if len(actions) != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} actions, got {len(actions)}"
            )

        # SMAClite expects a list/array of ints
        actions_clean = [int(a) for a in actions]

        result = self.env.step(actions_clean)

        # Gymnasium: (obs, reward, terminated, truncated, info)
        # Legacy gym: (obs, reward, done, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated) or bool(truncated)
        else:
            obs, reward, done, info = result
            done = bool(done)

        self.current_step += 1

        arr = self._to_per_agent_array(obs, self.n_agents)
        obs_list = [arr[i] for i in range(self.n_agents)]

        # SMAClite reward is a single scalar shared across agents → broadcast
        r = float(reward)
        rewards = [r] * self.n_agents
        dones = [done] * self.n_agents

        # Win condition (SMAC convention)
        win = False
        if isinstance(info, dict):
            win = bool(info.get("battle_won", False))

        # Pass through useful info without losing what SMAClite returned
        out_info: Dict = {"win": win, "scenario": self._scenario}
        if isinstance(info, dict):
            for k, v in info.items():
                if k != "win":
                    out_info[k] = v

        return obs_list, rewards, dones, out_info

    # ---- SMAC-specific extras (action masking) ------------------------------

    def get_avail_actions(self) -> List[np.ndarray]:
        """
        Per-agent action mask (1 = available, 0 = invalid).

        Useful for QMIX / VDN / MAPPO with action masking.
        """
        unwrapped = self.env.unwrapped
        if not hasattr(unwrapped, "get_avail_actions"):
            # Defensive: all actions available
            return [np.ones(self.act_dim, dtype=np.int8) for _ in range(self.n_agents)]
        avail = unwrapped.get_avail_actions()
        return [np.asarray(a, dtype=np.int8) for a in avail]

    def get_state(self) -> Optional[np.ndarray]:
        """Global state vector (centralized training). None if unavailable."""
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "get_state"):
            return np.asarray(unwrapped.get_state(), dtype=np.float32)
        return None

    # ---- Misc ---------------------------------------------------------------

    def render(self, mode: str = "human"):
        try:
            return self.env.render()
        except Exception as e:
            logger.debug(f"SMAClite render failed: {e}")
            return None

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"SMACliteWrapper(scenario={self._scenario!r}, "
            f"n_agents={self.n_agents}, obs_dim={self.obs_dim}, "
            f"act_dim={self.act_dim}, max_steps={self.max_steps})"
        )


def make_smaclite(
    scenario: str = "2s_vs_1sc",
    *,
    time_limit: Optional[int] = None,
    use_cpp_rvo2: bool = False,
    verbose: bool = True,
    **gym_kwargs: Any,
) -> BaseEnv:
    """
    Create a SMAClite environment.

    Args:
        scenario: Scenario name. See `SMACLITE_SCENARIOS` for the standard list,
                  or any custom name registered in the SMAClite scenarios folder.
        time_limit: Optional override for episode length. None = SMAClite default.
        use_cpp_rvo2: Use the C++ RVO2 collision avoidance backend (faster but
                      requires the optional native extension). Default: False.
        verbose: Print status messages.
        **gym_kwargs: Extra kwargs forwarded to `gym.make`.

    Returns:
        SMACliteWrapper instance.

    Raises:
        ImportError if `smaclite` or `gymnasium` is not installed.
        ValueError if the scenario id is not registered.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(msg)
        logger.debug(msg.strip())

    try:
        import gymnasium as gym
    except ImportError as e:
        raise ImportError(
            "gymnasium is required for SMAClite. Install with: pip install gymnasium"
        ) from e

    try:
        import smaclite  # noqa: F401  (triggers env registration)
    except ImportError as e:
        raise ImportError(
            "smaclite is not installed. Install with:\n"
            "  git clone https://github.com/uoe-agents/smaclite.git\n"
            "  cd smaclite && pip install -e ."
        ) from e

    env_id = f"smaclite/{scenario}-v0"

    # Build gym.make kwargs
    make_kwargs: Dict[str, Any] = {"use_cpp_rvo2": use_cpp_rvo2}
    if time_limit is not None:
        make_kwargs["time_limit"] = int(time_limit)
    make_kwargs.update(gym_kwargs)

    try:
        env = gym.make(env_id, **make_kwargs)
    except gym.error.Error as e:
        available = ", ".join(SMACLITE_SCENARIOS)
        raise ValueError(
            f"Could not create SMAClite scenario '{scenario}' (env_id={env_id}). "
            f"Standard scenarios: {available}.\nOriginal error: {e}"
        ) from e

    _log(f"  ✓ Using SMAClite scenario '{scenario}' (rvo2={'cpp' if use_cpp_rvo2 else 'py'})")
    return SMACliteWrapper(env, scenario=scenario)


def list_smaclite_scenarios() -> List[str]:
    """Return the list of standard SMAClite scenarios."""
    return list(SMACLITE_SCENARIOS)