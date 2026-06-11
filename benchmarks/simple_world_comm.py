"""
Simple World Comm Environment (MPE)
====================================
Mixed cooperative-competitive: Good agents cooperate against adversaries.

Standard Benchmark:
- 2 good agents (controlled by RL)
- 4 adversaries (scripted/random - NOT controlled)
- Good agents must reach landmarks while avoiding adversaries
- Good agents can communicate

Reference:
    Lowe et al., 2017 - "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
    https://arxiv.org/abs/1706.02275

Note: Only GOOD agents are controlled. Adversaries use random/scripted policy.
      This is the standard evaluation protocol.

Author: H3C-BEACON Research Team
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_env import BaseEnv, PettingZooWrapper

logger = logging.getLogger(__name__)

# Action constants (PettingZoo MPE convention)
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3
ACTION_UP = 4
N_ACTIONS = 5

# Communication channel size (matches PettingZoo MPE default)
COMM_DIM = 4


class SimpleWorldCommEnv(BaseEnv):
    """
    Built-in Simple World Comm implementation.

    Controls only good agents. Adversaries are scripted (chase nearest good agent).
    Matches the standard evaluation protocol.

    Observation per good agent (size = 4 + 2*n_landmarks + 2*(n_good-1)
                                       + 2*n_adversaries + COMM_DIM):
        [self_vel(2), self_pos(2), rel_landmarks(L*2),
         rel_other_good((G-1)*2), rel_adversaries(A*2), comm(COMM_DIM)]

    Action space (discrete, 5):
        0 = no-op, 1 = left, 2 = right, 3 = down, 4 = up
    """

    metadata = {"name": "simple_world_comm_builtin_v1"}

    def __init__(
        self,
        n_good: int = 2,
        n_adversaries: int = 4,
        n_landmarks: int = 3,
        max_steps: int = 25,
        *,
        agent_size: float = 0.075,
        max_speed_good: float = 1.3,
        max_speed_adv: float = 1.0,
        damping: float = 0.25,
        accel: float = 5.0,
        dt: float = 0.1,
        world_bound: float = 1.5,
        catch_penalty: float = 10.0,
        win_threshold_mult: float = 3.0,
    ) -> None:
        if n_good < 1:
            raise ValueError(f"n_good must be >= 1, got {n_good}")
        if n_adversaries < 0:
            raise ValueError(f"n_adversaries must be >= 0, got {n_adversaries}")
        if n_landmarks < 1:
            raise ValueError(f"n_landmarks must be >= 1, got {n_landmarks}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")

        self.n_good = n_good
        self.n_adversaries = n_adversaries
        self.n_landmarks = n_landmarks

        # Only control good agents
        n_agents = n_good

        # self_vel(2) + self_pos(2) + landmarks(L*2) + other_good((G-1)*2)
        # + adversaries(A*2) + comm(COMM_DIM)
        obs_dim = (
            4
            + n_landmarks * 2
            + (n_good - 1) * 2
            + n_adversaries * 2
            + COMM_DIM
        )

        super().__init__(n_agents, obs_dim, N_ACTIONS, max_steps)

        # Physical parameters
        self.dt = float(dt)
        self.damping = float(damping)
        self.max_speed_good = float(max_speed_good)
        self.max_speed_adv = float(max_speed_adv)
        self.agent_size = float(agent_size)
        self.accel = float(accel)
        self.world_bound = float(world_bound)
        self.catch_penalty = float(catch_penalty)
        self.win_threshold = float(win_threshold_mult) * self.agent_size

        # State
        self.good_pos: Optional[np.ndarray] = None
        self.good_vel: Optional[np.ndarray] = None
        self.adv_pos: Optional[np.ndarray] = None
        self.adv_vel: Optional[np.ndarray] = None
        self.landmark_pos: Optional[np.ndarray] = None
        self.comm: Optional[np.ndarray] = None

        # Internal RNG (decoupled from global numpy state)
        self._rng = np.random.default_rng()

    # ---- Reproducibility ---------------------------------------------------

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed the environment's internal RNG."""
        self._rng = np.random.default_rng(seed)

    # ---- Core gym-style API ------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> List[np.ndarray]:
        if seed is not None:
            self.seed(seed)

        self.current_step = 0

        self.good_pos = self._rng.uniform(-1.0, 1.0, (self.n_good, 2)).astype(np.float32)
        self.good_vel = np.zeros((self.n_good, 2), dtype=np.float32)

        self.adv_pos = self._rng.uniform(-1.0, 1.0, (self.n_adversaries, 2)).astype(np.float32)
        self.adv_vel = np.zeros((self.n_adversaries, 2), dtype=np.float32)

        self.landmark_pos = self._rng.uniform(-1.0, 1.0, (self.n_landmarks, 2)).astype(np.float32)
        self.comm = np.zeros((self.n_good, COMM_DIM), dtype=np.float32)

        return self._get_obs()

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        if self.good_pos is None:
            raise RuntimeError("step() called before reset()")
        if len(actions) != self.n_good:
            raise ValueError(f"Expected {self.n_good} actions, got {len(actions)}")

        self.current_step += 1

        # --- Good agents: apply user actions --------------------------------
        forces = np.zeros((self.n_good, 2), dtype=np.float32)
        for i, action in enumerate(actions):
            forces[i] = self._action_to_force(int(action), agent_idx=i)

        self.good_vel = self.good_vel * (1.0 - self.damping) + forces * self.dt
        self._clip_speed(self.good_vel, self.max_speed_good)

        # --- Adversaries: chase nearest good agent --------------------------
        if self.n_adversaries > 0 and self.n_good > 0:
            # diffs[a, g] = good_g - adv_a  →  (n_adv, n_good, 2)
            diffs = self.good_pos[None, :, :] - self.adv_pos[:, None, :]
            d = np.linalg.norm(diffs, axis=2)               # (n_adv, n_good)
            nearest = np.argmin(d, axis=1)                  # (n_adv,)
            target_vec = diffs[np.arange(self.n_adversaries), nearest]  # (n_adv, 2)
            norms = np.linalg.norm(target_vec, axis=1, keepdims=True)
            safe = norms > 1e-8
            direction = np.where(safe, target_vec / np.maximum(norms, 1e-8), 0.0)
            adv_forces = (direction * self.accel * 0.5).astype(np.float32)
            self.adv_vel = self.adv_vel * (1.0 - self.damping) + adv_forces * self.dt
            self._clip_speed(self.adv_vel, self.max_speed_adv)

        # --- Integrate positions & clip to bounds ---------------------------
        self.good_pos = self.good_pos + self.good_vel * self.dt
        if self.n_adversaries > 0:
            self.adv_pos = self.adv_pos + self.adv_vel * self.dt

        np.clip(self.good_pos, -self.world_bound, self.world_bound, out=self.good_pos)
        if self.n_adversaries > 0:
            np.clip(self.adv_pos, -self.world_bound, self.world_bound, out=self.adv_pos)

        # --- Rewards & info -------------------------------------------------
        rewards, info = self._compute_rewards()

        done = self.current_step >= self.max_steps
        dones = [done] * self.n_agents

        return self._get_obs(), rewards, dones, info

    # ---- Helpers -----------------------------------------------------------

    def _action_to_force(self, action: int, agent_idx: int) -> np.ndarray:
        force = np.zeros(2, dtype=np.float32)
        if action == ACTION_NOOP:
            pass
        elif action == ACTION_LEFT:
            force[0] = -self.accel
        elif action == ACTION_RIGHT:
            force[0] = self.accel
        elif action == ACTION_DOWN:
            force[1] = -self.accel
        elif action == ACTION_UP:
            force[1] = self.accel
        else:
            raise ValueError(
                f"Invalid action {action} for agent {agent_idx} "
                f"(must be in [0, {N_ACTIONS - 1}])"
            )
        return force

    def _clip_speed(self, vel: np.ndarray, max_speed: float) -> None:
        """In-place clip per-agent speed to max_speed."""
        speeds = np.linalg.norm(vel, axis=1)
        over = speeds > max_speed
        if np.any(over):
            scale = (max_speed / speeds[over]).astype(vel.dtype)
            vel[over] = vel[over] * scale[:, None]

    def _get_obs(self) -> List[np.ndarray]:
        """Get observations for good agents only (vectorized)."""
        # Relative landmark positions: (n_good, n_landmarks, 2)
        rel_landmarks = self.landmark_pos[None, :, :] - self.good_pos[:, None, :]

        # Relative good-good: (n_good, n_good, 2)
        rel_good = self.good_pos[None, :, :] - self.good_pos[:, None, :]

        # Relative adversary positions: (n_good, n_adv, 2)
        if self.n_adversaries > 0:
            rel_adv = self.adv_pos[None, :, :] - self.good_pos[:, None, :]
        else:
            rel_adv = np.zeros((self.n_good, 0, 2), dtype=np.float32)

        obs: List[np.ndarray] = []
        idx = np.arange(self.n_good)
        for i in range(self.n_good):
            others_mask = idx != i

            # Communication from another agent (n_good == 1 → zeros)
            if self.n_good > 1:
                other_comm = self.comm[(i + 1) % self.n_good]
            else:
                other_comm = np.zeros(COMM_DIM, dtype=np.float32)

            agent_obs = np.concatenate(
                [
                    self.good_vel[i],                       # (2,)
                    self.good_pos[i],                       # (2,)
                    rel_landmarks[i].reshape(-1),           # (L * 2,)
                    rel_good[i, others_mask].reshape(-1),   # ((G-1) * 2,)
                    rel_adv[i].reshape(-1),                 # (A * 2,)
                    other_comm,                             # (COMM_DIM,)
                ]
            ).astype(np.float32, copy=False)
            obs.append(agent_obs)

        return obs

    def _compute_rewards(self) -> Tuple[List[float], Dict]:
        """Compute shared reward for good agents + rich info dict."""
        # Min distance from each good agent to any landmark
        # diffs: (n_good, n_landmarks, 2)
        diffs_lm = self.good_pos[:, None, :] - self.landmark_pos[None, :, :]
        dists_lm = np.linalg.norm(diffs_lm, axis=2)
        min_dists_lm = dists_lm.min(axis=1)                # (n_good,)
        coverage_reward = -float(min_dists_lm.sum())

        # Collisions with adversaries
        n_caught = 0
        if self.n_adversaries > 0:
            diffs_adv = self.good_pos[:, None, :] - self.adv_pos[None, :, :]
            dists_adv = np.linalg.norm(diffs_adv, axis=2)  # (n_good, n_adv)
            n_caught = int(np.sum(dists_adv < 2.0 * self.agent_size))

        total_reward = coverage_reward - self.catch_penalty * n_caught

        # Shared reward among good agents (per-agent share)
        per_agent = total_reward / self.n_good
        rewards = [per_agent] * self.n_good

        # Win = each good agent close to some landmark AND no captures this step
        reached_any = bool(np.all(min_dists_lm < self.win_threshold))
        win = reached_any and (n_caught == 0)

        info: Dict = {
            "win": win,
            "n_caught": n_caught,
            "coverage_reward": coverage_reward,
            "min_landmark_dists": min_dists_lm.astype(np.float32),
        }

        return rewards, info

    # ---- Misc --------------------------------------------------------------

    def render(self, mode: str = "human"):
        print(f"\nStep {self.current_step}/{self.max_steps}")
        print("Good agents:", self.good_pos)
        print("Adversaries:", self.adv_pos)

    def close(self) -> None:
        """Clean up resources (no-op for built-in)."""
        return None

    def __repr__(self) -> str:
        return (
            f"SimpleWorldCommEnv(n_good={self.n_good}, "
            f"n_adversaries={self.n_adversaries}, "
            f"n_landmarks={self.n_landmarks}, max_steps={self.max_steps})"
        )


def make_simple_world_comm(
    n_good: int = 2,
    n_adversaries: int = 4,
    max_steps: int = 25,
    *,
    num_obstacles: int = 1,
    num_food: int = 2,
    num_forests: int = 2,
    verbose: bool = True,
) -> BaseEnv:
    """
    Create Simple World Comm environment.

    Resolution order:
      1) mpe2 (new official package, PettingZoo >= 1.26)
      2) pettingzoo.mpe (legacy, PettingZoo < 1.26)
      3) Built-in fallback

    Standard setup: 2 good agents vs 4 adversaries.
    Only good agents are controlled; adversaries use scripted policy.

    Args:
        n_good: Number of good agents to control (default: 2)
        n_adversaries: Number of adversaries (default: 4)
        max_steps: Maximum steps per episode (default: 25)
        num_obstacles: Number of obstacles (PettingZoo only, default: 1)
        num_food: Number of food items (PettingZoo only, default: 2)
        num_forests: Number of forests (PettingZoo only, default: 2)
        verbose: Whether to print status messages

    Returns:
        Environment instance (PettingZooWrapper or SimpleWorldCommEnv)
    """

    def _log(msg: str) -> None:
        if verbose:
            print(msg)
        logger.debug(msg.strip())

    pz_kwargs = dict(
        num_good=n_good,
        num_adversaries=n_adversaries,
        num_obstacles=num_obstacles,
        num_food=num_food,
        num_forests=num_forests,
        max_cycles=max_steps,
        continuous_actions=False,
    )

    # 1) New official MPE2 package
    try:
        from mpe2 import simple_world_comm_v3 as _swc  # type: ignore
        env = _swc.parallel_env(**pz_kwargs)
        _log("  ✓ Using mpe2 simple_world_comm_v3")
        # Good agents are named 'agent_*', adversaries are 'adversary_*'/'leadadversary_*'
        return PettingZooWrapper(env, agent_filter="agent")
    except ImportError:
        pass
    except Exception as e:  # pragma: no cover
        _log(f"  ⚠️ mpe2 error: {e}")

    # 2) Legacy PettingZoo MPE (< 1.26)
    try:
        from pettingzoo.mpe import simple_world_comm_v3 as _swc  # type: ignore
        env = _swc.parallel_env(**pz_kwargs)
        _log("  ✓ Using PettingZoo simple_world_comm_v3 (legacy)")
        return PettingZooWrapper(env, agent_filter="agent")
    except ImportError:
        _log("  ⚠️ Neither mpe2 nor pettingzoo.mpe available, using built-in")
    except Exception as e:  # pragma: no cover
        _log(f"  ⚠️ PettingZoo error: {e}, using built-in")

    # 3) Built-in fallback
    return SimpleWorldCommEnv(
        n_good=n_good,
        n_adversaries=n_adversaries,
        max_steps=max_steps,
    )