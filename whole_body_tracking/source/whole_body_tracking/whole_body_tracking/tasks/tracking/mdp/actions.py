# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term replicating the deploy-side three-segment joint target clamp inside training."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.envs import ManagerBasedEnv


class DeployTargetClampAction(JointPositionAction):
    """Joint position action with the deployment-side target clamp applied before PD.

    The deploy core (``G1_deploy/deploy/include/beyondmimic/core.h``, ``Core::targets()``) clamps
    every policy joint target through three segments before it reaches the motors:

    1. joint limits with a small margin (deploy: ±0.01 rad inside the hard limits);
    2. a rate limit (deploy: 8 rad/s) w.r.t. the *previous clamped* target, which accumulates lag;
    3. a torque-radius clamp: the target stays inside ``q + kd*dq/kp ± effort/kp`` so the PD
       torque ``kp*(target - q) - kd*dq`` never saturates, followed by a joint-limit re-clamp.

    The training stack normally sends ``default + scale * action`` straight into the PD with no
    such clamp, so the policy never experiences clipped targets. Training with this term injects
    the same clamp (same order, same semantics) so that the deployment wall is inside the training
    distribution.

    Parity notes:

    * The observation manager stores the *raw* action (``env.action_manager.action``) before the
      term processes it, matching the deploy ``last_action_`` which also keeps the raw action.
    * On reset the deploy core sets ``last_target_ = s.q`` (measured joints at entry). Here the
      rate-limit reference is lazily initialized from the joint state on the first
      ``process_actions`` after a reset, which (after RSI) is exactly the reference-frame pose the
      command manager wrote — the same "enter at the reference first frame" situation as the real
      robot entering BeyondMimic from FixStand.
    """

    cfg: JointPositionActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the base joint position action term
        super().__init__(cfg, env)

        # clamp parameters (defaults mirror the deploy core)
        self._max_target_speed = float(cfg.max_target_speed)
        self._limit_margin = float(cfg.limit_margin)
        # one clamp per policy step, like one Core::targets() call per control tick
        self._step_dt = env.step_dt

        # hard joint limits for the term's joints, shape (num_envs, num_joints)
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._lower = limits[..., 0].clone()
        self._upper = limits[..., 1].clone()

        # read back the per-joint PD gains and effort limits from the actuator groups
        # (each actuator stores (num_envs, num_joints_in_group) tensors)
        kp_all = torch.zeros(self.num_envs, self._asset.num_joints, device=self.device)
        kd_all = torch.zeros_like(kp_all)
        effort_all = torch.zeros_like(kp_all)
        for actuator in self._asset.actuators.values():
            indices = actuator.joint_indices
            if isinstance(indices, slice):
                indices = slice(None)
            kp_all[:, indices] = actuator.stiffness
            kd_all[:, indices] = actuator.damping
            effort_all[:, indices] = actuator.effort_limit_sim
        self._kp = kp_all[:, self._joint_ids]
        self._kd = kd_all[:, self._joint_ids]
        self._effort = effort_all[:, self._joint_ids]

        # previous clamped target (deploy: last_target_) and per-env pending-init flag
        self._last_target = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._needs_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        # resolve to slice for scalar indexing (ActionManager passes slice(None) when None)
        if env_ids is None:
            env_ids = slice(None)
        self._needs_init[env_ids] = True

    def process_actions(self, actions: torch.Tensor):
        # lazily initialize the rate-limit reference from the measured joint state for envs that
        # were reset since the last step; the command manager's RSI write already mirrored the
        # reference-frame pose into the cached buffers, so this reads the post-reset state even
        # though action_manager.reset runs before command_manager.reset inside _reset_idx
        needs_init = torch.where(self._needs_init)[0]
        if needs_init.numel() > 0:
            self._last_target[needs_init] = self._asset.data.joint_pos[needs_init][:, self._joint_ids]
            self._needs_init[needs_init] = False

        # store raw actions and apply the affine transformation (and clip, if configured)
        super().process_actions(actions)

        # three-segment clamp on the position targets, in the deploy core's order
        targets = self._processed_actions
        # 1) joint limits with margin
        lower = self._lower + self._limit_margin
        upper = self._upper - self._limit_margin
        targets = torch.clamp(targets, min=lower, max=upper)
        # 2) rate limit w.r.t. the previous clamped target (accumulates lag, like the deploy)
        delta = self._max_target_speed * self._step_dt
        targets = torch.clamp(targets, min=self._last_target - delta, max=self._last_target + delta)
        # 3) torque radius: |kp*(target - q) - kd*dq| <= effort  <=>  target in q + kd*dq/kp ± effort/kp
        center = self._asset.data.joint_pos[:, self._joint_ids] + self._kd * self._asset.data.joint_vel[
            :, self._joint_ids
        ] / self._kp
        radius = self._effort / self._kp
        targets = torch.clamp(targets, min=center - radius, max=center + radius)
        # 4) joint-limit re-clamp (the deploy core clamps again after the torque radius)
        targets = torch.clamp(targets, min=lower, max=upper)

        self._last_target[:] = targets
        self._processed_actions = targets


@configclass
class DeployTargetClampActionCfg(JointPositionActionCfg):
    """Configuration for the deploy target clamp action term."""

    class_type: type = DeployTargetClampAction

    # deploy core.h defaults: rate limit 8 rad/s, joint-limit margin 0.01 rad
    max_target_speed: float = 8.0
    limit_margin: float = 0.01
