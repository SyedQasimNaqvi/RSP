# understand code

# Minimal PPO for a MuJoCo "free box" that learns to move on the XY plane to a goal.
# Requirements: mujoco (python bindings), numpy, torch
# Tested conceptually. Keep it simple and readable.

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import mujoco
from mujoco import viewer

# -------------------------
# 1) Simple MuJoCo Env
# -------------------------
class BoxXYEnv:
    """
    Observations:
      [x, y, vx, vy, gx_minus_x, gy_minus_y]
    Actions:
      2D force [fx, fy] applied to the body in world frame (z-force is zero)
    Reward:
      -distance_to_goal  with small control penalty
    Episode ends on success, fall, or time limit.
    """
    def __init__(self, xml_path, max_steps=300, goal_radius=0.03, force_limit=5.0, frame_skip=5, seed=0):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.body(1).name)  # the only moving body
        # Note: body(0) is "world", body(1) is the free box created in the xml
        self.rng = np.random.default_rng(seed)

        self.max_steps = max_steps
        self.goal_radius = goal_radius
        self.force_limit = force_limit
        self.frame_skip = frame_skip

        self.goal = np.zeros(2, dtype=np.float64)
        self.step_count = 0

        # Convenience
        self.dt = self.model.opt.timestep

    def _zero_forces(self):
        # Clear external forces after stepping
        self.data.xfrc_applied[self.body_id, :] = 0.0

    def _set_xy_force(self, fx, fy):
        # Apply force in world coordinates at the body's CoM: [fx, fy, fz, tx, ty, tz]
        # self.data.xfrc_applied[self.body_id, 0] = fx
        # self.data.xfrc_applied[self.body_id, 1] = fy
        # self.data.xfrc_applied[self.body_id, 2] = 0.0

        self.data.qvel[0] = fx
        self.data.qvel[1] = fy
        # self.data.qvel[2] = 0.0

    def _get_xy_pos_vel(self):
        # qpos for a free joint is [qx, qy, qz, qw, x, y, z]? In MuJoCo it is [x, y, z, qw, qx, qy, qz].
        # The exact layout can be read via mj_state. For python bindings:
        # For free joint, qpos[0:3]=pos(x,y,z), qpos[3:7]=quat(w,x,y,z), qvel[0:3]=lin vel, qvel[3:6]=ang vel.
        x, y, z = self.data.qpos[0:3]
        vx, vy = self.data.qvel[0:2]
        return x, y, z, vx, vy

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        # Start the box near origin, slightly above ground to avoid penetration
        # Position
        self.data.qpos[0:3] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # Identity orientation quaternion (w, x, y, z)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # Zero linear and angular velocities
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # Sample a goal on the plane near the origin
        # self.goal = self.rng.uniform(low=-0.3, high=0.3, size=2)
        self.goal = np.array([.25, .25])
        self.step_count = 0
        self._zero_forces()
        return self._get_obs()

    def _get_obs(self):
        x, y, z, vx, vy = self._get_xy_pos_vel()
        gx, gy = self.goal
        obs = np.array([x, y, vx, vy, gx - x, gy - y], dtype=np.float32)
        return obs

    def step(self, action):
        self.step_count += 1
        fx, fy = np.clip(action, -self.force_limit, self.force_limit)

        # Apply forces and step simulator
        self._set_xy_force(float(fx), float(fy))
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._zero_forces()

        # Compute reward and done
        obs = self._get_obs()
        x, y, z, vx, vy = self._get_xy_pos_vel()
        dist = np.linalg.norm(self.goal - np.array([x, y]))
        # reward = -dist - 0.001 * (fx * fx + fy * fy)
        reward = -(np.linalg.norm(obs[:2] - self.goal))
        if np.linalg.norm(obs[:2] - self.goal) < 0.05:
            reward += 10.0

        # Success, fall, or time limit
        success = dist < self.goal_radius
        # fell = z < 0.2  # if it face-plants or falls off something
        fell = False
        timeout = self.step_count >= self.max_steps
        done = success or fell or timeout

        info = {"success": success, "fell": fell, "timeout": timeout, "dist": dist}
        return self._get_obs(), float(reward), bool(done), info


# -------------------------
# 2) Tiny PPO implementation
# -------------------------
def mlp(in_dim, out_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        super().__init__()
        self.pi = mlp(obs_dim, act_dim, hidden)
        self.v = mlp(obs_dim, 1, hidden)
        # log std is a learnable parameter vector
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        # not used jointly
        raise NotImplementedError

    def step(self, obs):
        with torch.no_grad():
            mu = self.pi(obs)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mu, std)
            act = dist.sample()
            logp = dist.log_prob(act).sum(axis=-1)
            v = self.v(obs).squeeze(-1)
        return act, logp, v

    def evaluate_actions(self, obs, act):
        mu = self.pi(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(act).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        v = self.v(obs).squeeze(-1)
        return logp, entropy, v

# Simple on-policy buffer for PPO
class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0.0):
        """
        Compute GAE-Lambda advantages and rewards-to-go.
        last_val is V(s_T) for bootstrapping at terminal or timeout.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE advantage
        deltas = rews[:-1] - vals[:-1] + self.gamma * vals[1:]
        adv = np.zeros_like(deltas)
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.lam * gae
            adv[t] = gae
        self.adv_buf[path_slice] = adv

        # Reward-to-go
        ret = np.zeros_like(rews[:-1])
        running = 0.0
        for t in reversed(range(len(rews) - 1)):
            running = rews[t] + self.gamma * running
            ret[t] = running
        self.ret_buf[path_slice] = ret

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        # Normalize advantages
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf) + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf,
                    act=self.act_buf,
                    ret=self.ret_buf,
                    adv=self.adv_buf,
                    logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

# -------------------------
# 3) Training Loop
# -------------------------
def train_ppo(xml_path,
              steps_per_epoch=4000,
              epochs=50,
              gamma=0.99,
              lam=0.95,
              clip_ratio=0.2,
              pi_lr=3e-4,
              vf_lr=1e-3,
              train_iters=80,
              target_kl=0.01,
              max_ep_len=300,
              seed=1,
              render=False):
    env = BoxXYEnv(xml_path, max_steps=max_ep_len, seed=seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = 6
    act_dim = 2

    ac = ActorCritic(obs_dim, act_dim)
    pi_optimizer = optim.Adam(list(ac.pi.parameters()) + [ac.log_std], lr=pi_lr)
    vf_optimizer = optim.Adam(ac.v.parameters(), lr=vf_lr)
    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # For optional rendering
    viewer_handle = viewer.launch_passive(env.model, env.data) if render else None
    rewards_per_episode = []

    def compute_v(obs_t):
        with torch.no_grad():
            return ac.v(torch.as_tensor(obs_t, dtype=torch.float32).unsqueeze(0)).item()

    for epoch in range(epochs):
        obs = env.reset()
        ep_ret, ep_len = 0.0, 0

        for t in range(steps_per_epoch):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act_t, logp_t, v_t = ac.step(obs_t)
            act = act_t.squeeze(0).numpy()
            next_obs, rew, done, info = env.step(act)

            buf.store(obs, act, rew, v_t.item(), logp_t.item())
            ep_ret += rew
            ep_len += 1
            obs = next_obs

            if render and (epoch % 1 == 0):  # draw occasionally
                viewer_handle.sync()

            timeout = (ep_len == max_ep_len)
            terminal = done
            epoch_ended = (t == steps_per_epoch - 1)

            if terminal or epoch_ended:
                last_val = 0.0 if done else compute_v(obs)
                buf.finish_path(last_val)
                if terminal:
                    # Simple log
                    print(f"Epoch {epoch:3d} | EpRet {ep_ret:7.3f} | EpLen {ep_len:4d} | success={info.get('success')}")
                    rewards_per_episode.append(ep_ret)
                obs, ep_ret, ep_len = env.reset(), 0.0, 0

        # Get trajectory data
        data = buf.get()

        # Policy update
        for i in range(train_iters):
            logp_old = data["logp"]
            adv = data["adv"]
            obs_b = data["obs"]
            act_b = data["act"]
            ret_b = data["ret"]

            # New logp, entropy, and value
            logp, entropy, v = ac.evaluate_actions(obs_b, act_b)
            ratio = torch.exp(logp - logp_old)
            # Clipped surrogate objective
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            pi_optimizer.zero_grad()
            loss_pi.backward()
            pi_optimizer.step()

            # Value function loss
            loss_v = ((v - ret_b) ** 2).mean()
            vf_optimizer.zero_grad()
            loss_v.backward()
            vf_optimizer.step()

            # Early stop if KL too large
            approx_kl = (logp_old - logp).mean().item()
            if approx_kl > 1.5 * target_kl:
                # print(f"Early stop at iter {i} due to KL {approx_kl:.4f}")
                break

        # Reinit buffer for next epoch
        buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    print("Training finished.")

    plt.figure(figsize=(8, 5))
    plt.plot(rewards_per_episode, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode (PPO)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Point this to your xml. If you are running inside the same environment where you uploaded the file:
    # xml_path = "/mnt/data/box (1).xml"
    # Otherwise, replace with the local path to the same xml content.
    xml_path = "box.xml"  # change this if needed
    train_ppo(xml_path, epochs=16, steps_per_epoch=4000)