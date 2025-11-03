import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt


# -------------------------
# 1) MuJoCo Environment
# -------------------------
class RobotEnv:
    def __init__(self, xml_path, goal=np.array([1.0, 0.0]), max_steps=1000, frame_skip=5, seed=0):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.goal = np.array(goal, dtype=np.float64)
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.step_count = 0

        # Use correct actuator names
        self.act_left = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_vel")
        self.act_right = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_vel")

        # Correct body reference
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")

        self.obs_dim = 6
        self.act_dim = 2

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs()

    def _get_chassis_xy_vel(self):
        x, y, z = self.data.xpos[self.body_id]
        vx, vy, vz = self.data.cvel[self.body_id][:3]
        return x, y, vx, vy

    def _get_obs(self):
        x, y, vx, vy = self._get_chassis_xy_vel()
        gx, gy = self.goal
        return np.array([x, y, vx, vy, gx - x, gy - y], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        left_vel, right_vel = np.clip(action, -30, 30)

        # Apply control
        self.data.ctrl[self.act_left] = left_vel
        self.data.ctrl[self.act_right] = right_vel

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        x, y, vx, vy = self._get_chassis_xy_vel()

        dist = np.linalg.norm(np.array([x, y]) - self.goal)
        reward = -dist

        done = (dist < 0.05) or (self.step_count >= self.max_steps)
        info = {"dist": dist, "success": dist < 0.05}
        return obs, reward, done, info


# -------------------------
# 2) PPO Core
# -------------------------
def mlp(in_dim, out_dim, hidden=(64, 64), act=nn.Tanh):
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        super().__init__()
        self.pi = mlp(obs_dim, act_dim, hidden)
        self.v = mlp(obs_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def step(self, obs):
        with torch.no_grad():
            mu = self.pi(obs)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mu, std)
            act = dist.sample()
            logp = dist.log_prob(act).sum(-1)
            v = self.v(obs).squeeze(-1)
        return act, logp, v

    def evaluate_actions(self, obs, act):
        mu = self.pi(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(act).sum(-1)
        entropy = dist.entropy().sum(-1)
        v = self.v(obs).squeeze(-1)
        return logp, entropy, v


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = np.zeros_like(deltas)
        gae = 0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.lam * gae
            adv[t] = gae
        self.adv_buf[path_slice] = adv

        # discounted returns
        ret = np.zeros_like(rews[:-1])
        running = 0
        for t in reversed(range(len(rews) - 1)):
            running = rews[t] + self.gamma * running
            ret[t] = running
        self.ret_buf[path_slice] = ret
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf) + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


# -------------------------
# 3) Train PPO
# -------------------------
def train_ppo(xml_path, steps_per_epoch=4000, epochs=30, gamma=0.99, lam=0.95,
              clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, train_iters=80,
              target_kl=0.01, max_ep_len=10000, seed=1, render=True):

    env = RobotEnv(xml_path, max_steps=max_ep_len, seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim, act_dim = env.obs_dim, env.act_dim
    ac = ActorCritic(obs_dim, act_dim)
    pi_opt = optim.Adam(list(ac.pi.parameters()) + [ac.log_std], lr=pi_lr)
    vf_opt = optim.Adam(ac.v.parameters(), lr=vf_lr)
    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    viewer_handle = viewer.launch_passive(env.model, env.data) if render else None
    rewards_per_episode = []

    for epoch in range(epochs):
        obs = env.reset()
        ep_ret, ep_len = 0, 0

        for t in range(steps_per_epoch):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            act_t, logp_t, v_t = ac.step(obs_t)
            act = act_t.squeeze(0).numpy()
            next_obs, rew, done, info = env.step(act)

            buf.store(obs, act, rew, v_t.item(), logp_t.item())
            ep_ret += rew
            ep_len += 1
            obs = next_obs

            if render:
                viewer_handle.sync()

            timeout = ep_len == max_ep_len
            if done or timeout or (t == steps_per_epoch - 1):
                last_val = 0 if done else ac.v(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).item()
                buf.finish_path(last_val)
                if done:
                    print(f"Epoch {epoch:3d} | EpRet {ep_ret:7.3f} | Len {ep_len:4d} | success={info['success']}")
                    rewards_per_episode.append(ep_ret)
                obs, ep_ret, ep_len = env.reset(), 0, 0

        data = buf.get()

        # Policy update
        for i in range(train_iters):
            logp_old = data["logp"]
            adv, obs_b, act_b, ret_b = data["adv"], data["obs"], data["act"], data["ret"]

            logp, entropy, v = ac.evaluate_actions(obs_b, act_b)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            pi_opt.zero_grad()
            loss_pi.backward()
            pi_opt.step()

            loss_v = ((v - ret_b) ** 2).mean()
            vf_opt.zero_grad()
            loss_v.backward()
            vf_opt.step()

            approx_kl = (logp_old - logp).mean().item()
            if approx_kl > 1.5 * target_kl:
                break

        buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    print("Training complete.")
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward")
    plt.show()


if __name__ == "__main__":
    xml_path = "ddrive.xml"
    train_ppo(xml_path, epochs=30, steps_per_epoch=100000)