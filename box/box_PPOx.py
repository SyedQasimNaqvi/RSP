import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco as mj
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class BoxEnv(gym.Env):
    def __init__(self, model, data, target=np.array([.25, .25])):
        super(BoxEnv, self).__init__()
        self.model = model
        self.data = data
        self.target = target
        self.max_steps = 200
        self.current_step = 0

        # Continuous actions: [x_vel, y_vel]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observations: [x, y, x_vel, y_vel]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        mj.mj_resetData(self.model, self.data)
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.current_step += 1
        # apply continuous velocity
        self.data.qvel[0] = np.clip(action[0], -1, 1)
        self.data.qvel[1] = np.clip(action[1], -1, 1)

        for _ in range(10):
            mj.mj_step(self.model, self.data)

        obs = self._get_obs()
        dist = np.linalg.norm(obs[:2] - self.target)
        reward = -dist
        if dist < 0.05:
            reward += 10.0

        done = (self.current_step >= self.max_steps) or (dist < 0.05)
        return obs, reward, done, False, {}

    def _get_obs(self):
        return np.array([
            self.data.qpos[0], self.data.qpos[1],
            self.data.qvel[0], self.data.qvel[1]
        ], dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mu = self.net(obs)
        std = torch.exp(self.log_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        return self.net(obs)


class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip=0.2, lam=0.95):
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.gamma = gamma
        self.clip = clip
        self.lam = lam

    def get_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
        mu, std = self.actor(obs)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action.cpu().numpy(), log_prob.cpu().item()

    def compute_returns(self, rewards, dones, values, next_value):
        returns, advs = [], []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advs.insert(0, gae)
            next_value = values[step]
        returns = [a + v for a, v in zip(advs, values)]
        return torch.tensor(advs, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def update(self, obs, actions, log_probs_old, returns, advs):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
        actions = torch.as_tensor(actions, dtype=torch.float32).to(device)
        log_probs_old = torch.as_tensor(log_probs_old, dtype=torch.float32).to(device)
        returns = returns.to(device)
        advs = advs.to(device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for _ in range(10):  # epochs
            mu, std = self.actor(obs)
            dist = Normal(mu, std)
            log_probs = dist.log_prob(actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()

            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advs
            actor_loss = -torch.min(surr1, surr2).mean()

            value_pred = self.critic(obs).squeeze()
            critic_loss = nn.MSELoss()(value_pred, returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def train(env, agent, steps_per_epoch=2048, epochs=1000):
    for epoch in range(epochs):
        obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf = [], [], [], [], [], []
        obs, _ = env.reset()
        done = False

        for step in range(steps_per_epoch):
            action, logp = agent.get_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            value = agent.critic(torch.as_tensor(obs, dtype=torch.float32).to(device)).item()

            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            done_buf.append(done)
            logp_buf.append(logp)
            val_buf.append(value)

            obs = next_obs
            if done:
                obs, _ = env.reset()

        next_value = agent.critic(torch.as_tensor(obs, dtype=torch.float32).to(device)).item()
        advs, rets = agent.compute_returns(rew_buf, done_buf, val_buf, next_value)
        agent.update(np.array(obs_buf), np.array(act_buf), np.array(logp_buf), rets, advs)

        print(f"Epoch {epoch+1}: Avg Reward = {np.mean(rew_buf):.3f}")

model = mj.MjModel.from_xml_path('box.xml')
data = mj.MjData(model)
env = BoxEnv(model, data)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
agent = PPOAgent(obs_dim, act_dim)

train(env, agent, epochs=500)