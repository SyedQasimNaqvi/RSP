from torch.distributions import Normal
import torch.nn.functional as f
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os


xml_path = 'box.xml' #xml file (assumes this is in the same folder as this file)
simend = 100 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

class BoxEnv(gym.Env):
    def __init__(self, model, data, target=np.array([.25, .25])):
        super(BoxEnv, self).__init__()
        self.model = model
        self.data = data
        self.target = target
        self.max_steps = 200
        self.current_step = 0

        self.observation_space = gym.spaces.Box(low=-np.inf, 
                                                    high=np.inf,
                                                    shape=(4,),
                                                    dtype=np.float32)

        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), 
                                               high=np.array([1.0, 1.0], dtype=np.float32), 
                                               dtype=np.float32)
        
    def reset(self, *, seed=None, options=None):
        mj.mj_resetData(self.model, self.data)
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}
    
    # def step(self, action, n_substeps=10, debug=True):
    #     """
    #     Step the environment.
    #     action: np.array shape (2,) clipped to action_space
    #     n_substeps: how many physics steps to take per env step
    #     debug: if True, print helpful debug info
    #     """
    #     # 1) bookkeeping
    #     self.current_step += 1

    #     # 2) clip action
    #     action = np.asarray(action, dtype=np.float64)
    #     action = np.clip(action, self.action_space.low, self.action_space.high)

    #     # 3) Decide how to map action -> qvel.
    #     #    We try to set the first two velocity DOFs, but check sizes first.
    #     n_qvel = self.data.qvel.shape[0]

    #     if n_qvel >= 2:
    #         # Overwrite the first two velocity DOFs with the action.
    #         # NOTE: depending on your XML, these DOFs might not be x/y linear velocities
    #         # for a freebody. Inspect model to be sure (see debug below).
    #         self.data.qvel[:2] = action
    #     else:
    #         # Fallback: if no qvel DOFs, write into ctrl if available, else raise.
    #         if self.data.ctrl.shape[0] >= action.size:
    #             self.data.ctrl[:action.size] = action
    #         else:
    #             raise RuntimeError(
    #                 f"Model has qvel.shape={self.data.qvel.shape} and ctrl.shape={self.data.ctrl.shape}; "
    #                 "cannot apply action. Add actuators or adjust action mapping."
    #             )

    #     # 4) Make derived quantities consistent before stepping
    #     #    (important after directly setting qpos/qvel or ctrl)
    #     mj.mj_forward(self.model, self.data)

    #     # 5) Advance physics n_substeps times
    #     for _ in range(int(n_substeps)):
    #         mj.mj_step(self.model, self.data)

    #     # 6) Get observation and compute reward (keeps your reward)
    #     obs = self._get_obs()
    #     pos = obs[:2]    # your _get_obs returns [qpos0, qpos1, qvel0, qvel1]
    #     reward = -(np.linalg.norm(pos - self.target))
    #     if np.linalg.norm(pos - self.target) < 0.05:
    #         reward += 10.0

    #     # 7) Termination flags
    #     terminated = (self.current_step >= self.max_steps) or (np.linalg.norm(pos - self.target) < 0.05)
    #     truncated = False

    #     if debug:
    #         print("DEBUG step:", {
    #             "action": action,
    #             "qpos[:4]": self.data.qpos[:4].copy(),
    #             "qvel[:6]": self.data.qvel[:6].copy() if self.data.qvel.shape[0] >= 6 else self.data.qvel.copy(),
    #             "reward": reward,
    #             "terminated": terminated,
    #             "ctrl_shape": self.data.ctrl.shape
    #         })

    #     return obs, reward, terminated, truncated, {}

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        print(action)

        self.data.qvel[:action.size] = action

        for __ in range(10):
            mj.mj_step(self.model, self.data)        # <-- self.model/self.data

        obs = self._get_obs()
        reward = -(np.linalg.norm(obs[:2] - self.target))
        # print(1/(np.linalg.norm(obs[:2] - self.target)+(10**-6)))
        if np.linalg.norm(obs[:2] - self.target) < 0.05:
            reward += 10.0

        # obs = self._get_obs()
        # reward = 1/(np.linalg.norm(obs[:2] - self.target) + (10**-6))
        # print(1/(np.linalg.norm(obs[:2] - self.target)+(10**-6)))
        # if np.linalg.norm(obs[:2] - self.target) < 0.05:
        #     reward += 10.0
        done = (
            self.current_step >= self.max_steps
            or np.linalg.norm(obs[:2] - self.target) < 0.05
        )
        # print(reward)
        return obs, reward, done, False, {}
    
    def _get_obs(self):
        return np.array([
            self.data.qpos[0], self.data.qpos[1],
            self.data.qvel[0], self.data.qvel[1]
        ], dtype=np.float32)

# model = mj.MjModel.from_xml_path(xml_path)
data_train = mj.MjData(model)
data_test = mj.MjData(model)
env_train = BoxEnv(model, data_train)
env_test  = BoxEnv(model, data_test)

class BackboneNetwork(nn.Module):
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = nn.Linear(hidden_dimensions, out_features)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.layer1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x
    
class ActorCritic(nn.Module):
    def __init__(self, actor, critic, action_dim):
        super().__init__()
        self.actor = actor
        self.critic = critic
        # learnable log std for continuous Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(action_dim) * 0.5)

    def forward(self, state):
        action_mean = self.actor(state)
        # expand log_std to match action_mean shape at runtime
        log_std = self.log_std.expand_as(action_mean)
        value_pred = self.critic(state)
        return action_mean, log_std, value_pred

def create_agent(hidden_dimensions, dropout):
    INPUT_FEATURES = env_train.observation_space.shape[0]
    HIDDEN_DIMENSIONS = hidden_dimensions
    ACTION_DIM = env_train.action_space.shape[0]   # continuous action dim
    CRITIC_OUTPUT_FEATURES = 1
    DROPOUT = dropout
    actor = BackboneNetwork(INPUT_FEATURES, HIDDEN_DIMENSIONS, ACTION_DIM, DROPOUT)
    critic = BackboneNetwork(INPUT_FEATURES, HIDDEN_DIMENSIONS, CRITIC_OUTPUT_FEATURES, DROPOUT)
    agent = ActorCritic(actor, critic, ACTION_DIM)
    return agent

def calculate_returns(rewards, discount_factor):
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)
    returns = torch.tensor(returns, dtype=torch.float32)
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def calculate_advantages(returns, values):
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def calculate_surrogate_loss(
        actions_log_probability_old,
        actions_log_probability_new,
        epsilon,
        advantages):
    advantages = advantages.detach()
    policy_ratio = (
        actions_log_probability_new - actions_log_probability_old
    ).exp()
    surrogate_loss_1 = policy_ratio * advantages
    surrogate_loss_2 = torch.clamp(
        policy_ratio, min=1.0-epsilon, max=1.0+epsilon
    )*advantages
    surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
    return surrogate_loss

def calculate_losses(
        surrogate_loss, entropy, entropy_coefficient, returns, value_pred
):
    entropy_bonus = entropy_coefficient * entropy
    policy_loss = -(surrogate_loss + entropy_bonus).mean()
    value_loss = f.mse_loss(value_pred, returns).mean()
    return policy_loss, value_loss

def init_training():
    states = []
    actions = []
    actions_log_probability = []
    values = []
    rewards = []
    done = False
    epsilon_reward = 0
    return states, actions, actions_log_probability, values, rewards, done, epsilon_reward

def forward_pass(env, agent, optimizer, discount_factor):
    states, actions, actions_log_probability, values, rewards, done, episode_reward = init_training()
    obs, _ = env.reset()
    agent.train()
    while not done:
        state = torch.from_numpy(obs).float().unsqueeze(0)
        states.append(state)
        action_mean, log_std, value_pred = agent(state)
        std = log_std.exp()
        dist = Normal(action_mean, std)
        action = dist.sample()
        log_prob_action = dist.log_prob(action).sum(-1)

        action_np = action.detach().cpu().numpy().squeeze(0)
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        actions.append(torch.from_numpy(action_np).float())
        actions_log_probability.append(log_prob_action.squeeze(0))
        values.append(value_pred.squeeze(0))
        rewards.append(reward)
        episode_reward += reward
        obs = next_obs

    states = torch.cat(states, dim=0)
    actions = torch.stack(actions, dim=0)
    actions_log_probability = torch.stack(actions_log_probability).detach()
    values = torch.stack(values).squeeze(-1).detach()
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
    return episode_reward, states, actions, actions_log_probability, advantages, returns

def update_policy(
        agent,
        states,
        actions,
        actions_log_probability_old,
        advantages,
        returns,
        optimizer,
        ppo_steps,
        epsilon,
        entropy_coefficient
):
    BATCH_SIZE = 128
    total_policy_loss = 0
    total_value_loss = 0
    actions_log_probability_old = actions_log_probability_old.detach()
    actions = actions.detach()
    training_results_dataset = TensorDataset(
        states,
        actions,
        actions_log_probability_old,
        advantages,
        returns
    )
    batch_dataset = DataLoader(
        training_results_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    for _ in range(ppo_steps):
        for batch_idx, (b_states, b_actions, b_actions_logprob_old, b_advantages, b_returns) in enumerate(batch_dataset):
            action_mean, log_std, value_pred = agent(b_states)
            value_pred = value_pred.squeeze(-1)
            std = log_std.exp()
            dist = Normal(action_mean, std)
            # new log prob for the taken actions (sum across action dim)
            actions_log_probability_new = dist.log_prob(b_actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            surrogate = calculate_surrogate_loss(
                b_actions_logprob_old,
                actions_log_probability_new,
                epsilon,
                b_advantages
            ).mean()

            policy_loss, value_loss = calculate_losses(
                surrogate,
                entropy,
                entropy_coefficient,
                b_returns,
                value_pred
            )

            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    # normalize by number of batches * ppo_steps
    n_batches = max(1, len(batch_dataset))
    return total_policy_loss / (ppo_steps * n_batches), total_value_loss / (ppo_steps * n_batches)

def evaluate(env, agent):
    agent.eval()
    done = False
    episode_reward = 0.0
    obs, _ = env.reset()
    while not done:
        state = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            action_mean, log_std, _ = agent(state)
            std = log_std.exp()
            # sample stochastically (or use mean for deterministic eval)
            dist = Normal(action_mean, std)
            action = dist.sample()
        action_np = action.cpu().numpy().squeeze(0)
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        episode_reward += reward
        obs = next_obs
    return episode_reward

# def evaluate(env, agent):
#     agent.eval()
#     done = False
#     episode_reward = 0.0
#     obs, _ = env.reset()
#     while not done:
#         state = torch.from_numpy(obs).float().unsqueeze(0)
        
#         with torch.no_grad():
#             action_mean, log_std, _ = agent(state)
#             dist = Normal(action_mean, log_std.exp())
#             action_np = dist.sample().cpu().numpy().squeeze(0)
 
#         # with torch.no_grad():
#         #     action_mean, log_std, _ = agent(state)

#         action_np = action_mean.cpu().numpy().squeeze(0)
#         action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
#         next_obs, reward, terminated, truncated, info = env.step(action_np)
#         done = terminated or truncated
#         episode_reward += reward
#         obs = next_obs
#     return episode_reward


def run_ppo():
    MAX_EPISODES = 500
    DISCOUNT_FACTOR = 0.99
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 1
    PPO_STEPS = 8
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFFICIENT = 0.01
    HIDDEN_DIMENSIONS = 64
    DROPOUT = 0.2
    LEARNING_RATE = 1e-3
    RENDER_INTERVAL = 1   # render every 10 episodes

    train_rewards, test_rewards = [], []
    policy_losses, value_losses = [], []

    agent = create_agent(HIDDEN_DIMENSIONS, DROPOUT)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    for episode in range(1, MAX_EPISODES + 1):
        # --- collect rollout ---
        train_reward, states, actions, actions_log_probability, advantages, returns = forward_pass(
            env_train, agent, optimizer, DISCOUNT_FACTOR
        )

        # --- PPO update ---
        policy_loss, value_loss = update_policy(
            agent,
            states,
            actions,
            actions_log_probability,
            advantages,
            returns,
            optimizer,
            PPO_STEPS,
            EPSILON,
            ENTROPY_COEFFICIENT
        )

        # --- evaluate current policy ---
        test_reward = evaluate(env_test, agent)

        # stats
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        mean_abs_policy_loss = np.mean(np.abs(policy_losses[-N_TRIALS:]))
        mean_abs_value_loss = np.mean(np.abs(value_losses[-N_TRIALS:]))

        if episode % PRINT_INTERVAL == 0:
            print(f'Episode: {episode:3} | '
                  f'Mean Train Rewards: {mean_train_rewards:3.1f} '
                  f'| Mean Test Rewards: {mean_test_rewards:3.1f} '
                #   f'| Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} '
                #   f'| Mean Abs Value Loss: {mean_abs_value_loss:2.2f}'
                )

        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break

        # --- occasionally render the environment ---
        if episode % RENDER_INTERVAL == 0:
            obs, _ = env_test.reset()
            done = False
            while not done and not glfw.window_should_close(window):
                mj.mj_step(model, data)
                viewport_width, viewport_height = glfw.get_framebuffer_size(window)
                viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
                mj.mjv_updateScene(model, data, opt, None, cam,
                                   mj.mjtCatBit.mjCAT_ALL.value, scene)
                mj.mjr_render(viewport, scene, context)
                glfw.swap_buffers(window)
                glfw.poll_events()
                _, _, terminated, truncated, _ = env_test.step(np.zeros(env_test.action_space.shape))
                done = terminated or truncated

    return agent, train_rewards, test_rewards

run_ppo()

glfw.terminate()