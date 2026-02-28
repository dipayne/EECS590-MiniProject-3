"""
Davis_Payne_Mini_Project3_Problem2.py

Problem 2: Superhuman Atari via PPO (Actor-Critic) and Saliency Analysis
=========================================================================

HOW TO RUN
----------
1.  Install dependencies (once):
        pip install torch torchvision gymnasium[atari] ale-py opencv-python matplotlib numpy

2.  Accept ROM licence (once):
        pip install "gymnasium[accept-rom-license]"

3.  Full training (~30-60 min on CPU, 100 k steps):
        python Davis_Payne_Mini_Project3_Problem2.py

4.  Quick smoke-test (~2 min):
        python Davis_Payne_Mini_Project3_Problem2.py --smoke-test

ALGORITHM: Proximal Policy Optimisation (PPO, Schulman et al. 2017)
--------------------------------------------------------------------
PPO is an on-policy, model-free Actor-Critic algorithm.  A shared CNN
backbone produces both a *policy* π(a|s) (actor) and a *state-value
function* V(s) (critic).

Network
    Input:  (batch, 4, 84, 84) stacked grayscale frames
    Shared: 3 conv layers (Mnih et al. 2015) + Linear(3136, 512) + ReLU
    Actor:  Linear(512, n_actions)  ->  logits -> π(a|s)
    Critic: Linear(512, 1)          ->  V(s)

Loss function (Q1a)
    L(θ) = L_CLIP(θ) + c_v · L_V(θ) - c_e · H[π(·|s)]

    L_CLIP = E_t[ min( r_t(θ) · Â_t ,
                       clip(r_t(θ), 1-ε, 1+ε) · Â_t ) ]
    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)      (probability ratio)
    Â_t    = GAE advantage  (Generalised Advantage Estimation)
    L_V    = E_t[ (V_θ(s_t) - R_t)² ]               (MSE value target)
    H[π]   = -E_t[ Σ_a π(a|s_t) log π(a|s_t) ]      (entropy bonus)

Target network (Q1b)
    PPO does not maintain a separate frozen target network.  Instead,
    the clipping ratio ε prevents any single update from deviating far
    from the old policy, providing a similar stabilising effect.

Replay buffer (Q1c)
    PPO is on-policy and uses no replay buffer.  Fresh transitions are
    collected from the current policy, used for n_epochs gradient steps,
    then *discarded*.  This avoids the off-policy distribution shift that
    a replay buffer would introduce in a policy-gradient method.

Bellman / on-policy / model-free (Q1d)
    The critic is trained via the Bellman consistency
        V(s) = E[r + γ V(s')]
    using n-step bootstrapped returns with GAE.
    PPO is ON-POLICY (data must come from the current policy) and
    MODEL-FREE (no environment dynamics model is learned or used).

Saliency scalar used in place of Q̂(f*, a*)
    For an Actor-Critic / PPO agent we use  log π(a|f)  as the
    action-specific scalar; it plays the same role as Q̂(f*, a*) in the
    original DQN saliency formulation:
        Sal(i,j) = | log π(a | f*) − log π(a | f̃*_ij) |

Figures produced
----------------
fig08_ppo_learning_curve.png        Q2a  episode return vs training step + losses
fig09_value_over_episode.png        Q2b  V(s) per eval step + pivotal annotations
fig10_pivotal_frames.png            Q3   5 pivotal game screenshots
fig11_saliency_greedy.png           Q4   perturbation saliency, greedy a*
fig12_saliency_nongreedy.png        Q5   greedy a* vs non-greedy a' saliency
fig13_patch_size_comparison.png     Q6a  P=4, P=8, P=14 on one frame
fig14_saliency_entropy_training.png Q6b  saliency entropy vs training step
fig15_gradient_vs_perturbation.png  Q7a  gradient saliency vs perturbation saliency
fig16_adversarial_frames.png        Q7c  adversarial perturbation δ and flipped action
"""

import os
import argparse
import random
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def fig_path(name):
    return os.path.join(SCRIPT_DIR, name)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true")
parser.add_argument("--game", default="BreakoutNoFrameskip-v4")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=0,
                    help="Override total training steps (0 = use built-in default)")
parser.add_argument("--n-envs", type=int, default=1,
                    help="Number of parallel environments (1=CPU default, 8=GPU/Colab)")
args   = parser.parse_args()
SMOKE  = args.smoke_test
GAME   = args.game
SEED   = args.seed
STEPS_OVERRIDE  = args.steps
N_ENVS          = args.n_envs

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
print(f"Game   : {GAME}")
print(f"Smoke  : {SMOKE}")


# =============================================================================
# 1.  ALE Preprocessing Wrappers  (Mnih et al. 2015 standard pipeline)
# =============================================================================

class NoopResetEnv(gym.Wrapper):
    """Random number of no-op actions at episode start for state diversity."""
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(np.random.randint(1, self.noop_max + 1)):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat action for `skip` frames; return pixel-max of last two."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        self._obs_buffer = deque(maxlen=2)

    def step(self, action):
        total_reward, terminated, truncated = 0.0, False, False
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        return np.max(np.stack(self._obs_buffer), axis=0), total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """Treat loss of a life as end-of-episode for better credit assignment."""
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset for games that require it (e.g. Breakout)."""
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    """Grayscale + resize to 84×84."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)[:, :, np.newaxis]


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1} during training."""
    def reward(self, r):
        return np.sign(r)


class FrameStack(gym.Wrapper):
    """Stack k most recent frames along the channel axis."""
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self._frames = deque(maxlen=k)
        h, w, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, k), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self._frames.append(obs)
        return np.concatenate(list(self._frames), axis=2), info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return np.concatenate(list(self._frames), axis=2), r, terminated, truncated, info


class TransposeImage(gym.ObservationWrapper):
    """HWC -> CHW for PyTorch."""
    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c, h, w), dtype=np.uint8)

    def observation(self, obs):
        return obs.transpose(2, 0, 1)


def make_atari_env(game, seed=0):
    env = gym.make(game)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, k=4)
    env = TransposeImage(env)
    env.reset(seed=seed)
    return env


def make_vec_env(game, n_envs, seed=0):
    """
    Create n_envs parallel Atari environments using SyncVectorEnv.

    Each environment runs the full standard preprocessing pipeline
    (NoopReset, MaxAndSkip, EpisodicLife, WarpFrame, ClipReward,
    FrameStack, TransposeImage) in its own independent instance.
    SyncVectorEnv stacks their (4,84,84) observations into (n_envs,4,84,84).

    With n_envs=8 on a GPU, effective throughput is ~8× a single env,
    enabling 10 M training steps in ~25 min on a T4 (vs ~28 hrs on CPU).
    """
    def _make(rank):
        def _init():
            return make_atari_env(game, seed=seed + rank)
        return _init
    return gym.vector.SyncVectorEnv([_make(i) for i in range(n_envs)])


# =============================================================================
# 2.  Actor-Critic Network  (shared CNN, Mnih et al. 2015 architecture)
# =============================================================================

class ActorCriticNet(nn.Module):
    """
    Shared convolutional backbone with two independent heads:
      • actor_head  -> logits -> π(a|s)   (policy / actor)
      • critic_head -> scalar V(s)         (state-value / critic)

    Input : (batch, 4, 84, 84) uint8  →  float/255 inside forward
    """
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc_shared   = nn.Linear(64 * 7 * 7, 512)
        self.actor_head  = nn.Linear(512, n_actions)
        self.critic_head = nn.Linear(512, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def _features(self, x):
        x = x.float() / 255.0
        x = self.conv(x)
        return F.relu(self.fc_shared(x.reshape(x.size(0), -1)))

    def forward(self, x):
        """Returns (logits, value)."""
        h = self._features(x)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

    def policy_probs(self, x):
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)


# =============================================================================
# 3.  Rollout Buffer  (stores one complete rollout for PPO updates)
# =============================================================================

class RolloutBuffer:
    """
    Rollout buffer supporting N parallel environments.

    Storage shape:  (n_steps, n_envs, ...)
    After GAE:      flattened to (n_steps * n_envs, ...) for mini-batching.

    With n_envs=1  this is identical to the original single-env buffer.
    With n_envs=8  each rollout holds 128 * 8 = 1024 transitions, giving
                   larger and more diverse mini-batches per gradient step.
    """
    def __init__(self, n_steps, n_envs=1, obs_shape=(4, 84, 84)):
        self.n_steps  = n_steps
        self.n_envs   = n_envs
        self.total    = n_steps * n_envs

        self.obs        = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.uint8)
        self.actions    = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.rewards    = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values     = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs  = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones      = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns    = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.ptr        = 0

    def add(self, obs, actions, rewards, values, log_probs, dones):
        """
        Store one timestep across all n_envs.
        All inputs must have leading dimension n_envs, which act_vec guarantees.
        """
        t = self.ptr
        self.obs[t]       = obs
        self.actions[t]   = actions
        self.rewards[t]   = rewards
        self.values[t]    = values
        self.log_probs[t] = log_probs
        self.dones[t]     = dones
        self.ptr += 1

    def compute_gae(self, last_values, last_dones, gamma, lam):
        """
        Generalised Advantage Estimation — vectorised over n_envs.

            δ_t  = r_t + γ · V(s_{t+1}) · (1−done_t) − V(s_t)
            Â_t  = δ_t + (γλ)(1−done_t) · Â_{t+1}

        last_values : (n_envs,) bootstrap values at end of rollout
        last_dones  : (n_envs,) bool — True if last step was terminal
        """
        gae       = np.zeros(self.n_envs, dtype=np.float32)
        next_val  = np.array(last_values,              dtype=np.float32)
        next_done = np.array(last_dones,  dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            delta      = (self.rewards[t]
                          + gamma * next_val * (1.0 - next_done)
                          - self.values[t])
            gae        = delta + gamma * lam * (1.0 - next_done) * gae
            self.advantages[t] = gae
            next_val   = self.values[t]
            next_done  = self.dones[t]
        self.returns = self.advantages + self.values

    def mini_batches(self, batch_size):
        """Flatten (n_steps, n_envs, …) → (total, …) then yield mini-batches."""
        obs_f  = self.obs.reshape(self.total, *self.obs.shape[2:])
        act_f  = self.actions.reshape(self.total)
        logp_f = self.log_probs.reshape(self.total)
        adv_f  = self.advantages.reshape(self.total)
        ret_f  = self.returns.reshape(self.total)

        idx = np.random.permutation(self.total)
        for start in range(0, self.total, batch_size):
            b = idx[start:start + batch_size]
            yield (
                torch.tensor(obs_f[b],  dtype=torch.uint8,   device=DEVICE),
                torch.tensor(act_f[b],  dtype=torch.long,    device=DEVICE),
                torch.tensor(logp_f[b], dtype=torch.float32, device=DEVICE),
                torch.tensor(adv_f[b],  dtype=torch.float32, device=DEVICE),
                torch.tensor(ret_f[b],  dtype=torch.float32, device=DEVICE),
            )

    def reset(self):
        self.ptr = 0


# =============================================================================
# 4.  PPO Agent
# =============================================================================

class PPOAgent:
    """
    Proximal Policy Optimisation (Schulman et al. 2017).

    Each call to `update()`:
      1. Normalises advantages over the rollout.
      2. Performs n_epochs passes over the rollout, each time splitting it
         into mini-batches and computing the clipped surrogate loss.

    The probability ratio  r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  is
    clipped to [1-ε, 1+ε], preventing excessively large policy updates.
    """
    def __init__(self, n_actions,
                 lr=2.5e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.1, value_coef=0.5, entropy_coef=0.01,
                 n_epochs=4, mini_batch_size=32, max_grad_norm=0.5):
        self.n_actions       = n_actions
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_eps        = clip_eps
        self.value_coef      = value_coef
        self.entropy_coef    = entropy_coef
        self.n_epochs        = n_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm   = max_grad_norm

        self.net       = ActorCriticNet(n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        self.updates    = 0
        self.losses     = []
        self.pol_losses = []
        self.val_losses = []
        self.clip_fracs = []

    # ── Inference helpers ─────────────────────────────────────────────────────

    def act(self, obs):
        """Sample action from π; return (action, log_prob, value)."""
        with torch.no_grad():
            obs_t          = torch.tensor(obs[np.newaxis], dtype=torch.uint8, device=DEVICE)
            logits, value  = self.net(obs_t)
            dist           = torch.distributions.Categorical(logits=logits)
            action         = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def act_vec(self, obs_batch):
        """
        Vectorised action selection for N parallel environments.

        Parameters
        ----------
        obs_batch : np.array (n_envs, 4, 84, 84) uint8

        Returns
        -------
        actions   : np.array (n_envs,) int
        log_probs : np.array (n_envs,) float
        values    : np.array (n_envs,) float
        """
        with torch.no_grad():
            obs_t         = torch.tensor(obs_batch, dtype=torch.uint8, device=DEVICE)
            logits, values = self.net(obs_t)
            dist           = torch.distributions.Categorical(logits=logits)
            actions        = dist.sample()
            return (actions.cpu().numpy(),
                    dist.log_prob(actions).cpu().numpy(),
                    values.cpu().numpy())

    def act_greedy(self, obs):
        """Deterministic greedy action: argmax π(·|s)."""
        with torch.no_grad():
            obs_t     = torch.tensor(obs[np.newaxis], dtype=torch.uint8, device=DEVICE)
            logits, _ = self.net(obs_t)
            return int(logits.argmax(-1).item())

    def policy_probs(self, obs):
        """Full policy distribution π(·|s) as numpy array."""
        with torch.no_grad():
            obs_t     = torch.tensor(obs[np.newaxis], dtype=torch.uint8, device=DEVICE)
            logits, _ = self.net(obs_t)
            return F.softmax(logits, dim=-1).cpu().numpy()[0]

    def state_value(self, obs):
        """V(s) — used as analogue of max_a Q(s,a) for evaluation plots."""
        with torch.no_grad():
            obs_t  = torch.tensor(obs[np.newaxis], dtype=torch.uint8, device=DEVICE)
            _, v   = self.net(obs_t)
            return float(v.item())

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(self, buffer: RolloutBuffer):
        """
        Normalise advantages, then perform n_epochs of mini-batch PPO updates.
        PPO clipped loss prevents destructively large policy steps.
        """
        adv = buffer.advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buffer.advantages = adv

        for _ in range(self.n_epochs):
            for obs_b, act_b, old_logp_b, adv_b, ret_b in buffer.mini_batches(self.mini_batch_size):
                logits, values = self.net(obs_b)
                dist     = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(act_b)
                entropy   = dist.entropy().mean()

                # Clipped surrogate objective
                ratio  = torch.exp(log_probs - old_logp_b)
                surr1  = ratio * adv_b
                surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                pol_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                val_loss = F.mse_loss(values, ret_b)

                # Diagnostic: fraction of ratios that were clipped
                self.clip_fracs.append(((ratio - 1).abs() > self.clip_eps).float().mean().item())

                loss = pol_loss + self.value_coef * val_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                self.updates    += 1
                self.losses.append(float(loss.item()))
                self.pol_losses.append(float(pol_loss.item()))
                self.val_losses.append(float(val_loss.item()))


# =============================================================================
# 5.  Saliency Helpers
#     log π(a|f) serves as the action-specific scalar analogous to Q̂(f, a).
# =============================================================================

def perturbation_saliency(agent: PPOAgent, obs: np.ndarray,
                           action: int, patch_size: int = 8) -> np.ndarray:
    """
    Perturbation-based saliency map (Q4).

    For each P×P pixel patch:
      1. Replace all 4 channel values with the frame mean (neutral grey).
      2. Measure the drop in log π(action | perturbed_frame).

        Sal(i,j) = | log π(a | f*) − log π(a | f̃*_ij) |

    Parameters
    ----------
    obs        : np.array (4, 84, 84) uint8
    action     : int
    patch_size : int  P

    Returns
    -------
    sal_map : np.array (84, 84) float, normalised to [0, 1]
    """
    obs_t = torch.tensor(obs[np.newaxis], dtype=torch.uint8, device=DEVICE)
    with torch.no_grad():
        logits, _ = agent.net(obs_t)
        base_logp = float(F.log_softmax(logits, dim=-1)[0, action].item())

    mean_val  = float(obs.mean())
    n_patches = 84 // patch_size
    sal_map   = np.zeros((84, 84), dtype=float)

    for pi in range(n_patches):
        for pj in range(n_patches):
            r0, r1 = pi * patch_size, (pi + 1) * patch_size
            c0, c1 = pj * patch_size, (pj + 1) * patch_size
            obs_p = obs.copy().astype(float)
            obs_p[:, r0:r1, c0:c1] = mean_val
            with torch.no_grad():
                obs_t2    = torch.tensor(obs_p.clip(0, 255).astype(np.uint8)[np.newaxis],
                                         dtype=torch.uint8, device=DEVICE)
                logits2, _ = agent.net(obs_t2)
                logp2 = float(F.log_softmax(logits2, dim=-1)[0, action].item())
            sal_map[r0:r1, c0:c1] = abs(base_logp - logp2)

    mx = sal_map.max()
    return sal_map / mx if mx > 0 else sal_map


def gradient_saliency(agent: PPOAgent, obs: np.ndarray, action: int) -> np.ndarray:
    """
    Gradient-based saliency (Challenge Q7a).

        Sal_grad(pixel) = | ∂ log π(action|f) / ∂ pixel |

    Summed over the 4 stacked frame channels to give an (84, 84) map.
    One backward pass — much faster than perturbation.

    Returns
    -------
    sal_map : np.array (84, 84) float, normalised to [0, 1]
    """
    obs_t = torch.tensor(obs[np.newaxis], dtype=torch.float32,
                         device=DEVICE, requires_grad=True)
    logits, _ = agent.net(obs_t)
    log_prob   = F.log_softmax(logits, dim=-1)[0, action]
    log_prob.backward()

    grad    = obs_t.grad[0].abs().cpu().numpy()   # (4, 84, 84)
    sal_map = grad.sum(axis=0)                     # (84, 84)
    mx = sal_map.max()
    return sal_map / mx if mx > 0 else sal_map


def overlay_saliency(frame_gray: np.ndarray, sal_map: np.ndarray,
                     alpha: float = 0.5) -> np.ndarray:
    """
    Blend a hot-colourmap saliency heatmap over a grayscale frame.

    Parameters
    ----------
    frame_gray : (84, 84) uint8
    sal_map    : (84, 84) float [0, 1]
    alpha      : heatmap opacity

    Returns
    -------
    overlay : (84, 84, 3) uint8 RGB
    """
    rgb  = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
    heat = (plt.cm.hot(sal_map)[:, :, :3] * 255).astype(np.uint8)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)


def saliency_entropy(sal_map: np.ndarray) -> float:
    """Shannon entropy of a saliency map (high = diffuse, low = focused)."""
    s = sal_map.flatten().astype(float) + 1e-9
    s /= s.sum()
    return float(-(s * np.log(s)).sum())


def adversarial_perturbation(agent: PPOAgent, obs: np.ndarray,
                              a_star: int, eps: float = 0.05,
                              n_iters: int = 50, step_size: float = 1.0) -> np.ndarray:
    """
    Challenge Q7c — find a small L∞ perturbation δ that flips the greedy action.

    Minimises log π(a*|f+δ) via projected gradient descent, constrained to
    ||δ||∞ ≤ eps * 255  (eps expressed as fraction of pixel range [0,1]).

    Returns
    -------
    delta : (4, 84, 84) float32 — raw pixel perturbation
    """
    eps_raw   = eps * 255.0
    obs_float = torch.tensor(obs[np.newaxis], dtype=torch.float32, device=DEVICE)
    delta     = torch.zeros_like(obs_float, requires_grad=False)

    for _ in range(n_iters):
        delta = delta.detach().requires_grad_(True)
        perturbed = (obs_float + delta).clamp(0, 255)
        logits, _ = agent.net(perturbed)
        log_prob   = F.log_softmax(logits, dim=-1)[0, a_star]
        log_prob.backward()   # maximise δ that minimises log π(a*)

        with torch.no_grad():
            delta = delta - step_size * delta.grad.sign()
            delta = delta.clamp(-eps_raw, eps_raw)

    return delta.detach().cpu().numpy()[0]   # (4, 84, 84)


def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


# =============================================================================
# 6.  Hyperparameters
# =============================================================================

if SMOKE:
    TOTAL_STEPS   = 10_000
    N_STEPS       = 128
    N_EPOCHS      = 4
    EVAL_EPISODES = 2
    LR            = 2.5e-4
else:
    TOTAL_STEPS   = 500_000
    N_STEPS       = 128    # rollout steps per environment
    N_EPOCHS      = 4      # PPO update epochs per rollout
    EVAL_EPISODES = 5      # >= 5 required by assignment
    LR            = 2.5e-4

# --steps flag overrides default TOTAL_STEPS
if STEPS_OVERRIDE > 0:
    TOTAL_STEPS = STEPS_OVERRIDE

# Mini-batch size scales with n_envs so each epoch has ~4 mini-batches:
#   n_envs=1  -> batch=128,  MINI_BS=32  (4 mini-batches/epoch)
#   n_envs=8  -> batch=1024, MINI_BS=256 (4 mini-batches/epoch)
MINI_BS = 32 * N_ENVS

# Snap steps: 5 evenly-spaced saliency checkpoints across the run
_s = TOTAL_STEPS
SNAP_STEPS = {_s // 5, 2*_s // 5, 3*_s // 5, 4*_s // 5, _s}

GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.1
VALUE_COEF    = 0.5
ENTROPY_COEF  = 0.01
MAX_GRAD_NORM = 0.5
LOG_FREQ      = 10_000  # log approx every this many steps

HUMAN_BASELINE = 31.8   # Breakout — Mnih et al. (2015)


# =============================================================================
# 7.  Training Loop
# =============================================================================

# ── Build training environment(s) ─────────────────────────────────────────────
# N_ENVS=1  -> single env (CPU mode, backward compatible)
# N_ENVS>1  -> SyncVectorEnv with N_ENVS parallel copies (GPU/Colab mode)
if N_ENVS == 1:
    train_env = make_atari_env(GAME, seed=SEED)
else:
    train_env = make_vec_env(GAME, n_envs=N_ENVS, seed=SEED)

# Resolve n_actions from either single or vectorised env
n_acts = (train_env.single_action_space.n if N_ENVS > 1
          else train_env.action_space.n)

agent  = PPOAgent(
    n_actions       = n_acts,
    lr              = LR,
    gamma           = GAMMA,
    gae_lambda      = GAE_LAMBDA,
    clip_eps        = CLIP_EPS,
    value_coef      = VALUE_COEF,
    entropy_coef    = ENTROPY_COEF,
    n_epochs        = N_EPOCHS,
    mini_batch_size = MINI_BS,
    max_grad_norm   = MAX_GRAD_NORM,
)
buffer = RolloutBuffer(N_STEPS, n_envs=N_ENVS)

# Single env for saliency snapshots (always needed regardless of N_ENVS)
snap_env = make_atari_env(GAME, seed=SEED + 999)

_meanings_env = make_atari_env(GAME, seed=0)
action_meanings = _meanings_env.unwrapped.get_action_meanings()
_meanings_env.close()

total_batch = N_STEPS * N_ENVS
print(f"\nStarting PPO training: {TOTAL_STEPS:,} steps on {GAME}")
print(f"Parallel envs : {N_ENVS}  |  N_STEPS={N_STEPS}  |  "
      f"batch/rollout={total_batch}  |  MINI_BS={MINI_BS}")
print(f"Actions : {n_acts}  |  {action_meanings}")
print(f"Rollouts: ~{TOTAL_STEPS // total_batch}  ×  {total_batch} steps  "
      f"×  {N_EPOCHS} epochs\n")

ep_returns     = []
ep_step_totals = []
ep_lengths     = []
snap_data      = {}

step       = 0
prev_step  = 0
ep_count   = 0
t_start    = time.time()
snaps_done = set()

# Per-environment running episode return/length accumulators
ep_rets = np.zeros(N_ENVS, dtype=np.float64)
ep_lens = np.zeros(N_ENVS, dtype=np.int32)

# Initial reset
if N_ENVS == 1:
    obs, _ = train_env.reset()
    obs    = obs[np.newaxis]          # shape (1, 4, 84, 84)
    dones  = np.array([False])
else:
    obs, _ = train_env.reset()        # shape (N_ENVS, 4, 84, 84)
    dones  = np.zeros(N_ENVS, dtype=bool)

while step < TOTAL_STEPS:
    buffer.reset()

    # ── Collect N_STEPS timesteps across all N_ENVS environments ──────────────
    for t in range(N_STEPS):
        actions, log_probs, values = agent.act_vec(obs)

        if N_ENVS == 1:
            next_obs, rew, terminated, truncated, _ = train_env.step(int(actions[0]))
            next_obs = next_obs[np.newaxis]
            rew      = np.array([rew],        dtype=np.float32)
            dones    = np.array([terminated or truncated])
        else:
            next_obs, rew, terminated, truncated, _ = train_env.step(actions)
            dones = terminated | truncated

        buffer.add(obs, actions, rew, values, log_probs, dones)

        ep_rets += rew
        ep_lens += 1
        step    += N_ENVS

        # Record completed episodes
        for i in range(N_ENVS):
            if dones[i]:
                ep_returns.append(float(ep_rets[i]))
                ep_step_totals.append(step)
                ep_lengths.append(int(ep_lens[i]))
                ep_count += 1
                ep_rets[i] = 0.0
                ep_lens[i] = 0

        # SyncVectorEnv auto-resets; single env needs manual reset
        if N_ENVS == 1 and dones[0]:
            reset_obs, _ = train_env.reset()
            next_obs = reset_obs[np.newaxis]

        obs = next_obs

        if step >= TOTAL_STEPS:
            break

    # ── Bootstrap final values, compute GAE ───────────────────────────────────
    with torch.no_grad():
        obs_t      = torch.tensor(obs, dtype=torch.uint8, device=DEVICE)
        _, last_v  = agent.net(obs_t)
        last_vals  = last_v.cpu().numpy()
    last_vals = np.where(dones, 0.0, last_vals)
    buffer.compute_gae(last_vals, dones, GAMMA, GAE_LAMBDA)

    # ── PPO update ─────────────────────────────────────────────────────────────
    agent.update(buffer)

    # ── Logging (boundary-crossing) ────────────────────────────────────────────
    if (step // LOG_FREQ) > (prev_step // LOG_FREQ) or step >= TOTAL_STEPS:
        avg50   = np.mean(ep_returns[-50:]) if ep_returns else 0.0
        elapsed = time.time() - t_start
        cf      = np.mean(agent.clip_fracs[-100:]) if agent.clip_fracs else 0.0
        pct     = 100.0 * step / TOTAL_STEPS
        eta_s   = (elapsed / step) * (TOTAL_STEPS - step) if step > 0 else 0
        print(f"  step {step:>10,} ({pct:5.1f}%)  avg50={avg50:6.2f}  "
              f"ep={ep_count}  clip={cf:.2f}  "
              f"sps={step/elapsed:.0f}  ETA={eta_s/60:.1f}min", flush=True)

    # ── Saliency entropy snapshot (Q6b) ────────────────────────────────────────
    for snap_s in SNAP_STEPS:
        if snap_s not in snaps_done and prev_step < snap_s <= step and ep_returns:
            snaps_done.add(snap_s)
            snap_obs, _ = snap_env.reset()
            ents = []
            for _ in range(5):
                a   = agent.act_greedy(snap_obs)
                sal = perturbation_saliency(agent, snap_obs, a, patch_size=8)
                ents.append(saliency_entropy(sal))
                snap_obs, _, t2, tr2, _ = snap_env.step(a)
                if t2 or tr2:
                    snap_obs, _ = snap_env.reset()
            snap_data[snap_s] = float(np.mean(ents))
            print(f"    [snap] step={snap_s:,}  saliency_entropy={snap_data[snap_s]:.4f}",
                  flush=True)

    prev_step = step

train_env.close()
snap_env.close()
print(f"\nTraining done — steps: {step:,}  episodes: {ep_count}  "
      f"gradient updates: {agent.updates:,}")


# =============================================================================
# 8.  Greedy Evaluation Rollout  (collect frames, V(s), actions for analysis)
# =============================================================================

print("\nRunning greedy evaluation ...")
eval_env = make_atari_env(GAME, seed=SEED + 1)

e_obs, e_frames, e_actions, e_values, e_rewards = [], [], [], [], []
total_eval = 0.0

for ep_i in range(EVAL_EPISODES):
    obs, _ = eval_env.reset()
    done   = False
    ep_sc  = 0.0
    while not done:
        probs = agent.policy_probs(obs)
        a     = int(np.argmax(probs))
        v     = agent.state_value(obs)
        if ep_i == 0:
            e_obs.append(obs.copy())
            e_frames.append(obs[-1].copy())   # most recent grayscale frame
            e_actions.append(a)
            e_values.append(v)
        obs, r, terminated, truncated, _ = eval_env.step(a)
        done   = terminated or truncated
        ep_sc += r
        if ep_i == 0:
            e_rewards.append(r)
    total_eval += ep_sc
    print(f"  Eval ep {ep_i+1}: score = {ep_sc:.1f}")

eval_env.close()
mean_eval = total_eval / EVAL_EPISODES
print(f"\nMean eval score ({EVAL_EPISODES} eps) : {mean_eval:.1f}")
print(f"Human baseline (Breakout)          : {HUMAN_BASELINE}")

e_obs    = np.array(e_obs)
e_frames = np.array(e_frames)
e_values = np.array(e_values)
T_eval   = len(e_obs)
print(f"Eval episode length                : {T_eval} steps")


# =============================================================================
# 9.  Pivotal Frame Selection  (Q3)
#     Save frames every K steps AND whenever V(s) changes sharply.
#     Select 5 pivotal frames: start, peak-V, trough-V, mid, end.
# =============================================================================

K_SAVE = max(1, T_eval // 30)
V_THRESH = 0.2

cands = list(range(0, T_eval, K_SAVE))
for i in range(1, T_eval):
    if abs(e_values[i] - e_values[i - 1]) > V_THRESH:
        cands.append(i)
cands = sorted(set(cands))
v_c   = e_values[cands]

piv = sorted({
    cands[0],
    cands[int(np.argmax(v_c))],
    cands[int(np.argmin(v_c))],
    cands[len(cands) // 2],
    cands[-1],
})[:5]
while len(piv) < 5:
    piv.append(piv[-1])
print(f"Pivotal frame indices: {piv}")

PIV_LABELS = [
    "P1: Episode start",
    "P2: Peak V(s)\n(high confidence)",
    "P3: Trough V(s)\n(crisis moment)",
    "P4: Mid-episode",
    "P5: Final frame",
]


# =============================================================================
# 10.  Figures
# =============================================================================

SMOOTH_W = max(10, len(ep_returns) // 20)

# ─────────────────────────────────────────────────────────────────────────────
# fig08  Q2a  PPO Learning Curve + Training Losses
# ─────────────────────────────────────────────────────────────────────────────
print("\nSaving fig08_ppo_learning_curve.png ...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Episode return
ax = axes[0]
ax.plot(ep_step_totals, ep_returns, color="#AED6F1", lw=0.5, alpha=0.4, label="Raw")
if len(ep_returns) >= SMOOTH_W:
    sm_r = smooth(ep_returns, SMOOTH_W)
    sm_s = ep_step_totals[SMOOTH_W - 1:]
    ax.plot(sm_s, sm_r, color="#2E5FA3", lw=2, label=f"{SMOOTH_W}-ep smoothed")
ax.axhline(HUMAN_BASELINE, color="red", ls="--", lw=1.2,
           label=f"Human baseline ({HUMAN_BASELINE})")
ax.set_xlabel("Training step", fontsize=11)
ax.set_ylabel("Episode return (clipped reward)", fontsize=11)
ax.set_title(f"PPO Learning Curve — {GAME}", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Value loss
ax = axes[1]
if agent.val_losses:
    stride = max(1, len(agent.val_losses) // 2000)
    ax.plot(agent.val_losses[::stride], color="#C0392B", lw=0.7, alpha=0.8)
ax.set_xlabel("Gradient update", fontsize=11)
ax.set_ylabel("Value MSE loss", fontsize=11, color="#C0392B")
ax.set_title("Critic Value Loss", fontsize=11)
ax.grid(True, alpha=0.3)

# Policy loss
ax = axes[2]
if agent.pol_losses:
    stride = max(1, len(agent.pol_losses) // 2000)
    ax.plot(agent.pol_losses[::stride], color="#27AE60", lw=0.7, alpha=0.8)
    cf_stride = max(1, len(agent.clip_fracs) // 2000)
    ax2 = ax.twinx()
    ax2.plot(agent.clip_fracs[::cf_stride], color="#F39C12", lw=0.5,
             alpha=0.5, label="Clip fraction")
    ax2.set_ylabel("Clip fraction", fontsize=10, color="#F39C12")
    ax2.legend(fontsize=9, loc="upper right")
ax.set_xlabel("Gradient update", fontsize=11)
ax.set_ylabel("Policy (clipped surrogate) loss", fontsize=11, color="#27AE60")
ax.set_title("Actor Policy Loss + Clip Fraction", fontsize=11)
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Problem 2 — PPO on Atari  |  Mean eval: {mean_eval:.1f}  |  "
    f"Human: {HUMAN_BASELINE}  (Q2a)",
    fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(fig_path("fig08_ppo_learning_curve.png"), dpi=150)
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# fig09  Q2b  V(s) Over Evaluation Episode + Cumulative Reward
# ─────────────────────────────────────────────────────────────────────────────
print("Saving fig09_value_over_episode.png ...")
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

ax = axes[0]
ax.plot(e_values, color="#2E5FA3", lw=1)
for rank, idx in enumerate(piv):
    ax.axvline(idx, color="red", lw=1, ls="--", alpha=0.7)
    off = max(1, T_eval // 60)
    ax.annotate(f"P{rank+1}", xy=(idx, e_values[idx]),
                xytext=(idx + off, e_values[idx] + 0.05),
                fontsize=9, color="darkred",
                arrowprops=dict(arrowstyle="-", color="red", lw=0.7))
ax.set_ylabel("Critic value  V(s)", fontsize=12)
ax.set_title(
    "Critic V(s) During Greedy Evaluation Episode (Q2b)\n"
    "V(s) ≈ expected discounted future return from state s",
    fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(np.cumsum(e_rewards), color="#117A65", lw=1.5)
ax.set_xlabel("Step in evaluation episode", fontsize=12)
ax.set_ylabel("Cumulative reward", fontsize=12)
ax.set_title("Cumulative Reward (unclipped)", fontsize=11)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(fig_path("fig09_value_over_episode.png"), dpi=150)
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# fig10  Q3  Pivotal Game Frames
# ─────────────────────────────────────────────────────────────────────────────
print("Saving fig10_pivotal_frames.png ...")
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for ax, idx, lbl in zip(axes, piv, PIV_LABELS):
    ax.imshow(e_frames[idx], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    action_name = action_meanings[e_actions[idx]]
    ax.set_title(f"{lbl}\nstep {idx}  a={action_name}\nV={e_values[idx]:.3f}",
                 fontsize=8)
    ax.axis("off")
fig.suptitle(
    "Pivotal Frames — 5 Decision-Critical Moments in the Evaluation Episode (Q3)\n"
    "Selected by: episode start/end, peak/trough V(s), and sharp value changes",
    fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(fig_path("fig10_pivotal_frames.png"), dpi=150)
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# fig11  Q4  Perturbation Saliency Maps — Greedy Action a*
# ─────────────────────────────────────────────────────────────────────────────
print("Saving fig11_saliency_greedy.png ...")
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for ci, (idx, lbl) in enumerate(zip(piv, PIV_LABELS)):
    obs_i   = e_obs[idx]
    frame   = e_frames[idx]
    probs   = agent.policy_probs(obs_i)
    a_star  = int(np.argmax(probs))
    sal     = perturbation_saliency(agent, obs_i, a_star, patch_size=8)

    axes[0][ci].imshow(frame, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0][ci].set_title(f"{lbl}\nstep {idx}", fontsize=8)
    axes[0][ci].axis("off")

    axes[1][ci].imshow(overlay_saliency(frame, sal, alpha=0.55), interpolation="nearest")
    axes[1][ci].set_title(
        f"a*={action_meanings[a_star]}\n"
        f"π(a*)={probs[a_star]:.2f}  V={e_values[idx]:.2f}",
        fontsize=7)
    axes[1][ci].axis("off")

axes[0][0].set_ylabel("Original frame", fontsize=10)
axes[1][0].set_ylabel("Saliency (P=8)\ngreedy a*", fontsize=10)
fig.suptitle(
    "Perturbation Saliency Maps — Greedy Action a*  (Q4)\n"
    r"Sal(i,j) = |log π(a*|f*) − log π(a*|f̃*_ij)|   "
    "Warmer = higher Q sensitivity",
    fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(fig_path("fig11_saliency_greedy.png"), dpi=150)
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# fig12  Q5  Saliency Across Actions — Greedy a* vs Non-Greedy a'
# ─────────────────────────────────────────────────────────────────────────────
print("Saving fig12_saliency_nongreedy.png ...")
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
for ci, (idx, lbl) in enumerate(zip(piv, PIV_LABELS)):
    obs_i  = e_obs[idx]
    frame  = e_frames[idx]
    probs  = agent.policy_probs(obs_i)
    a_star = int(np.argmax(probs))
    a_alt  = int(np.argsort(probs)[-2])

    sal_star = perturbation_saliency(agent, obs_i, a_star, patch_size=8)
    sal_alt  = perturbation_saliency(agent, obs_i, a_alt,  patch_size=8)

    axes[0][ci].imshow(frame, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0][ci].set_title(f"{lbl}\nstep {idx}", fontsize=8)
    axes[0][ci].axis("off")

    axes[1][ci].imshow(overlay_saliency(frame, sal_star, alpha=0.55), interpolation="nearest")
    axes[1][ci].set_title(f"a* = {action_meanings[a_star]}\nπ = {probs[a_star]:.2f}", fontsize=8)
    axes[1][ci].axis("off")

    axes[2][ci].imshow(overlay_saliency(frame, sal_alt, alpha=0.55), interpolation="nearest")
    axes[2][ci].set_title(f"a' = {action_meanings[a_alt]}\nπ = {probs[a_alt]:.2f}", fontsize=8)
    axes[2][ci].axis("off")

axes[0][0].set_ylabel("Original frame", fontsize=10)
axes[1][0].set_ylabel("Greedy a*\nsaliency", fontsize=10)
axes[2][0].set_ylabel("Non-greedy a'\nsaliency", fontsize=10)
fig.suptitle(
    "Saliency Across Actions — Greedy a* vs Second-Best a'  (Q5)\n"
    "Shared bright regions = globally important features; "
    "differing regions = action-discriminative features",
    fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(fig_path("fig12_saliency_nongreedy.png"), dpi=150)
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# fig13  Q6a  Effect of Patch Size on Saliency Resolution
# ─────────────────────────────────────────────────────────────────────────────
print("Saving fig13_patch_size_comparison.png ...")
idx_p2 = piv[1]   # peak-value pivotal frame
obs_p2 = e_obs[idx_p2]
frm_p2 = e_frames[idx_p2]
a_p2   = int(np.argmax(agent.policy_probs(obs_p2)))

PATCH_SIZES = [4, 8, 14]
fig, axes   = plt.subplots(2, 3, figsize=(13, 7))
for ci, ps in enumerate(PATCH_SIZES):
    axes[0][ci].imshow(frm_p2, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0][ci].set_title(f"Original frame  P={ps}", fontsize=10)
    axes[0][ci].axis("off")

for ci, ps in enumerate(PATCH_SIZES):
    sal = perturbation_saliency(agent, obs_p2, a_p2, patch_size=ps)
    ent = saliency_entropy(sal)
    axes[1][ci].imshow(overlay_saliency(frm_p2, sal, alpha=0.55), interpolation="nearest")
    axes[1][ci].set_title(
        f"P={ps}  ({84//ps}×{84//ps} grid)\nEntropy={ent:.2f}", fontsize=10)
    axes[1][ci].axis("off")

axes[0][0].set_ylabel("Original frame", fontsize=10)
axes[1][0].set_ylabel("Saliency overlay", fontsize=10)
fig.suptitle(
    "Effect of Patch Size P on Saliency Resolution (Q6a) — Pivotal Frame P2\n"
    "Small P = fine-grained but noisy;  Large P = coarse but robust",
    fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(fig_path("fig13_patch_size_comparison.png"), dpi=150)
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# fig14  Q6b  Saliency Entropy vs Training Step
# ─────────────────────────────────────────────────────────────────────────────
print("Saving fig14_saliency_entropy_training.png ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
if snap_data:
    xv = sorted(snap_data.keys())
    yv = [snap_data[s] for s in xv]
    ax.plot([x / 1e3 for x in xv], yv, "o-", color="#2E5FA3", lw=2, ms=8)
    for x, y in zip(xv, yv):
        ax.annotate(f"{y:.3f}", (x / 1e3, y),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9)
else:
    ax.text(0.5, 0.5, "No snapshot data\n(run full training, not --smoke-test)",
            ha="center", va="center", transform=ax.transAxes, fontsize=11)
ax.set_xlabel("Training step (thousands)", fontsize=12)
ax.set_ylabel("Mean saliency entropy", fontsize=12)
ax.set_title(
    "Saliency Entropy vs Training Step (Q6b)\n"
    "Early training: high entropy (diffuse attention)\n"
    "After convergence: low entropy (focused on game-relevant pixels)",
    fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
if len(ep_returns) >= SMOOTH_W:
    sm_r = smooth(ep_returns, SMOOTH_W)
    sm_s = [s / 1e3 for s in ep_step_totals[SMOOTH_W - 1:]]
    ax.plot([s / 1e3 for s in ep_step_totals], ep_returns,
            color="#AED6F1", lw=0.5, alpha=0.4)
    ax.plot(sm_s, sm_r, color="#C0392B", lw=2, label=f"{SMOOTH_W}-ep smoothed")
else:
    ax.plot([s / 1e3 for s in ep_step_totals], ep_returns, color="#C0392B", lw=1)
for sv in snap_data:
    ax.axvline(sv / 1e3, color="gray", lw=0.8, ls=":", alpha=0.7)
ax.axhline(HUMAN_BASELINE, color="red", ls="--", lw=1,
           label=f"Human ({HUMAN_BASELINE})")
ax.set_xlabel("Training step (thousands)", fontsize=12)
ax.set_ylabel("Episode return", fontsize=12)
ax.set_title("Learning Curve (dotted = entropy snapshot points)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle(
    "Saliency Entropy Decreases as PPO Policy Specialises (Q6b)",
    fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(fig_path("fig14_saliency_entropy_training.png"), dpi=150)
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# fig15  Q7a  Gradient vs Perturbation Saliency (Challenge)
# ─────────────────────────────────────────────────────────────────────────────
print("Saving fig15_gradient_vs_perturbation.png ...")
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
for ci, (idx, lbl) in enumerate(zip(piv, PIV_LABELS)):
    obs_i  = e_obs[idx]
    frame  = e_frames[idx]
    a_star = int(np.argmax(agent.policy_probs(obs_i)))

    sal_pert = perturbation_saliency(agent, obs_i, a_star, patch_size=8)
    sal_grad = gradient_saliency(agent, obs_i, a_star)

    sp  = sal_pert.flatten()
    sg  = sal_grad.flatten()
    cos = float(np.dot(sp, sg) / (np.linalg.norm(sp) * np.linalg.norm(sg) + 1e-9))

    axes[0][ci].imshow(frame, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0][ci].set_title(f"{lbl}\nstep {idx}", fontsize=8)
    axes[0][ci].axis("off")

    axes[1][ci].imshow(overlay_saliency(frame, sal_pert, alpha=0.55), interpolation="nearest")
    axes[1][ci].set_title(f"Perturbation\nP=8  cos={cos:.2f}", fontsize=8)
    axes[1][ci].axis("off")

    axes[2][ci].imshow(overlay_saliency(frame, sal_grad, alpha=0.55), interpolation="nearest")
    axes[2][ci].set_title(f"Gradient\n|∂logπ/∂f|  cos={cos:.2f}", fontsize=8)
    axes[2][ci].axis("off")

axes[0][0].set_ylabel("Original frame", fontsize=10)
axes[1][0].set_ylabel("Perturbation\nsaliency (P=8)", fontsize=10)
axes[2][0].set_ylabel("Gradient\nsaliency", fontsize=10)
fig.suptitle(
    "Challenge Q7a: Gradient vs Perturbation Saliency\n"
    "Cosine similarity measures spatial agreement between the two methods\n"
    "Gradient: fine-grained but sensitive to noise; "
    "Perturbation: coarser but interaction-aware",
    fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(fig_path("fig15_gradient_vs_perturbation.png"), dpi=150)
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# fig16  Q7c  Adversarial Frames (Challenge)
#   Find δ with ||δ||∞ ≤ ε that flips the greedy action.
# ─────────────────────────────────────────────────────────────────────────────
print("Saving fig16_adversarial_frames.png ...")
ADV_IDX = piv[1]   # use peak-V pivotal frame for adversarial analysis
obs_adv = e_obs[ADV_IDX]
frm_adv = e_frames[ADV_IDX]
probs_orig = agent.policy_probs(obs_adv)
a_star_adv = int(np.argmax(probs_orig))

# Binary search for minimum ε that flips the greedy action
def action_after_delta(obs, delta):
    perturbed = (obs.astype(float) + delta).clip(0, 255).astype(np.uint8)
    return agent.act_greedy(perturbed)

eps_values  = [0.01, 0.02, 0.05, 0.10]
min_flip_eps = None
delta_flip   = None
a_flipped    = None

for eps_try in eps_values:
    delta_try = adversarial_perturbation(agent, obs_adv, a_star_adv,
                                          eps=eps_try, n_iters=50)
    a_new = action_after_delta(obs_adv, delta_try)
    if a_new != a_star_adv:
        min_flip_eps = eps_try
        delta_flip   = delta_try
        a_flipped    = a_new
        break

if delta_flip is None:
    # Use largest epsilon result anyway for visualisation
    delta_flip = adversarial_perturbation(agent, obs_adv, a_star_adv,
                                           eps=eps_values[-1], n_iters=50)
    a_flipped  = action_after_delta(obs_adv, delta_flip)
    min_flip_eps = eps_values[-1]

obs_perturbed = (obs_adv.astype(float) + delta_flip).clip(0, 255).astype(np.uint8)
frm_perturbed = obs_perturbed[-1]

# Saliency of original vs perturbed frame
sal_orig = perturbation_saliency(agent, obs_adv, a_star_adv, patch_size=8)
sal_pert_adv = perturbation_saliency(agent, obs_perturbed,
                                      a_flipped if a_flipped != a_star_adv else a_star_adv,
                                      patch_size=8)

# δ visualisation: amplify for visibility
delta_vis = delta_flip[-1]  # last (most recent) channel
delta_norm = (delta_vis - delta_vis.min())
dmax = delta_norm.max()
delta_norm = (delta_norm / dmax * 255).astype(np.uint8) if dmax > 0 else delta_norm.astype(np.uint8)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0][0].imshow(frm_adv, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
axes[0][0].set_title(f"Original frame\na* = {action_meanings[a_star_adv]}\n"
                     f"π(a*) = {probs_orig[a_star_adv]:.3f}", fontsize=9)
axes[0][0].axis("off")

axes[0][1].imshow(delta_norm, cmap="RdBu", interpolation="nearest")
axes[0][1].set_title(f"Perturbation δ (amplified)\n||δ||∞ = {min_flip_eps:.2f}×255\n"
                     f"Red=+, Blue=−", fontsize=9)
axes[0][1].axis("off")

axes[0][2].imshow(frm_perturbed, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
probs_adv = agent.policy_probs(obs_perturbed)
axes[0][2].set_title(f"Perturbed frame\na' = {action_meanings[a_flipped]}\n"
                     f"π(a') = {probs_adv[a_flipped]:.3f}", fontsize=9)
axes[0][2].axis("off")

# Bar chart: policy before vs after
ax = axes[0][3]
x     = np.arange(n_acts)
width = 0.35
ax.bar(x - width/2, probs_orig,    width, label="Original π",    color="#2E5FA3", alpha=0.8)
ax.bar(x + width/2, probs_adv,     width, label="Perturbed π",   color="#C0392B", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(action_meanings, rotation=30, fontsize=8)
ax.set_ylabel("π(a|s)", fontsize=10)
ax.set_title(f"Policy Before vs After δ\nMin ε to flip: {min_flip_eps:.2f}", fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# Bottom row: saliency comparison
axes[1][0].imshow(overlay_saliency(frm_adv, sal_orig, alpha=0.55), interpolation="nearest")
axes[1][0].set_title(f"Saliency on original\na*={action_meanings[a_star_adv]}", fontsize=9)
axes[1][0].axis("off")

axes[1][1].imshow(overlay_saliency(frm_perturbed, sal_pert_adv, alpha=0.55), interpolation="nearest")
axes[1][1].set_title(f"Saliency on perturbed\na'={action_meanings[a_flipped]}", fontsize=9)
axes[1][1].axis("off")

# Difference of saliency maps
sal_diff = np.abs(sal_orig - sal_pert_adv)
axes[1][2].imshow(sal_diff, cmap="hot", interpolation="nearest")
axes[1][2].set_title("Saliency difference\n|sal_orig − sal_perturbed|", fontsize=9)
axes[1][2].axis("off")

# δ concentration vs saliency concentration
axes[1][3].scatter(sal_orig.flatten(),
                   np.abs(delta_flip[-1]).flatten() / (np.abs(delta_flip[-1]).max() + 1e-9),
                   alpha=0.15, s=5, color="#2E5FA3")
axes[1][3].set_xlabel("Saliency score", fontsize=9)
axes[1][3].set_ylabel("Normalised |δ|", fontsize=9)
axes[1][3].set_title("Do adversarial pixels\nconcentrate on salient regions?", fontsize=9)
axes[1][3].grid(True, alpha=0.3)

fig.suptitle(
    f"Challenge Q7c: Adversarial Perturbation — Minimal ε={min_flip_eps:.2f} flips "
    f"{action_meanings[a_star_adv]} → {action_meanings[a_flipped]}\n"
    "Perturbations that exploit salient regions flip the action with smaller ε",
    fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(fig_path("fig16_adversarial_frames.png"), dpi=150)
plt.close(fig)


# =============================================================================
# 11.  Checkpoint
# =============================================================================

ckpt = fig_path("ppo_checkpoint.pt")
torch.save({
    "updates":    agent.updates,
    "net_state":  agent.net.state_dict(),
    "opt_state":  agent.optimizer.state_dict(),
    "ep_returns": ep_returns,
    "ep_steps":   ep_step_totals,
}, ckpt)
print(f"\nCheckpoint saved: {ckpt}")


# =============================================================================
# 12.  Summary
# =============================================================================
print(f"\n{'='*65}")
print(f"  TRAINING SUMMARY — PPO (Proximal Policy Optimisation)")
print(f"{'='*65}")
print(f"  Game                        : {GAME}")
print(f"  Total env steps             : {step:,}")
print(f"  Total episodes              : {ep_count}")
print(f"  Gradient updates            : {agent.updates:,}")
print(f"  Final 50-ep avg return      : "
      f"{np.mean(ep_returns[-50:] if ep_returns else [0]):.2f}")
print(f"  Mean eval score ({EVAL_EPISODES} eps)     : {mean_eval:.2f}")
print(f"  Human baseline (Breakout)   : {HUMAN_BASELINE}")
print(f"  Superhuman?                 : "
      f"{'YES' if mean_eval > HUMAN_BASELINE else 'NO — more training needed'}")
print(f"  Min adv. eps to flip action : {min_flip_eps:.2f}")
print()
print(f"  Figures (fig08–fig16)       : {SCRIPT_DIR}")
print(f"  Checkpoint                  : {ckpt}")
print(f"{'='*65}")
