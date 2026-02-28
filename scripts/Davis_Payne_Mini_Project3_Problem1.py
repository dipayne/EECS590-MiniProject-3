"""
Davis_Payne_Mini_Project3_Problem1.py

Problem 1: Cadmium Rod Control in a Nuclear Reactor
====================================================

Dependencies: numpy, matplotlib   (standard scientific Python stack)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Save figures next to this script, wherever it is run from ─────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def fig_path(name):
    return os.path.join(SCRIPT_DIR, name)

# ── Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)


# =============================================================================
# ENVIRONMENT
# =============================================================================
class ReactorEnv:
    """
    Cadmium Rod Control MDP.

    The true reactivity mu_t is hidden. The agent observes a noisy,
    discretised gauge reading z_t ~ N(mu_t, sigma^2).

    Parameters
    ----------
    n_bins   : number of observation bins partitioning [mu_min, mu_max]
    k        : max rod action magnitude  (A = {-k, ..., 0, ..., +k})
    mu_min   : lower physical limit
    mu_max   : upper limit / meltdown threshold
    mu_lo    : lower productive-zone boundary
    mu_hi    : upper productive-zone boundary
    mu_hot   : threshold above which intrinsic upward drift activates
    alpha    : rod effectiveness (one unit shifts mean by alpha)
    delta    : drift magnitude when mu >= mu_hot
    sigma    : sensor noise std
    sigma_T  : process noise std
    c        : rod-movement cost coefficient
    M        : meltdown penalty magnitude
    T        : maximum episode length
    """

    def __init__(
        self,
        n_bins=25, k=2,
        mu_min=0.0, mu_max=10.0,
        mu_lo=3.0,  mu_hi=7.0, mu_hot=5.0,
        alpha=0.5,  delta=0.3,
        sigma=0.8,  sigma_T=0.2,
        c=0.05,     M=20.0,
        T=200,
    ):
        self.n_bins   = n_bins
        self.k        = k
        self.actions  = list(range(-k, k + 1))
        self.n_actions = len(self.actions)
        self.mu_min   = mu_min
        self.mu_max   = mu_max
        self.mu_lo    = mu_lo
        self.mu_hi    = mu_hi
        self.mu_hot   = mu_hot
        self.alpha    = alpha
        self.delta    = delta
        self.sigma    = sigma
        self.sigma_T  = sigma_T
        self.c        = c
        self.M        = M
        self.T        = T
        self.bin_width = (mu_max - mu_min) / n_bins

    def _to_bin(self, z):
        idx = int((z - self.mu_min) / self.bin_width)
        return int(np.clip(idx, 0, self.n_bins - 1))

    def _drift(self, mu):
        return self.delta if mu >= self.mu_hot else 0.0

    def reset(self):
        self.mu   = self.mu_min + np.random.uniform(0.0, 0.5)
        self.t    = 0
        self.done = False
        z = self.mu + np.random.randn() * self.sigma
        return self._to_bin(float(np.clip(z, self.mu_min, self.mu_max)))

    def step(self, action_idx):
        assert not self.done
        a      = self.actions[action_idx]
        eps    = np.random.randn() * self.sigma_T
        new_mu = self.mu - self.alpha * a + self._drift(self.mu) + eps
        self.mu = float(np.clip(new_mu, self.mu_min, self.mu_max))
        self.t += 1

        # reward
        if self.mu >= self.mu_max:
            reward = -self.M
        elif self.mu < self.mu_lo:
            reward = -self.c * abs(a)
        else:
            reward = (self.mu - self.mu_lo) - self.c * abs(a)

        meltdown  = self.mu >= self.mu_max
        self.done = meltdown or (self.t >= self.T)

        z     = self.mu + np.random.randn() * self.sigma
        s     = self._to_bin(float(np.clip(z, self.mu_min, self.mu_max)))
        info  = {"mu": self.mu, "meltdown": meltdown}
        return s, reward, self.done, info


# =============================================================================
# AGENTS
# =============================================================================
class SARSALambda:
    """
    On-policy SARSA with accumulating eligibility traces.
    lambda=0 reduces to one-step TD(0); lambda=1 to Monte Carlo.
    """

    def __init__(self, n_states, n_actions,
                 alpha=0.1, gamma=0.99, lam=0.85,
                 eps_start=1.0, eps_end=0.05, eps_decay=500):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.lam       = lam
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.episode   = 0
        self.Q = np.zeros((n_states, n_actions))
        self.E = np.zeros((n_states, n_actions))

    @property
    def epsilon(self):
        frac = min(self.episode / max(self.eps_decay, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def reset_traces(self):
        self.E[:] = 0.0

    def update(self, s, a, r, s_next, a_next, done):
        target     = r if done else r + self.gamma * self.Q[s_next, a_next]
        delta      = target - self.Q[s, a]
        self.E[s, a] += 1.0
        self.Q    += self.alpha * delta * self.E
        if done:
            self.E[:] = 0.0
        else:
            self.E *= self.gamma * self.lam

    def end_episode(self):
        self.episode += 1
        self.reset_traces()


class QLearning:
    """Off-policy Q-learning (greedy target, lambda = 0)."""

    def __init__(self, n_states, n_actions,
                 alpha=0.1, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=500):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.episode   = 0
        self.Q = np.zeros((n_states, n_actions))

    @property
    def epsilon(self):
        frac = min(self.episode / max(self.eps_decay, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def reset_traces(self): pass

    def update(self, s, a, r, s_next, a_next, done):
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def end_episode(self):
        self.episode += 1


class RBFAgent:
    """
    Linear Q-function approximator with RBF (Gaussian) features.
    phi(s, a) = Gaussian bumps centred on each bin, one set per action.
    Provides smooth generalisation across nearby bins.
    """

    def __init__(self, n_states, n_actions,
                 alpha=0.05, gamma=0.99, bandwidth=1.5,
                 eps_start=1.0, eps_end=0.05, eps_decay=500):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.episode   = 0
        self.h2        = bandwidth ** 2
        self.centres   = np.arange(n_states, dtype=float)
        self.w         = np.zeros(n_actions * n_states)

    @property
    def epsilon(self):
        frac = min(self.episode / max(self.eps_decay, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _phi(self, s, a):
        phi = np.zeros(self.n_actions * self.n_states)
        rbf = np.exp(-((s - self.centres) ** 2) / (2 * self.h2))
        phi[a * self.n_states: (a + 1) * self.n_states] = rbf
        return phi

    def _q(self, s, a):
        return float(self._phi(s, a) @ self.w)

    def _q_all(self, s):
        return np.array([self._q(s, a) for a in range(self.n_actions)])

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self._q_all(state)))

    def reset_traces(self): pass

    def update(self, s, a, r, s_next, a_next, done):
        target = r if done else r + self.gamma * np.max(self._q_all(s_next))
        delta  = target - self._q(s, a)
        self.w += self.alpha * delta * self._phi(s, a)

    def end_episode(self):
        self.episode += 1

    @property
    def Q(self):
        """Return full Q-table for heatmap plotting."""
        Q = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                Q[s, a] = self._q(s, a)
        return Q


# =============================================================================
# TRAINING UTILITIES
# =============================================================================
def run_episode(env, agent):
    """Run one full episode; return (total_return, meltdown_flag)."""
    s    = env.reset()
    agent.reset_traces()
    done = False
    ret  = 0.0
    a    = agent.select_action(s)
    while not done:
        s2, r, done, info = env.step(a)
        a2 = agent.select_action(s2)
        agent.update(s, a, r, s2, a2, done)
        ret += r
        s, a = s2, a2
    agent.end_episode()
    return ret, info["meltdown"]


def train(env, agent, n_episodes):
    returns   = np.zeros(n_episodes)
    meltdowns = np.zeros(n_episodes, dtype=int)
    for i in range(n_episodes):
        returns[i], meltdowns[i] = run_episode(env, agent)
    return returns, meltdowns


def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


# =============================================================================
# HYPERPARAMETERS
# =============================================================================
N_EPISODES = 3000
N_BINS     = 25
ALPHA      = 0.1
GAMMA      = 0.99
EPS_START  = 1.0
EPS_END    = 0.05
EPS_DECAY  = 500
SMOOTH_WIN = 50

SIGMA_LOW  = np.sqrt(0.1)
SIGMA_HIGH = np.sqrt(1.0)

PALETTE = {
    "sarsa_low":   "#2E5FA3",
    "qlearn_low":  "#C0392B",
    "sarsa_high":  "#5DADE2",
    "qlearn_high": "#E74C3C",
    "lam0":        "#1A5276",
    "lam08":       "#A93226",
    "rbf":         "#117A65",
    "tabular":     "#6C3483",
}


# =============================================================================
# TRAIN ALL AGENTS
# =============================================================================
def make_env(sigma):
    return ReactorEnv(n_bins=N_BINS, sigma=sigma)

def make_sarsa(sigma, lam):
    env = make_env(sigma)
    return env, SARSALambda(N_BINS, env.n_actions, alpha=ALPHA, gamma=GAMMA,
                             lam=lam, eps_start=EPS_START, eps_end=EPS_END,
                             eps_decay=EPS_DECAY)

def make_qlearn(sigma):
    env = make_env(sigma)
    return env, QLearning(N_BINS, env.n_actions, alpha=ALPHA, gamma=GAMMA,
                           eps_start=EPS_START, eps_end=EPS_END,
                           eps_decay=EPS_DECAY)


print("Training SARSA(lam=0.85) - low noise  ...")
env, agent = make_sarsa(SIGMA_LOW, 0.85)
ret_sarsa_low, melt_sarsa_low = train(env, agent, N_EPISODES)
sarsa_low_agent = agent

print("Training Q-learning      - low noise  ...")
env, agent = make_qlearn(SIGMA_LOW)
ret_qlearn_low, melt_qlearn_low = train(env, agent, N_EPISODES)
qlearn_low_agent = agent

print("Training SARSA(lam=0.85) - high noise ...")
env, agent = make_sarsa(SIGMA_HIGH, 0.85)
ret_sarsa_high, melt_sarsa_high = train(env, agent, N_EPISODES)

print("Training Q-learning      - high noise ...")
env, agent = make_qlearn(SIGMA_HIGH)
ret_qlearn_high, melt_qlearn_high = train(env, agent, N_EPISODES)

print("Training SARSA(lam=0)    - low noise  ...")
env, agent = make_sarsa(SIGMA_LOW, 0.0)
ret_lam0, melt_lam0 = train(env, agent, N_EPISODES)

print("Training SARSA(lam=0.8)  - low noise  ...")
env, agent = make_sarsa(SIGMA_LOW, 0.8)
ret_lam08, melt_lam08 = train(env, agent, N_EPISODES)

print("Training RBF agent       - low noise  ...")
env_rbf   = make_env(SIGMA_LOW)
rbf_agent = RBFAgent(N_BINS, env_rbf.n_actions, alpha=0.05, gamma=GAMMA,
                     bandwidth=1.5, eps_start=EPS_START, eps_end=EPS_END,
                     eps_decay=EPS_DECAY)
ret_rbf, melt_rbf = train(env_rbf, rbf_agent, N_EPISODES)

print("All training complete.\n")

# x-axis for smoothed plots
eps_x = np.arange(SMOOTH_WIN, N_EPISODES + 1)

# Reference env for zone annotations
env_ref = make_env(SIGMA_LOW)
actions = env_ref.actions


# =============================================================================
# FIGURE 1: Learning curves — low noise
# =============================================================================
print("Saving fig1_learning_curves_low_noise.png ...")
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(eps_x, smooth(ret_sarsa_low,  SMOOTH_WIN),
        color=PALETTE["sarsa_low"],  lw=2, label="SARSA(λ=0.85)")
ax.plot(eps_x, smooth(ret_qlearn_low, SMOOTH_WIN),
        color=PALETTE["qlearn_low"], lw=2, label="Q-learning")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Smoothed Return (50-ep rolling mean)", fontsize=12)
ax.set_title(r"Learning Curves — Low Noise ($\sigma^2 = 0.1$)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(fig_path("fig1_learning_curves_low_noise.png"), dpi=150)
plt.close(fig)


# =============================================================================
# FIGURE 2: Learning curves — high noise
# =============================================================================
print("Saving fig2_learning_curves_high_noise.png ...")
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(eps_x, smooth(ret_sarsa_high,  SMOOTH_WIN),
        color=PALETTE["sarsa_high"],  lw=2, label="SARSA(λ=0.85)")
ax.plot(eps_x, smooth(ret_qlearn_high, SMOOTH_WIN),
        color=PALETTE["qlearn_high"], lw=2, label="Q-learning")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Smoothed Return (50-ep rolling mean)", fontsize=12)
ax.set_title(r"Learning Curves — High Noise ($\sigma^2 = 1.0$)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(fig_path("fig2_learning_curves_high_noise.png"), dpi=150)
plt.close(fig)


# =============================================================================
# FIGURE 3: Meltdown rates
# =============================================================================
print("Saving fig3_meltdown_rates.png ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (ml_s, ml_q, title) in zip(axes, [
    (melt_sarsa_low,  melt_qlearn_low,  r"Low Noise ($\sigma^2=0.1$)"),
    (melt_sarsa_high, melt_qlearn_high, r"High Noise ($\sigma^2=1.0$)"),
]):
    ax.plot(eps_x, smooth(ml_s.astype(float), SMOOTH_WIN) * 100,
            color=PALETTE["sarsa_low"],  lw=2, label="SARSA(λ=0.85)")
    ax.plot(eps_x, smooth(ml_q.astype(float), SMOOTH_WIN) * 100,
            color=PALETTE["qlearn_low"], lw=2, label="Q-learning")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Meltdown Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
fig.suptitle("Meltdown Rate during Training", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(fig_path("fig3_meltdown_rates.png"), dpi=150)
plt.close(fig)


# =============================================================================
# FIGURE 4: Q-function heatmap — SARSA
# =============================================================================
print("Saving fig4_qfunction_heatmap_sarsa.png ...")
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(sarsa_low_agent.Q.T, aspect="auto", origin="lower",
               cmap="RdYlGn", interpolation="nearest")
plt.colorbar(im, ax=ax, label="Q-value")
ax.set_xlabel("State bin  (0 = cold  →  24 = near meltdown)", fontsize=12)
ax.set_ylabel("Action index", fontsize=12)
ax.set_yticks(range(len(actions)))
ax.set_yticklabels([f"a={a}" for a in actions], fontsize=9)
ax.set_title("Learned Q-Function: SARSA(λ=0.85), Low Noise", fontsize=13)
ax.axvline(int(N_BINS * env_ref.mu_lo / env_ref.mu_max) - 0.5,
           color="blue",   lw=1.5, ls="--", label="µ_lo")
ax.axvline(int(N_BINS * env_ref.mu_hi / env_ref.mu_max) - 0.5,
           color="orange", lw=1.5, ls="--", label="µ_hi")
ax.axvline(N_BINS - 1.5,   color="red", lw=1.5, ls="--", label="meltdown")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(fig_path("fig4_qfunction_heatmap_sarsa.png"), dpi=150)
plt.close(fig)


# =============================================================================
# FIGURE 5: Q-function heatmap — Q-learning
# =============================================================================
print("Saving fig5_qfunction_heatmap_qlearn.png ...")
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(qlearn_low_agent.Q.T, aspect="auto", origin="lower",
               cmap="RdYlGn", interpolation="nearest")
plt.colorbar(im, ax=ax, label="Q-value")
ax.set_xlabel("State bin  (0 = cold  →  24 = near meltdown)", fontsize=12)
ax.set_ylabel("Action index", fontsize=12)
ax.set_yticks(range(len(actions)))
ax.set_yticklabels([f"a={a}" for a in actions], fontsize=9)
ax.set_title("Learned Q-Function: Q-learning, Low Noise", fontsize=13)
ax.axvline(int(N_BINS * env_ref.mu_lo / env_ref.mu_max) - 0.5,
           color="blue",   lw=1.5, ls="--", label="µ_lo")
ax.axvline(int(N_BINS * env_ref.mu_hi / env_ref.mu_max) - 0.5,
           color="orange", lw=1.5, ls="--", label="µ_hi")
ax.axvline(N_BINS - 1.5,   color="red", lw=1.5, ls="--", label="meltdown")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(fig_path("fig5_qfunction_heatmap_qlearn.png"), dpi=150)
plt.close(fig)


# =============================================================================
# FIGURE 6: Lambda comparison (Q3c)
# =============================================================================
print("Saving fig6_lambda_comparison.png ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(eps_x, smooth(ret_lam0,  SMOOTH_WIN),
        color=PALETTE["lam0"],  lw=2, label="SARSA(λ=0)")
ax.plot(eps_x, smooth(ret_lam08, SMOOTH_WIN),
        color=PALETTE["lam08"], lw=2, label="SARSA(λ=0.8)")
ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Smoothed Return", fontsize=12)
ax.set_title("Episode Return: λ=0 vs λ=0.8", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(eps_x, smooth(melt_lam0.astype(float),  SMOOTH_WIN) * 100,
        color=PALETTE["lam0"],  lw=2, label="SARSA(λ=0)")
ax.plot(eps_x, smooth(melt_lam08.astype(float), SMOOTH_WIN) * 100,
        color=PALETTE["lam08"], lw=2, label="SARSA(λ=0.8)")
ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Meltdown Rate (%)", fontsize=12)
ax.set_title("Meltdown Rate: λ=0 vs λ=0.8", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

fig.suptitle(r"Effect of Eligibility Trace Parameter $\lambda$ (Low Noise)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(fig_path("fig6_lambda_comparison.png"), dpi=150)
plt.close(fig)


# =============================================================================
# FIGURE 7: RBF approximator vs tabular (Q4a)
# =============================================================================
print("Saving fig7_rbf_vs_tabular.png ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(eps_x, smooth(ret_sarsa_low, SMOOTH_WIN),
        color=PALETTE["tabular"], lw=2, label="Tabular SARSA(λ=0.85)")
ax.plot(eps_x, smooth(ret_rbf,       SMOOTH_WIN),
        color=PALETTE["rbf"],     lw=2, label="RBF Q-learning")
ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Smoothed Return", fontsize=12)
ax.set_title("Tabular vs RBF Approximator — Episode Return", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
im = ax.imshow(rbf_agent.Q.T, aspect="auto", origin="lower",
               cmap="RdYlGn", interpolation="nearest")
plt.colorbar(im, ax=ax, label="Q-value")
ax.set_xlabel("State bin", fontsize=12)
ax.set_ylabel("Action index", fontsize=12)
ax.set_yticks(range(len(actions)))
ax.set_yticklabels([f"a={a}" for a in actions], fontsize=9)
ax.set_title("RBF Approximator Q-Function Heatmap", fontsize=13)

fig.suptitle("Challenge Q4a: Linear Function Approximation with RBF Features",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(fig_path("fig7_rbf_vs_tabular.png"), dpi=150)
plt.close(fig)


# =============================================================================
# SUMMARY TABLE
# =============================================================================
last = 500
print(f"\n{'='*62}")
print(f"{'Agent':<35} {'Mean Return':>12} {'Meltdown %':>12}")
print(f"{'='*62}")
rows = [
    ("SARSA(λ=0.85) low noise",  ret_sarsa_low,  melt_sarsa_low),
    ("Q-learning    low noise",  ret_qlearn_low, melt_qlearn_low),
    ("SARSA(λ=0.85) high noise", ret_sarsa_high, melt_sarsa_high),
    ("Q-learning    high noise", ret_qlearn_high,melt_qlearn_high),
    ("SARSA(λ=0)    low noise",  ret_lam0,       melt_lam0),
    ("SARSA(λ=0.8)  low noise",  ret_lam08,      melt_lam08),
    ("RBF Q-learning low noise", ret_rbf,        melt_rbf),
]
for name, rets, melts in rows:
    print(f"{name:<35} {rets[-last:].mean():>12.2f} {melts[-last:].mean()*100:>11.1f}%")

print(f"\nAll 7 figures saved to: {SCRIPT_DIR}")
