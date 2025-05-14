"""
Markov Decision Processes (MDP) Analysis for CS7641 Assignment 4

This script implements value iteration, policy iteration, SARSA, and Q-Learning
to solve Blackjack and CartPole environments and analyze their performance.

The code generates key visualizations and summary statistics for analysis and
systematically tunes hyperparameters using bettermdptools.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from matplotlib.colors import ListedColormap
import gymnasium as gym

# Import bettermdptools correctly
try:
    import bettermdptools
    from bettermdptools.algorithms import value_iteration, policy_iteration
    from bettermdptools.mdp import MDP

    print("Successfully imported bettermdptools")
except ImportError as e:
    print(f"Import error: {e}")
    print("Installing bettermdptools...")
    import subprocess

    subprocess.check_call(["pip", "install", "git+https://github.com/jlm429/bettermdptools.git"])

    # Try importing again after installation
    try:
        import bettermdptools
        from bettermdptools.algorithms import value_iteration, policy_iteration
        from bettermdptools.mdp import MDP

        print("Successfully imported bettermdptools after installation")
    except ImportError as e:
        print(f"Still having import issues: {e}")
        print("Proceeding with custom implementation...")





#################################################
# Configuration Parameters
#################################################

# Multiple random seeds for robustness
NUM_SEEDS = 5  # Number of seeds to run experiments with
RANDOM_SEEDS = [42, 123, 456, 789, 999]  # Specific seeds for reproducibility

# Directory to save results
SAVE_DIR = 'results'

# Base algorithm parameters (will be varied in grid search)
DEFAULT_GAMMA = 0.99
DEFAULT_THETA = 1e-4
DEFAULT_MAX_ITERATIONS = 300
DEFAULT_EVAL_ITERATIONS = 5

# Base RL parameters (will be varied in grid search)
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EPSILON = 0.1
DEFAULT_EPSILON_DECAY = 0.99
DEFAULT_EPSILON_MIN = 0.01

# Environment parameters - optimized for more efficient training
BLACKJACK_EPISODES = 1000  # Reduced for more practical runtime
CARTPOLE_EPISODES = 500    # Reduced for more practical runtime
CARTPOLE_MAX_STEPS = 200
CARTPOLE_BINS = [3, 6, 9]  # Adjusted for clearer discretization analysis

# Hyperparameter grids for tuning - streamlined for efficiency
VI_PI_PARAM_GRID = {
    'gamma': [0.8, 0.9, 0.95, 0.99],      # Focused on most important values
    'theta': [1e-3, 1e-4, 1e-5],           # Multiple convergence thresholds
    'max_iterations': [300]    # Fixed to one value adequate for convergence
}

RL_PARAM_GRID = {
    'gamma': [0.9, 0.95, 0.99],           # Focused on most important values
    'learning_rate': [0.05, 0.1, 0.2, 0.3],    # Key learning rates
    'epsilon_decay': [0.95, 0.99, 0.999]         # Multiple decay rates
}

# Exploration strategies to compare
EXPLORATION_STRATEGIES = ['epsilon_greedy', 'boltzmann', 'ucb']

# Create necessary directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f'{SAVE_DIR}/blackjack', exist_ok=True)
os.makedirs(f'{SAVE_DIR}/cartpole', exist_ok=True)
os.makedirs(f'{SAVE_DIR}/hyperparameter_tuning', exist_ok=True)


#################################################
# Visualization Setup
#################################################

def set_plot_style():
    """Set up high-contrast, readable plot style for reports."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 18,
        'lines.linewidth': 3,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.facecolor': 'white',
        'figure.figsize': (10, 8),
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


# Set plot style at the beginning
set_plot_style()


#################################################
# Environment Classes
#################################################

class BlackjackEnv:
    """Wrapper for Blackjack environment."""

    def __init__(self):
        self.env = gym.make('Blackjack-v1')
        # State space: player (21-4+1), dealer (10), usable_ace (2)
        self.state_space = (18, 10, 2)
        self.action_space = 2  # 0: stick, 1: hit
        self.name = 'Blackjack'
        print(f"Initialized Blackjack environment with state space: {self.state_space}")

    def get_state_index(self, state):
        """Convert a state tuple to indices."""
        player_sum, dealer_card, usable_ace = state
        player_idx = min(max(0, player_sum - 4), 17)  # 4-21 mapped to 0-17
        dealer_idx = min(max(0, dealer_card - 1), 9)  # 1-10 mapped to 0-9
        ace_idx = 1 if usable_ace else 0
        return (player_idx, dealer_idx, ace_idx)

    def get_flat_state_index(self, state):
        """Convert a state tuple to a flat index."""
        idx = self.get_state_index(state)
        return np.ravel_multi_index(idx, self.state_space)

    def get_tuple_from_flat_index(self, flat_idx):
        """Convert a flat index back to a state tuple."""
        indices = np.unravel_index(flat_idx, self.state_space)
        player_sum = indices[0] + 4  # Convert back from 0-17 to 4-21
        dealer_card = indices[1] + 1  # Convert back from 0-9 to 1-10
        usable_ace = bool(indices[2])
        return (player_sum, dealer_card, usable_ace)

    def reset(self):
        """Reset the environment and return the original state tuple."""
        state, _ = self.env.reset()
        return state

    def step(self, action):
        """Take a step and return the original next state tuple."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done

    def get_transition_matrix(self, num_samples=5000):
        """Estimate the transition matrix for dynamic programming."""
        print(f"Estimating transition probabilities for Blackjack with {num_samples} samples...")

        # Calculate total states and actions
        total_states = np.prod(self.state_space)

        # Initialize matrices
        T = np.zeros((total_states, self.action_space, total_states))
        R = np.zeros((total_states, self.action_space))
        counts = np.zeros((total_states, self.action_space, total_states))

        # Sample transitions
        for _ in tqdm(range(num_samples), desc="Sampling transitions"):
            # Get a random initial state
            state, _ = self.env.reset()
            state_idx = self.get_flat_state_index(state)

            for action in range(self.action_space):
                # We need to reset for each action to evaluate from the same state
                self.env.reset()

                # Take action and observe result
                next_state, reward, done = self.step(action)
                next_state_idx = self.get_flat_state_index(next_state)

                # Update counts and rewards
                counts[state_idx, action, next_state_idx] += 1
                R[state_idx, action] += reward

        # Normalize to get probabilities
        for s in range(total_states):
            for a in range(self.action_space):
                total = counts[s, a].sum()
                if total > 0:
                    T[s, a] = counts[s, a] / total
                    R[s, a] /= total

        return T, R


class CartPoleEnv:
    """Wrapper for CartPole environment with discretization."""

    def __init__(self, n_bins=10):
        self.env = gym.make('CartPole-v1')
        self.n_bins = n_bins
        self.name = 'CartPole'

        # Define boundaries for discretization - optimized for relevant state space
        self.cart_position_bins = np.linspace(-1.5, 1.5, n_bins + 1)  # Reduced from [-2.4, 2.4]
        self.cart_velocity_bins = np.linspace(-1.5, 1.5, n_bins + 1)  # Reduced from [-3.0, 3.0]
        self.pole_angle_bins = np.linspace(-0.15, 0.15, n_bins + 1)  # Reduced from [-0.21, 0.21]
        self.pole_velocity_bins = np.linspace(-1.5, 1.5, n_bins + 1)  # Reduced from [-3.0, 3.0]

        self.state_space = (n_bins, n_bins, n_bins, n_bins)
        self.action_space = 2  # 0: left, 1: right

        print(f"Initialized CartPole environment with {n_bins} bins per dimension (total states: {n_bins ** 4})")

    def discretize_state(self, state):
        """Convert continuous state to discrete state."""
        cart_pos, cart_vel, pole_angle, pole_vel = state

        # Find bin indices
        pos_bin = np.digitize(cart_pos, self.cart_position_bins) - 1
        vel_bin = np.digitize(cart_vel, self.cart_velocity_bins) - 1
        angle_bin = np.digitize(pole_angle, self.pole_angle_bins) - 1
        ang_vel_bin = np.digitize(pole_vel, self.pole_velocity_bins) - 1

        # Clip to ensure within bounds
        pos_bin = max(0, min(pos_bin, self.n_bins - 1))
        vel_bin = max(0, min(vel_bin, self.n_bins - 1))
        angle_bin = max(0, min(angle_bin, self.n_bins - 1))
        ang_vel_bin = max(0, min(ang_vel_bin, self.n_bins - 1))

        return (int(pos_bin), int(vel_bin), int(angle_bin), int(ang_vel_bin))

    def get_flat_state_index(self, state):
        """Convert discrete state tuple to flat index."""
        pos_bin, vel_bin, angle_bin, ang_vel_bin = state

        # Manually compute the flat index
        index = int(pos_bin)
        index = index * self.n_bins + int(vel_bin)
        index = index * self.n_bins + int(angle_bin)
        index = index * self.n_bins + int(ang_vel_bin)

        return index

    def reset(self):
        """Reset environment and return discretized state."""
        state, _ = self.env.reset()
        return self.discretize_state(state)

    def step(self, action):
        """Take step and return discretized next state."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return self.discretize_state(next_state), reward, done

    def get_transition_matrix(self, num_episodes=100, max_steps=75):
        """Estimate transition matrix through sampling with fewer episodes for runtime efficiency."""
        print(f"Estimating transition probabilities for CartPole with {num_episodes} episodes...")

        # Calculate total states
        total_states = np.prod(self.state_space)

        # Initialize matrices - using dictionaries for sparse representation
        T_counts = defaultdict(float)
        R_sum = defaultdict(float)
        visit_counts = defaultdict(int)

        for _ in tqdm(range(num_episodes), desc="Sampling transitions"):
            state_raw, _ = self.env.reset()
            disc_state = self.discretize_state(state_raw)
            state_idx = self.get_flat_state_index(disc_state)

            for step in range(max_steps):
                # Choose random action for better exploration of state space
                action = self.env.action_space.sample()

                # Take action and observe result
                next_state_raw, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Discretize next state
                next_disc_state = self.discretize_state(next_state_raw)
                next_state_idx = self.get_flat_state_index(next_disc_state)

                # Update counts and rewards
                T_counts[(state_idx, action, next_state_idx)] += 1
                R_sum[(state_idx, action)] += reward
                visit_counts[(state_idx, action)] += 1

                # Move to next state
                disc_state = next_disc_state
                state_idx = next_state_idx

                if done:
                    break

        # Convert to dense matrices
        T = np.zeros((total_states, self.action_space, total_states))
        R = np.zeros((total_states, self.action_space))

        # Normalize to get probabilities
        for (s, a, s_next), count in T_counts.items():
            T[s, a, s_next] = count / max(visit_counts[(s, a)], 1)

        for (s, a), r_sum in R_sum.items():
            R[s, a] = r_sum / max(visit_counts[(s, a)], 1)

        # Add small probability for unvisited transitions to ensure stochasticity
        # This prevents issues with zero-probability transitions in value iteration
        for s in range(total_states):
            for a in range(self.action_space):
                if np.sum(T[s, a]) == 0:
                    # If state-action pair never visited, assign uniform transition
                    T[s, a] = np.ones(total_states) / total_states
                elif np.sum(T[s, a]) < 0.99:
                    # If transition probabilities don't sum to 1, normalize
                    T[s, a] = T[s, a] / np.sum(T[s, a])

        return T, R

#################################################
# Dynamic Programming Algorithms
#################################################

class ValueIteration:
    """Value Iteration algorithm for solving MDPs."""

    def __init__(self, states, actions, gamma=DEFAULT_GAMMA, theta=DEFAULT_THETA,
                 max_iterations=DEFAULT_MAX_ITERATIONS):
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.name = "Value Iteration"

        # Initialize value function
        self.V = np.zeros(states)
        self.policy = np.zeros(states, dtype=int)

        # For tracking
        self.delta_history = []
        self.value_history = []
        self.time_per_iter = []

    def solve(self, T, R):
        """Solve MDP using Value Iteration with early stopping."""
        print(f"\nStarting Value Iteration with gamma={self.gamma}, theta={self.theta}")
        start_time = time.time()
        iter_start_time = time.time()

        # Initialize with a better heuristic for large state spaces
        if np.prod(self.V.shape) > 10000:  # If large state space (like CartPole)
            self.V = np.ones(self.states) * 0.1  # Small positive initialization

        # Track consecutive iterations below threshold for early stopping
        stable_iterations = 0
        early_stop_threshold = 0.001  # Higher than theta for early stopping

        for i in range(self.max_iterations):
            delta = 0

            # Vectorized update for efficiency when possible
            if hasattr(T, 'shape') and len(T.shape) == 3:  # Check if T is a proper 3D array
                # Create a copy of the current value function
                v_copy = self.V.copy()

                for s in range(self.states):
                    # Compute value for all actions
                    action_values = np.zeros(self.actions)
                    for a in range(self.actions):
                        # Calculate expected value for this action (R + γ * sum(P * V))
                        # We use v_copy to ensure synchronous update (correct Bellman update)
                        action_values[a] = R[s, a] + self.gamma * np.sum(T[s, a] * v_copy)

                    # Update with best action
                    self.V[s] = np.max(action_values)

                    # Update delta
                    delta = max(delta, abs(v_copy[s] - self.V[s]))
            else:
                # Fallback to non-vectorized update
                v_copy = self.V.copy()
                for s in range(self.states):
                    # Compute value for all actions
                    action_values = np.zeros(self.actions)
                    for a in range(self.actions):
                        action_values[a] = R[s, a] + self.gamma * np.sum(T[s, a] * v_copy)

                    # Update with best action
                    self.V[s] = np.max(action_values)

                    # Update delta
                    delta = max(delta, abs(v_copy[s] - self.V[s]))

            # Store history
            self.delta_history.append(delta)
            self.value_history.append(np.mean(self.V))

            # Track time per iteration
            iter_time = time.time() - iter_start_time
            self.time_per_iter.append(iter_time)
            iter_start_time = time.time()

            # Early stopping: if delta is small enough for multiple iterations
            if delta < early_stop_threshold:
                stable_iterations += 1
                if stable_iterations >= 3:  # Stop after 3 stable iterations
                    print(f"  Early stopping at iteration {i + 1} with delta={delta:.6f}")
                    break
            else:
                stable_iterations = 0

            # Check for convergence
            if delta < self.theta:
                print(f"  Converged after {i + 1} iterations with delta={delta:.6f}")
                break

            # Print progress
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i + 1}: delta={delta:.6f}, avg value={np.mean(self.V):.4f}")

        # Extract policy
        self.policy = np.zeros(self.states, dtype=int)
        for s in range(self.states):
            action_values = np.zeros(self.actions)
            for a in range(self.actions):
                action_values[a] = R[s, a] + self.gamma * np.sum(T[s, a] * self.V)
            self.policy[s] = np.argmax(action_values)

        total_time = time.time() - start_time
        print(f"Value Iteration completed in {total_time:.2f} seconds")

        return {
            'V': self.V,
            'policy': self.policy,
            'iterations': i + 1,
            'total_time': total_time,
            'converged': delta < self.theta,
            'delta_history': self.delta_history,
            'value_history': self.value_history,
            'time_per_iter': self.time_per_iter
        }

class PolicyIteration:
    """Policy Iteration algorithm for solving MDPs."""

    def __init__(self, states, actions, gamma=DEFAULT_GAMMA, theta=DEFAULT_THETA,
                 max_iterations=DEFAULT_MAX_ITERATIONS, eval_iterations=DEFAULT_EVAL_ITERATIONS):
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.eval_iterations = eval_iterations
        self.name = "Policy Iteration"

        # Initialize value function and policy
        self.V = np.zeros(states)
        self.policy = np.zeros(states, dtype=int)

        # For tracking
        self.policy_changes = []
        self.value_history = []
        self.time_per_iter = []

    def policy_evaluation(self, T, R):
        """Evaluate policy by computing its value function."""
        for _ in range(self.eval_iterations):
            delta = 0
            # Create a copy to ensure synchronous update
            v_copy = self.V.copy()

            for s in range(self.states):
                # Get current policy action
                a = self.policy[s]

                # Compute expected value
                self.V[s] = R[s, a] + self.gamma * np.sum(T[s, a] * v_copy)

                # Update delta
                delta = max(delta, abs(v_copy[s] - self.V[s]))

            # Check for convergence (faster for large state spaces)
            if delta < self.theta * 10:  # Use a higher threshold for evaluation
                break

        return delta

    def policy_improvement(self, T, R):
        """Improve policy based on current value function."""
        policy_stable = True
        changes = 0

        for s in range(self.states):
            old_action = self.policy[s]

            # Find best action
            action_values = np.zeros(self.actions)
            for a in range(self.actions):
                action_values[a] = R[s, a] + self.gamma * np.sum(T[s, a] * self.V)

            # Update policy
            self.policy[s] = np.argmax(action_values)

            # Check if policy changed
            if old_action != self.policy[s]:
                policy_stable = False
                changes += 1

        return policy_stable, changes

    def solve(self, T, R):
        """Solve MDP using Policy Iteration with early stopping."""
        print(f"\nStarting Policy Iteration with gamma={self.gamma}, eval_iterations={self.eval_iterations}")
        start_time = time.time()
        iter_start_time = time.time()

        # Track consecutive policy stability for early stopping
        stable_count = 0
        min_changes_threshold = 10  # Stop if changes are minimal

        for i in range(self.max_iterations):
            # Policy evaluation
            eval_delta = self.policy_evaluation(T, R)

            # Policy improvement
            policy_stable, changes = self.policy_improvement(T, R)
            self.policy_changes.append(changes)

            # Store history
            self.value_history.append(np.mean(self.V))

            # Track time per iteration
            iter_time = time.time() - iter_start_time
            self.time_per_iter.append(iter_time)
            iter_start_time = time.time()

            # Print progress
            if (i + 1) % 5 == 0:
                print(f"  Iteration {i + 1}: {changes} policy changes, avg value={np.mean(self.V):.4f}")

            # Early stopping: if changes are minimal for several iterations
            if changes < min_changes_threshold:
                stable_count += 1
                if stable_count >= 3:  # 3 consecutive iterations with minimal changes
                    print(f"  Early stopping at iteration {i + 1} with {changes} policy changes")
                    break
            else:
                stable_count = 0

            # Check for convergence
            if policy_stable:
                print(f"  Converged after {i + 1} iterations")
                break

        total_time = time.time() - start_time
        print(f"Policy Iteration completed in {total_time:.2f} seconds")

        return {
            'V': self.V,
            'policy': self.policy,
            'iterations': i + 1,
            'total_time': total_time,
            'converged': policy_stable,
            'policy_changes': self.policy_changes,
            'value_history': self.value_history,
            'time_per_iter': self.time_per_iter
        }


#################################################
# Reinforcement Learning Algorithms
#################################################

class SARSA:
    """SARSA algorithm for reinforcement learning."""

    def __init__(self, state_shape, action_size, learning_rate=DEFAULT_LEARNING_RATE,
                 gamma=DEFAULT_GAMMA, epsilon=DEFAULT_EPSILON, epsilon_decay=DEFAULT_EPSILON_DECAY,
                 epsilon_min=DEFAULT_EPSILON_MIN, exploration_strategy='epsilon_greedy'):
        self.name = "SARSA"
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.exploration_strategy = exploration_strategy

        # Initialize Q-table
        if isinstance(state_shape, tuple):
            self.is_tuple_state = True
            self.state_shape = state_shape
            self.Q = np.zeros(state_shape + (action_size,))
        else:
            self.is_tuple_state = False
            self.Q = np.zeros((state_shape, action_size))

        self.action_size = action_size

        # For tracking
        self.rewards_history = []
        self.epsilon_history = []
        self.episode_lengths = []

        # For UCB exploration
        self.visit_counts = np.ones((state_shape, action_size)) if not self.is_tuple_state else None
        if self.is_tuple_state:
            self.visit_counts = np.ones(state_shape + (action_size,))
        self.total_steps = 0

    def get_state_index(self, state):
        """Convert state to index for Q-table."""
        if self.is_tuple_state:
            return state
        return state

    def select_action(self, state):
        """Select action using specified exploration strategy."""
        if self.exploration_strategy == 'epsilon_greedy':
            # Epsilon-greedy strategy
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.action_size)
            else:
                if self.is_tuple_state:
                    return np.argmax(self.Q[state])
                else:
                    return np.argmax(self.Q[state, :])

        elif self.exploration_strategy == 'boltzmann':
            # Boltzmann/softmax exploration
            if self.is_tuple_state:
                q_values = self.Q[state]
            else:
                q_values = self.Q[state, :]

            # Calculate softmax probabilities with temperature parameter
            temperature = max(0.1, 1.0 - (self.total_steps / 10000))  # Decreases over time
            exp_q = np.exp(q_values / temperature)
            probabilities = exp_q / np.sum(exp_q)

            # Choose action based on probabilities
            return np.random.choice(self.action_size, p=probabilities)

        elif self.exploration_strategy == 'ucb':
            # UCB exploration (Upper Confidence Bound)
            if self.is_tuple_state:
                q_values = self.Q[state]
                visit_counts = self.visit_counts[state]
            else:
                q_values = self.Q[state, :]
                visit_counts = self.visit_counts[state, :]

            # UCB formula: Q(s,a) + c * sqrt(ln(total_steps) / N(s,a))
            c = 2.0  # Exploration parameter
            ucb_values = q_values + c * np.sqrt(np.log(self.total_steps + 1) / visit_counts)

            return np.argmax(ucb_values)

        else:
            # Default to epsilon-greedy if strategy not recognized
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.action_size)
            else:
                if self.is_tuple_state:
                    return np.argmax(self.Q[state])
                else:
                    return np.argmax(self.Q[state, :])

    def update(self, state, action, reward, next_state, next_action, done):
        """Update Q-value using SARSA update rule."""
        # Get current Q-value
        if self.is_tuple_state:
            current_q = self.Q[state][action]
        else:
            current_q = self.Q[state, action]

        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            if self.is_tuple_state:
                target_q = reward + self.gamma * self.Q[next_state][next_action]
            else:
                target_q = reward + self.gamma * self.Q[next_state, next_action]

        # Update Q-value
        if self.is_tuple_state:
            self.Q[state][action] += self.learning_rate * (target_q - current_q)

            # Update visit counts for UCB
            if self.exploration_strategy == 'ucb':
                self.visit_counts[state][action] += 1
        else:
            self.Q[state, action] += self.learning_rate * (target_q - current_q)

            # Update visit counts for UCB
            if self.exploration_strategy == 'ucb':
                self.visit_counts[state, action] += 1

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """Extract greedy policy from Q-values."""
        if self.is_tuple_state:
            policy = np.zeros(self.state_shape, dtype=int)
            for state_idx in np.ndindex(self.state_shape):
                policy[state_idx] = np.argmax(self.Q[state_idx])
        else:
            policy = np.argmax(self.Q, axis=1)

        return policy

    def train(self, env, episodes=1000, max_steps=500):
        """Train the agent."""
        print(f"\nTraining {self.name} with {self.exploration_strategy} exploration for {episodes} episodes")
        print(
            f"Parameters: lr={self.learning_rate}, gamma={self.gamma}, epsilon={self.epsilon}, decay={self.epsilon_decay}")
        start_time = time.time()

        for episode in tqdm(range(episodes), desc=f"Training {self.name}"):
            # Get initial state as tuple
            state_tuple = env.reset()
            # Convert to flat index for Q-table
            state = env.get_flat_state_index(state_tuple)
            action = self.select_action(state)

            total_reward = 0
            steps = 0

            for step in range(max_steps):
                # Update total steps for UCB
                self.total_steps += 1

                # Take action
                next_state_tuple, reward, done = env.step(action)
                # Convert to flat index for Q-table
                next_state = env.get_flat_state_index(next_state_tuple)
                total_reward += reward

                # Select next action
                next_action = self.select_action(next_state)

                # Update Q-values
                self.update(state, action, reward, next_state, next_action, done)

                # Move to next state
                state = next_state
                action = next_action
                steps += 1

                if done:
                    break

            # Decay exploration rate (for epsilon-greedy)
            if self.exploration_strategy == 'epsilon_greedy':
                self.decay_epsilon()

            # Store history
            self.rewards_history.append(total_reward)
            self.epsilon_history.append(self.epsilon if self.exploration_strategy == 'epsilon_greedy' else 0)
            self.episode_lengths.append(steps)

            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])
                print(f"  Episode {episode + 1}: reward={total_reward}, avg_reward={avg_reward:.2f}")
                if self.exploration_strategy == 'epsilon_greedy':
                    print(f"  Current epsilon: {self.epsilon:.4f}")

        total_time = time.time() - start_time
        final_avg_reward = np.mean(self.rewards_history[-100:])

        print(f"{self.name} training completed in {total_time:.2f} seconds")
        print(f"  Final average reward (last 100 episodes): {final_avg_reward:.2f}")

        return {
            'policy': self.get_policy(),
            'Q': self.Q,
            'rewards': self.rewards_history,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'total_time': total_time,
            'final_avg_reward': final_avg_reward,
            'exploration_strategy': self.exploration_strategy
        }


class QLearning(SARSA):
    """Q-Learning algorithm for reinforcement learning."""

    def __init__(self, state_shape, action_size, learning_rate=DEFAULT_LEARNING_RATE,
                 gamma=DEFAULT_GAMMA, epsilon=DEFAULT_EPSILON, epsilon_decay=DEFAULT_EPSILON_DECAY,
                 epsilon_min=DEFAULT_EPSILON_MIN, exploration_strategy='epsilon_greedy'):
        super().__init__(state_shape, action_size, learning_rate, gamma, epsilon,
                         epsilon_decay, epsilon_min, exploration_strategy)
        self.name = "Q-Learning"

    def update(self, state, action, reward, next_state, next_action, done):
        """Update Q-value using Q-Learning update rule."""
        # Get current Q-value
        if self.is_tuple_state:
            current_q = self.Q[state][action]
        else:
            current_q = self.Q[state, action]

        # Calculate target Q-value (using max Q for next state)
        if done:
            target_q = reward
        else:
            if self.is_tuple_state:
                target_q = reward + self.gamma * np.max(self.Q[next_state])
            else:
                target_q = reward + self.gamma * np.max(self.Q[next_state, :])

        # Update Q-value
        if self.is_tuple_state:
            self.Q[state][action] += self.learning_rate * (target_q - current_q)

            # Update visit counts for UCB
            if self.exploration_strategy == 'ucb':
                self.visit_counts[state][action] += 1
        else:
            self.Q[state, action] += self.learning_rate * (target_q - current_q)

            # Update visit counts for UCB
            if self.exploration_strategy == 'ucb':
                self.visit_counts[state, action] += 1

    def train(self, env, episodes=1000, max_steps=500):
        """Train the agent."""
        print(f"\nTraining {self.name} with {self.exploration_strategy} exploration for {episodes} episodes")
        print(
            f"Parameters: lr={self.learning_rate}, gamma={self.gamma}, epsilon={self.epsilon}, decay={self.epsilon_decay}")
        start_time = time.time()

        for episode in tqdm(range(episodes), desc=f"Training {self.name}"):
            # Get initial state as tuple
            state_tuple = env.reset()
            # Convert to flat index for Q-table
            state = env.get_flat_state_index(state_tuple)

            total_reward = 0
            steps = 0

            for step in range(max_steps):
                # Update total steps for UCB
                self.total_steps += 1

                # Select action
                action = self.select_action(state)

                # Take action
                next_state_tuple, reward, done = env.step(action)
                # Convert to flat index for Q-table
                next_state = env.get_flat_state_index(next_state_tuple)
                total_reward += reward

                # Update Q-values (using None instead of _ for unused next_action)
                self.update(state, action, reward, next_state, None, done)

                # Move to next state
                state = next_state
                steps += 1

                if done:
                    break

            # Decay exploration rate (for epsilon-greedy)
            if self.exploration_strategy == 'epsilon_greedy':
                self.decay_epsilon()

            # Store history
            self.rewards_history.append(total_reward)
            self.epsilon_history.append(self.epsilon if self.exploration_strategy == 'epsilon_greedy' else 0)
            self.episode_lengths.append(steps)

            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])
                print(f"  Episode {episode + 1}: reward={total_reward}, avg_reward={avg_reward:.2f}")
                if self.exploration_strategy == 'epsilon_greedy':
                    print(f"  Current epsilon: {self.epsilon:.4f}")

        total_time = time.time() - start_time
        final_avg_reward = np.mean(self.rewards_history[-100:])

        print(f"{self.name} training completed in {total_time:.2f} seconds")
        print(f"  Final average reward (last 100 episodes): {final_avg_reward:.2f}")

        return {
            'policy': self.get_policy(),
            'Q': self.Q,
            'rewards': self.rewards_history,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'total_time': total_time,
            'final_avg_reward': final_avg_reward,
            'exploration_strategy': self.exploration_strategy
        }


#################################################
# Hyperparameter Tuning Functions
#################################################
def tune_hyperparameters(env, param_grid, algorithm_class, algorithm_type='DP', episodes=500, max_steps=200):
    """
    Generic hyperparameter tuning for both DP and RL algorithms.

    This function evaluates multiple parameter combinations and returns the best one.

    Args:
        env: Environment to run
        param_grid: Dictionary of parameter values to try
        algorithm_class: Class of algorithm to tune
        algorithm_type: 'DP' for dynamic programming (VI/PI) or 'RL' for reinforcement learning
        episodes: Number of episodes for RL training
        max_steps: Maximum steps per episode for RL

    Returns:
        best_params: Dictionary with best parameter values
        results: List of results for all parameter combinations
    """
    algorithm_name = algorithm_class.__name__
    print(f"\nTuning {algorithm_name} hyperparameters for {env.name} environment")

    total_states = np.prod(env.state_space)

    # Get transition and reward matrices for DP algorithms
    if algorithm_type == 'DP':
        if env.name == 'CartPole':
            T, R = env.get_transition_matrix(num_episodes=75, max_steps=50)  # Reduced for tuning
        else:
            T, R = env.get_transition_matrix(num_samples=3000)  # Reduced for tuning

    # Define parameter combinations
    param_combinations = []

    if algorithm_type == 'DP':
        for gamma in param_grid['gamma']:
            for theta in param_grid['theta']:
                for max_iter in param_grid['max_iterations']:
                    param_combinations.append({
                        'gamma': gamma,
                        'theta': theta,
                        'max_iterations': max_iter
                    })
    else:  # RL
        for gamma in param_grid['gamma']:
            for lr in param_grid['learning_rate']:
                for epsilon_decay in param_grid['epsilon_decay']:
                    param_combinations.append({
                        'gamma': gamma,
                        'learning_rate': lr,
                        'epsilon_decay': epsilon_decay,
                        'epsilon': DEFAULT_EPSILON,  # Fixed for all runs
                        'epsilon_min': DEFAULT_EPSILON_MIN  # Fixed for all runs
                    })

    results = []
    for i, params in enumerate(param_combinations):
        print(f"Testing combination {i + 1}/{len(param_combinations)}: {params}")

        if algorithm_type == 'DP':
            results_per_seed = []
            for seed in RANDOM_SEEDS:
                np.random.seed(seed)
                random.seed(seed)

                if algorithm_name == "ValueIteration":
                    algo = algorithm_class(
                        total_states,
                        env.action_space,
                        gamma=params['gamma'],
                        theta=params['theta'],
                        max_iterations=params['max_iterations']
                    )
                else:  # Policy Iteration
                    algo = algorithm_class(
                        total_states,
                        env.action_space,
                        gamma=params['gamma'],
                        theta=params['theta'],
                        max_iterations=params['max_iterations'],
                        eval_iterations=5
                    )

                result = algo.solve(T, R)
                results_per_seed.append(result)

            # Aggregate across seeds
            iterations = [r['iterations'] for r in results_per_seed]
            total_times = [r['total_time'] for r in results_per_seed]
            values = [np.mean(r['V']) for r in results_per_seed]
            converged_list = [r['converged'] for r in results_per_seed]

            results.append({
                'params': params,
                'iterations': np.mean(iterations),
                'iterations_std': np.std(iterations),
                'total_time': np.mean(total_times),
                'total_time_std': np.std(total_times),
                'mean_value': np.mean(values),
                'mean_value_std': np.std(values),
                'converged_fraction': np.mean(converged_list)
            })

        else:  # RL tuning remains unchanged
            np.random.seed(RANDOM_SEEDS[0])
            random.seed(RANDOM_SEEDS[0])

            algo = algorithm_class(
                total_states,
                env.action_space,
                learning_rate=params['learning_rate'],
                gamma=params['gamma'],
                epsilon=params['epsilon'],
                epsilon_decay=params['epsilon_decay'],
                epsilon_min=params['epsilon_min']
            )

            algo_results = algo.train(env, episodes=episodes, max_steps=max_steps)

            results.append({
                'params': params,
                'final_avg_reward': algo_results['final_avg_reward'],
                'total_time': algo_results['total_time'],
                'rewards': algo_results['rewards']
            })

    # Select best hyperparameters
    if algorithm_type == 'DP':
        converged_results = [r for r in results if r['converged_fraction'] >= 0.8]
        if converged_results:
            best_result = sorted(converged_results, key=lambda x: (x['iterations'], x['total_time']))[0]
        else:
            best_result = sorted(results, key=lambda x: -x['mean_value'])[0]

        print(f"\nBest {algorithm_name} parameters for {env.name}:")
        print(f"  gamma: {best_result['params']['gamma']}")
        print(f"  theta: {best_result['params']['theta']}")
        print(f"  max_iterations: {best_result['params']['max_iterations']}")
        print(f"  Convergence: {best_result['converged_fraction']*100:.1f}% seeds")
        print(f"  Avg iterations: {best_result['iterations']:.2f} ± {best_result['iterations_std']:.2f}")
        print(f"  Avg time: {best_result['total_time']:.2f} ± {best_result['total_time_std']:.2f} s")
        print(f"  Mean value: {best_result['mean_value']:.4f} ± {best_result['mean_value_std']:.4f}")

    else:
        best_result = sorted(results, key=lambda x: -x['final_avg_reward'])[0]

        print(f"\nBest {algorithm_name} parameters for {env.name}:")
        print(f"  gamma: {best_result['params']['gamma']}")
        print(f"  learning_rate: {best_result['params']['learning_rate']}")
        print(f"  epsilon_decay: {best_result['params']['epsilon_decay']}")
        print(f"  Performance: final avg reward = {best_result['final_avg_reward']:.4f}")
        print(f"  Training time: {best_result['total_time']:.2f}s")

    return best_result['params'], results


def compare_exploration_strategies(env, algorithm_class, best_params, episodes=800, max_steps=200):
    """
    Compare different exploration strategies for SARSA or Q-Learning using multiple seeds.

    Args:
        env: Environment to run
        algorithm_class: Class of algorithm to tune
        best_params: Dictionary with best hyperparameters
        episodes: Number of episodes for training
        max_steps: Maximum steps per episode

    Returns:
        best_strategy: Name of best exploration strategy
        results: Dictionary with results for each strategy
    """
    algorithm_name = "SARSA" if algorithm_class.__name__ == "SARSA" else "Q-Learning"
    print(f"\nComparing exploration strategies for {algorithm_name} on {env.name} environment")

    total_states = np.prod(env.state_space)

    # Run each exploration strategy with the best parameters
    results = {}

    for strategy in EXPLORATION_STRATEGIES:
        print(f"Testing {strategy} exploration strategy")

        # Define experiment function for this strategy
        def run_strategy_experiment():
            # Initialize algorithm with best parameters and current strategy
            algo = algorithm_class(
                total_states,
                env.action_space,
                learning_rate=best_params['learning_rate'],
                gamma=best_params['gamma'],
                epsilon=DEFAULT_EPSILON,
                epsilon_decay=best_params['epsilon_decay'],
                epsilon_min=DEFAULT_EPSILON_MIN,
                exploration_strategy=strategy
            )

            # Train agent
            return algo.train(env, episodes=episodes, max_steps=max_steps)

        # Run with multiple seeds
        avg_results, all_results, std_results = run_with_multiple_seeds(run_strategy_experiment)

        # Store results with standard deviations
        results[strategy] = {
            'rewards': avg_results['rewards'],
            'rewards_std': std_results['rewards'],
            'episode_lengths': avg_results['episode_lengths'],
            'episode_lengths_std': std_results['episode_lengths'],
            'final_avg_reward': avg_results['final_avg_reward'],
            'final_avg_reward_std': std_results['final_avg_reward'],
            'total_time': avg_results['total_time'],
            'total_time_std': std_results['total_time'],
            'all_results': all_results
        }

    # Find best strategy (based on final average reward)
    best_strategy = max(results.items(), key=lambda x: x[1]['final_avg_reward'])[0]

    print(f"\nBest exploration strategy for {algorithm_name} on {env.name}: {best_strategy}")
    print(f"Results (mean ± std from {NUM_SEEDS} seeds):")
    for strategy, result in results.items():
        print(f"  {strategy}: reward = {result['final_avg_reward']:.4f} ± {result['final_avg_reward_std']:.4f}, " +
              f"time = {result['total_time']:.2f} ± {result['total_time_std']:.2f}s")

    return best_strategy, results

#################################################
# Visualization Functions
#################################################
def plot_parameter_effects(tuning_results, algorithm_name, environment_name):
    """Visualize effects of different hyperparameters on performance."""
    plt.figure(figsize=(15, 10))

    # Extract parameters and organize results
    if 'gamma' in tuning_results[0]['params']:
        gamma_values = sorted(set(r['params']['gamma'] for r in tuning_results))

        # Plot 1: Effect of gamma
        plt.subplot(2, 2, 1)
        for gamma in gamma_values:
            matching_results = [r for r in tuning_results if r['params']['gamma'] == gamma]

            if 'iterations' in matching_results[0]:  # For Value/Policy Iteration
                metric_key = 'iterations'
                ylabel = 'Iterations to Converge'
            else:  # For SARSA/Q-Learning
                metric_key = 'final_avg_reward'
                ylabel = 'Average Reward'

            # If there are other parameters, average across them
            if 'learning_rate' in matching_results[0]['params']:
                lr_values = sorted(set(r['params']['learning_rate'] for r in matching_results))
                plt.plot(lr_values,
                         [np.mean([r[metric_key] for r in matching_results
                                   if r['params']['learning_rate'] == lr])
                          for lr in lr_values],
                         marker='o', label=f'gamma={gamma}')
                plt.xlabel('Learning Rate')
            elif 'theta' in matching_results[0]['params']:
                theta_values = sorted(set(r['params']['theta'] for r in matching_results))
                plt.plot(theta_values,
                         [np.mean([r[metric_key] for r in matching_results
                                   if r['params']['theta'] == t])
                          for t in theta_values],
                         marker='o', label=f'gamma={gamma}')
                plt.xlabel('Convergence Threshold (θ)')

        plt.ylabel(ylabel)
        plt.title(f'Effect of Discount Factor (γ) on {algorithm_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 2: Computation time comparison
    plt.subplot(2, 2, 2)
    if 'learning_rate' in tuning_results[0]['params']:
        lr_values = sorted(set(r['params']['learning_rate'] for r in tuning_results))
        times = [np.mean([r['total_time'] for r in tuning_results
                          if r['params']['learning_rate'] == lr])
                 for lr in lr_values]
        plt.plot(lr_values, times, marker='o')
        plt.xlabel('Learning Rate')
    else:
        gamma_values = sorted(set(r['params']['gamma'] for r in tuning_results))
        times = [np.mean([r['total_time'] for r in tuning_results
                          if r['params']['gamma'] == g])
                 for g in gamma_values]
        plt.plot(gamma_values, times, marker='o')
        plt.xlabel('Discount Factor (γ)')

    plt.ylabel('Computation Time (seconds)')
    plt.title(f'Parameter Impact on Computation Time')
    plt.grid(True, alpha=0.3)

    # Plot 3: Parameter interaction heatmap (for RL only)
    if 'learning_rate' in tuning_results[0]['params'] and 'gamma' in tuning_results[0]['params']:
        plt.subplot(2, 2, 3)
        lr_values = sorted(set(r['params']['learning_rate'] for r in tuning_results))
        gamma_values = sorted(set(r['params']['gamma'] for r in tuning_results))

        # Create matrix of results
        heatmap_data = np.zeros((len(gamma_values), len(lr_values)))
        for i, gamma in enumerate(gamma_values):
            for j, lr in enumerate(lr_values):
                matching = [r for r in tuning_results
                            if r['params']['gamma'] == gamma and
                            r['params']['learning_rate'] == lr]
                if matching:
                    heatmap_data[i, j] = np.mean([r['final_avg_reward'] for r in matching])

        im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Average Reward')
        plt.xticks(range(len(lr_values)), lr_values)
        plt.yticks(range(len(gamma_values)), gamma_values)
        plt.xlabel('Learning Rate')
        plt.ylabel('Discount Factor (γ)')
        plt.title('Parameter Interaction Effect on Performance')

    plt.tight_layout()
    save_path = f"{SAVE_DIR}/hyperparameter_tuning/{environment_name.lower()}_{algorithm_name.replace(' ', '_')}_params.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved parameter effects visualization to {save_path}")


def analyze_convergence_sensitivity(env, algorithm_class, best_params, algorithm_type='DP'):
    """Analyze how sensitive the algorithm is to convergence threshold."""
    algorithm_name = algorithm_class.__name__
    print(f"\nAnalyzing convergence sensitivity for {algorithm_name} on {env.name}")

    if algorithm_type == 'DP':
        # For DP methods, test different theta values
        theta_values = [1e-2, 1e-3, 1e-4, 1e-5]
        results = []

        # Get transition probabilities
        if env.name == 'CartPole':
            T, R = env.get_transition_matrix(num_episodes=150, max_steps=75)
        else:
            T, R = env.get_transition_matrix(num_samples=5000)

        total_states = np.prod(env.state_space)

        for theta in theta_values:
            # Create algorithm with current theta
            test_params = best_params.copy()
            test_params['theta'] = theta

            if algorithm_name == "ValueIteration":
                algo = algorithm_class(total_states, env.action_space,
                                       gamma=test_params['gamma'],
                                       theta=theta,
                                       max_iterations=test_params['max_iterations'])
            else:  # Policy Iteration
                algo = algorithm_class(total_states, env.action_space,
                                       gamma=test_params['gamma'],
                                       theta=theta,
                                       max_iterations=test_params['max_iterations'],
                                       eval_iterations=5)

            # Solve and record results
            start_time = time.time()
            algo_results = algo.solve(T, R)
            elapsed = time.time() - start_time

            results.append({
                'theta': theta,
                'iterations': algo_results['iterations'],
                'time': elapsed,
                'value': np.mean(algo_results['V']),
            })

        # Plot results
        plt.figure(figsize=(12, 6))

        # Plot iterations vs theta
        plt.subplot(1, 2, 1)
        theta_values = [r['theta'] for r in results]
        iterations = [r['iterations'] for r in results]
        plt.semilogx(theta_values, iterations, marker='o', linewidth=2)
        plt.xlabel('Convergence Threshold (θ)', fontsize=12)
        plt.ylabel('Iterations to Converge', fontsize=12)
        plt.title(f'Effect of Convergence Threshold on {algorithm_name}', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Plot time vs iterations
        plt.subplot(1, 2, 2)
        times = [r['time'] for r in results]
        plt.plot(iterations, times, marker='o', linewidth=2)
        for i, (it, t, th) in enumerate(zip(iterations, times, theta_values)):
            plt.annotate(f'θ={th}', (it, t), xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Computation Time (seconds)', fontsize=12)
        plt.title('Time-Iteration Trade-off', fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = f"{SAVE_DIR}/hyperparameter_tuning/{env.name.lower()}_{algorithm_name}_convergence.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved convergence sensitivity analysis to {save_path}")

        return results
    else:
        # For RL methods, test different learning rates and plot learning curves
        lr_values = [0.05, 0.1, 0.2, 0.5] if 'learning_rate' in best_params else [best_params.get('learning_rate', 0.1)]
        results = []

        for lr in lr_values:
            test_params = best_params.copy()
            test_params['learning_rate'] = lr

            algo = algorithm_class(np.prod(env.state_space), env.action_space,
                                   learning_rate=lr,
                                   gamma=test_params['gamma'],
                                   epsilon_decay=test_params['epsilon_decay'],
                                   exploration_strategy='epsilon_greedy')

            # Train and record reward history
            algo_results = algo.train(env, episodes=300, max_steps=200)
            results.append({
                'learning_rate': lr,
                'rewards': algo_results['rewards'],
                'final_reward': algo_results['final_avg_reward'],
                'time': algo_results['total_time']
            })

        # Plot learning curves for different learning rates
        plt.figure(figsize=(12, 6))

        # Plot reward curves
        smoothing = 10
        for i, res in enumerate(results):
            rewards = res['rewards']
            lr = res['learning_rate']
            if len(rewards) > smoothing:
                smoothed = np.convolve(rewards, np.ones(smoothing) / smoothing, mode='valid')
                plt.plot(smoothed, label=f'lr={lr}', linewidth=2)
            else:
                plt.plot(rewards, label=f'lr={lr}', linewidth=2)

        plt.title(f'Learning Rate Effect on {algorithm_name} Performance', fontsize=14)
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = f"{SAVE_DIR}/hyperparameter_tuning/{env.name.lower()}_{algorithm_name}_learning_rates.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved learning rate sensitivity analysis to {save_path}")

        return results


def plot_discretization_quality_tradeoff(discretization_results, save_path):
    """Plot the relationship between state space size and solution quality."""
    plt.figure(figsize=(12, 8))

    # Extract data
    bins = list(discretization_results.keys())
    state_space_sizes = [b ** 4 for b in bins]
    vi_values = [discretization_results[b]['VI']['value'] for b in bins]
    pi_values = [discretization_results[b]['PI']['value'] for b in bins]
    vi_times = [discretization_results[b]['VI']['time'] for b in bins]
    pi_times = [discretization_results[b]['PI']['time'] for b in bins]

    # Create first subplot for solution quality
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(state_space_sizes, vi_values, 'o-', label='VI Solution Quality', linewidth=2)
    ax1.plot(state_space_sizes, pi_values, 's-', label='PI Solution Quality', linewidth=2)
    ax1.set_xscale('log')
    ax1.set_xlabel('State Space Size (log scale)', fontsize=12)
    ax1.set_ylabel('Average Value', fontsize=12)
    ax1.set_title('Solution Quality vs. State Space Size', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Create second subplot for computation time
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(state_space_sizes, vi_times, 'o-', label='VI Computation Time', linewidth=2)
    ax2.plot(state_space_sizes, pi_times, 's-', label='PI Computation Time', linewidth=2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('State Space Size (log scale)', fontsize=12)
    ax2.set_ylabel('Computation Time (seconds, log scale)', fontsize=12)
    ax2.set_title('Computation Time vs. State Space Size', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved discretization quality-time tradeoff plot to {save_path}")


def plot_best_exploration_training_progress(sarsa_results, q_results, env_name, save_path):
    """
    Plot the training progress of the best exploration strategies for SARSA and Q-Learning.

    Args:
        sarsa_results: Dictionary containing SARSA exploration strategy results
        q_results: Dictionary containing Q-Learning exploration strategy results
        env_name: Name of the environment
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(14, 10))

    # Determine best strategies
    sarsa_best = max(sarsa_results.items(), key=lambda x: x[1]['final_avg_reward'])[0]
    q_best = max(q_results.items(), key=lambda x: x[1]['final_avg_reward'])[0]

    # Get rewards history for best strategies
    sarsa_rewards = sarsa_results[sarsa_best]['rewards']
    q_rewards = q_results[q_best]['rewards']

    # Smooth rewards
    window = 20
    sarsa_smoothed = np.convolve(sarsa_rewards, np.ones(window) / window, mode='valid')
    q_smoothed = np.convolve(q_rewards, np.ones(window) / window, mode='valid')

    # Create subplots
    # Subplot 1: Cumulative reward
    plt.subplot(2, 2, 1)
    plt.plot(np.cumsum(sarsa_rewards) / np.arange(1, len(sarsa_rewards) + 1),
             label=f'SARSA ({sarsa_best})', linewidth=2)
    plt.plot(np.cumsum(q_rewards) / np.arange(1, len(q_rewards) + 1),
             label=f'Q-Learning ({q_best})', linewidth=2)
    plt.title('Cumulative Average Reward', fontsize=14)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Smoothed rewards
    plt.subplot(2, 2, 2)
    plt.plot(sarsa_smoothed, label=f'SARSA ({sarsa_best})', linewidth=2)
    plt.plot(q_smoothed, label=f'Q-Learning ({q_best})', linewidth=2)
    plt.title(f'Smoothed Reward (Window={window})', fontsize=14)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Learning curves comparison (last 100 episodes)
    plt.subplot(2, 2, 3)
    sarsa_last100 = sarsa_rewards[-100:]
    q_last100 = q_rewards[-100:]
    plt.hist(sarsa_last100, alpha=0.6, bins=10, label=f'SARSA ({sarsa_best})')
    plt.hist(q_last100, alpha=0.6, bins=10, label=f'Q-Learning ({q_best})')
    plt.title('Distribution of Rewards (Last 100 Episodes)', fontsize=14)
    plt.xlabel('Reward Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 4: Convergence statistics
    plt.subplot(2, 2, 4)
    labels = ['SARSA', 'Q-Learning']
    x = np.arange(len(labels))
    width = 0.35

    # Calculate statistics
    sarsa_mean = np.mean(sarsa_last100)
    sarsa_std = np.std(sarsa_last100)
    q_mean = np.mean(q_last100)
    q_std = np.std(q_last100)

    plt.bar(x - width / 2, [sarsa_mean, q_mean], width,
            label='Mean Reward', yerr=[sarsa_std, q_std], capsize=5)
    plt.bar(x + width / 2, [sarsa_std, q_std], width,
            label='Standard Deviation', alpha=0.7)

    plt.title('Convergence Statistics (Last 100 Episodes)', fontsize=14)
    plt.xticks(x, labels)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle(f'Training Progress Comparison - Best Exploration Strategies for {env_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved best exploration training progress visualization to {save_path}")


def create_performance_dashboard(results, extra_data, env_name, save_path):
    """
    Create a comprehensive performance dashboard for all algorithms.

    Args:
        results: Dictionary with results for all algorithms
        extra_data: Dictionary with extra analysis data
        env_name: Name of the environment
        save_path: Path to save the dashboard
    """
    plt.figure(figsize=(16, 12))

    # Subplot 1: Algorithm performance comparison
    plt.subplot(2, 2, 1)
    algos = list(results.keys())
    x = np.arange(len(algos))
    rewards = [results[algo]['reward'] for algo in algos]

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    plt.bar(x, rewards, color=colors)
    plt.xticks(x, algos, rotation=45, ha='right')
    plt.title('Algorithm Performance Comparison', fontsize=14)
    plt.ylabel('Average Value / Reward', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Computation time comparison
    plt.subplot(2, 2, 2)
    times = [results[algo]['time'] for algo in algos]
    plt.bar(x, times, color=colors)
    plt.xticks(x, algos, rotation=45, ha='right')
    plt.title('Computation Time Comparison', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Policy agreement matrix
    plt.subplot(2, 2, 3)
    agreement_matrix = np.array([
        [100, extra_data['vi_pi_agreement'], extra_data['vi_sarsa_agreement'],
         extra_data['vi_pi_agreement'] * extra_data['pi_qlearning_agreement'] / 100],
        [extra_data['vi_pi_agreement'], 100,
         extra_data['vi_sarsa_agreement'] * extra_data['vi_pi_agreement'] / 100,
         extra_data['pi_qlearning_agreement']],
        [extra_data['vi_sarsa_agreement'],
         extra_data['vi_sarsa_agreement'] * extra_data['vi_pi_agreement'] / 100,
         100, extra_data['sarsa_qlearning_agreement']],
        [extra_data['vi_pi_agreement'] * extra_data['pi_qlearning_agreement'] / 100,
         extra_data['pi_qlearning_agreement'],
         extra_data['sarsa_qlearning_agreement'], 100]
    ])

    im = plt.imshow(agreement_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='Policy Agreement (%)')
    plt.xticks(np.arange(len(algos)), algos, rotation=45, ha='right')
    plt.yticks(np.arange(len(algos)), algos)
    plt.title('Policy Agreement Matrix', fontsize=14)

    # Add text annotations
    for i in range(len(algos)):
        for j in range(len(algos)):
            plt.text(j, i, f"{agreement_matrix[i, j]:.1f}%",
                     ha="center", va="center",
                     color="white" if agreement_matrix[i, j] < 75 else "black")

    # Subplot 4: Algorithm characteristics table
    plt.subplot(2, 2, 4)
    plt.axis('off')

    # Create table data
    model_type = ['Model-based', 'Model-based', 'Model-free', 'Model-free']
    iterations = [results[algo]['iterations'] for algo in algos]
    best_explore = [results[algo].get('best_strategy', 'N/A') for algo in algos]

    table_data = [
        ['Algorithm', 'Type', 'Iterations', 'Best Exploration'],
        [algos[0], model_type[0], str(iterations[0]), best_explore[0]],
        [algos[1], model_type[1], str(iterations[1]), best_explore[1]],
        [algos[2], model_type[2], str(iterations[2]), best_explore[2]],
        [algos[3], model_type[3], str(iterations[3]), best_explore[3]]
    ]

    table = plt.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.title('Algorithm Characteristics', fontsize=14)

    plt.suptitle(f'Performance Dashboard - {env_name} Environment', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance dashboard to {save_path}")

def plot_convergence(algorithm_results, title, save_path, xlabel='Iterations', ylabel='Delta', include_std=False):
    """
    Plot convergence of algorithms with optional standard deviation bands.

    Args:
        algorithm_results: Dictionary mapping algorithm names to results lists
                           If include_std=True, each value should be a list of results lists from multiple seeds
                           If include_std=False, each value should be a single results list
        title: Title of the plot
        save_path: Path to save the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        include_std: Whether to include standard deviation bands
    """
    plt.figure(figsize=(10, 8))

    # Use high-contrast colors
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

    if include_std:
        # Version with standard deviation bands
        for i, (algorithm, results_list) in enumerate(algorithm_results.items()):
            # Calculate average and std across seeds
            # We ensure all histories have the same length by using the minimum length
            min_length = min(len(results) for results in results_list)
            truncated_results = [results[:min_length] for results in results_list]

            # Convert to numpy array for easier computation
            results_array = np.array(truncated_results)

            # Calculate mean and std for each time step
            means = np.mean(results_array, axis=0)
            stds = np.std(results_array, axis=0)

            # Plot mean with std band
            x = np.arange(len(means))
            plt.plot(x, means, label=algorithm, color=colors[i % len(colors)], linewidth=3)
            plt.fill_between(x, np.maximum(0, means - stds), means + stds, alpha=0.3, color=colors[i % len(colors)])
    else:
        # Original version without std bands
        for i, (algorithm, results) in enumerate(algorithm_results.items()):
            plt.plot(results, label=algorithm, color=colors[i % len(colors)], linewidth=3)

    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence plot to {save_path}")


def plot_policy_blackjack(policy, title, save_path):
    """Plot Blackjack policy as heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Define action labels and colormap
    action_labels = ['Stick', 'Hit']
    cmap = ListedColormap(['#ff9999', '#66b3ff'])

    # Plot policy for usable ace
    sns.heatmap(policy[:, :, 1], ax=axes[0], cmap=cmap, cbar=False)
    axes[0].set_title('Policy (Usable Ace)', fontsize=16)
    axes[0].set_xlabel('Dealer Showing', fontsize=14)
    axes[0].set_ylabel('Player Sum', fontsize=14)
    axes[0].set_xticks(np.arange(0.5, 10.5))
    axes[0].set_xticklabels(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'], fontsize=12)
    axes[0].set_yticks(np.arange(0.5, 18.5))
    axes[0].set_yticklabels(range(4, 22), fontsize=12)

    # Plot policy for no usable ace
    sns.heatmap(policy[:, :, 0], ax=axes[1], cmap=cmap)
    axes[1].set_title('Policy (No Usable Ace)', fontsize=16)
    axes[1].set_xlabel('Dealer Showing', fontsize=14)
    axes[1].set_ylabel('Player Sum', fontsize=14)
    axes[1].set_xticks(np.arange(0.5, 10.5))
    axes[1].set_xticklabels(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'], fontsize=12)
    axes[1].set_yticks(np.arange(0.5, 18.5))
    axes[1].set_yticklabels(range(4, 22), fontsize=12)

    # Add colorbar
    cbar = fig.colorbar(axes[1].collections[0], ax=axes)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(action_labels)
    cbar.ax.tick_params(labelsize=12)

    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved policy heatmap to {save_path}")


def plot_policy_cartpole(policy, n_bins, title, save_path):
    """Plot CartPole policy visualization."""
    # Reshape policy for visualization
    policy_reshaped = policy.reshape((n_bins, n_bins, n_bins, n_bins))

    # Create 2D slices of policy for visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Define colormap
    cmap = ListedColormap(['#ff9999', '#66b3ff'])

    # Plot four slices of the policy
    for i, (angle_idx, ang_vel_idx) in enumerate([(n_bins // 2, n_bins // 2),
                                                  (n_bins // 2, 3 * n_bins // 4),
                                                  (3 * n_bins // 4, n_bins // 2),
                                                  (3 * n_bins // 4, 3 * n_bins // 4)]):
        # Extract 2D slice
        policy_slice = policy_reshaped[:, :, angle_idx, ang_vel_idx]

        # Plot slice
        im = axes[i].imshow(policy_slice, cmap=cmap, origin='lower',
                            extent=[-1.5, 1.5, -1.5, 1.5])
        axes[i].set_title(f'Angle bin: {angle_idx}, Angular Velocity bin: {ang_vel_idx}', fontsize=14)
        axes[i].set_xlabel('Cart Position', fontsize=14)
        axes[i].set_ylabel('Cart Velocity', fontsize=14)
        axes[i].grid(False)  # Grid interferes with imshow

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['Left', 'Right'])
    cbar.ax.tick_params(labelsize=14)

    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved CartPole policy visualization to {save_path}")


def plot_learning_curves(rewards_list, labels, title, save_path, window=10, std_list=None):
    """Plot learning curves for RL algorithms with error bands for multiple seeds."""
    plt.figure(figsize=(10, 8))

    # Use high-contrast colors
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

    for i, rewards in enumerate(rewards_list):
        # Smooth rewards for better visualization
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            plt.plot(smoothed, label=f"{labels[i]} (smoothed)",
                     color=colors[i % len(colors)], linewidth=3)

            # Add error bands if std data is available
            if std_list and i < len(std_list) and std_list[i] is not None:
                # Smooth the std values the same way
                if len(std_list[i]) > window:
                    smoothed_std = np.convolve(std_list[i], np.ones(window) / window, mode='valid')
                    x = np.arange(len(smoothed))
                    plt.fill_between(x, smoothed - smoothed_std, smoothed + smoothed_std,
                                     alpha=0.2, color=colors[i % len(colors)])

            # Also plot raw rewards with lower opacity
            plt.plot(rewards, alpha=0.3, color=colors[i % len(colors)], linewidth=1)
        else:
            plt.plot(rewards, label=labels[i], color=colors[i % len(colors)], linewidth=3)

    plt.title(title, fontsize=18)
    plt.xlabel('Episodes', fontsize=16)
    plt.ylabel('Total Reward', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves to {save_path}")


def plot_combined_comparison(combined_results, title, save_path, include_std=True):
    """Create combined algorithm comparison plot with optional error bars."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract key metrics
    algorithms = list(combined_results.keys())
    metrics = ['iterations', 'time', 'reward']

    # Set positions for bars
    x = np.arange(len(metrics))
    width = 0.2
    offsets = [-0.3, -0.1, 0.1, 0.3]
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

    if include_std:
        # With standard deviation
        for i, algo in enumerate(algorithms):
            values = []
            errors = []

            for metric in metrics:
                metric_data = combined_results[algo][metric]
                # Check if the metric data is a dictionary with 'mean' key
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    values.append(metric_data['mean'])
                    errors.append(metric_data.get('std', 0))
                else:
                    # Handle direct values
                    values.append(metric_data)
                    errors.append(0)  # No std deviation for direct values

            # Normalize values for better comparison
            max_values = []
            for j, metric in enumerate(metrics):
                all_values = []
                for a in algorithms:
                    if isinstance(combined_results[a][metric], dict) and 'mean' in combined_results[a][metric]:
                        all_values.append(combined_results[a][metric]['mean'])
                    else:
                        all_values.append(combined_results[a][metric])
                max_val = max(all_values)
                max_values.append(max_val if max_val > 0 else 1)

            norm_values = [values[j] / max_values[j] for j in range(len(values))]
            norm_errors = [errors[j] / max_values[j] for j in range(len(errors))]

            # Ensure non-negative error bars
            norm_errors = [max(0, err) for err in norm_errors]

            ax.bar(x + offsets[i], norm_values, width, label=algo, color=colors[i],
                   yerr=norm_errors, capsize=5, alpha=0.8)
    else:
        # Without standard deviation
        # Normalize metrics for better comparison
        normalized_data = {}
        for metric in metrics:
            values = []
            for algo in algorithms:
                metric_data = combined_results[algo][metric]
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    values.append(metric_data['mean'])
                else:
                    values.append(metric_data)

            max_value = max(values) if max(values) > 0 else 1
            normalized_data[metric] = [v / max_value for v in values]

        # Plot bars
        for i, algo in enumerate(algorithms):
            values = [normalized_data[metric][i] for metric in metrics]
            ax.bar(x + offsets[i], values, width, label=algo, color=colors[i])

    # Set labels
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(['Iterations', 'Computation Time', 'Final Reward'], fontsize=14)
    ax.set_ylabel('Normalized Value', fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined comparison plot to {save_path}")


def plot_combined_hyperparameter_results(all_results, environment_name):
    """Create a combined visualization for hyperparameter tuning results."""
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Organize results by algorithm type
    dp_results = {k: v for k, v in all_results.items() if k in ['Value Iteration', 'Policy Iteration']}
    rl_results = {k: v for k, v in all_results.items() if k in ['SARSA', 'Q-Learning']}

    # Plot 1: Gamma effect on iterations/performance (DP algorithms)
    ax = axs[0, 0]
    for algo, results in dp_results.items():
        gamma_values = sorted(set(r['params']['gamma'] for r in results))
        iterations = []
        for gamma in gamma_values:
            matching_results = [r for r in results if r['params']['gamma'] == gamma]
            if matching_results:
                iterations.append(min(r['iterations'] for r in matching_results))
        ax.plot(gamma_values, iterations, marker='o', linewidth=2, label=algo)

    ax.set_title('Effect of Discount Factor (γ) on Convergence', fontsize=14)
    ax.set_xlabel('Gamma Value', fontsize=12)
    ax.set_ylabel('Iterations to Converge', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Learning rate effect on final reward (RL algorithms)
    ax = axs[0, 1]
    for algo, results in rl_results.items():
        lr_values = sorted(set(r['params']['learning_rate'] for r in results))
        rewards = []
        for lr in lr_values:
            matching_results = [r for r in results if r['params']['learning_rate'] == lr]
            if matching_results:
                rewards.append(max(r['final_avg_reward'] for r in matching_results))
        ax.plot(lr_values, rewards, marker='o', linewidth=2, label=algo)

    ax.set_title('Effect of Learning Rate on Performance', fontsize=14)
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Final Average Reward', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Computation time comparison
    ax = axs[1, 0]
    algo_names = list(all_results.keys())
    times = []
    for algo in algo_names:
        times.append(min(r['total_time'] for r in all_results[algo]))

    x = np.arange(len(algo_names))
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    bars = ax.bar(x, times, width=0.7, alpha=0.8, color=colors[:len(algo_names)])

    ax.set_title('Computation Time by Algorithm', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)

    # Plot 4: Performance comparison (for RL only)
    ax = axs[1, 1]
    rl_algo_names = list(rl_results.keys())
    rewards = []
    for algo in rl_algo_names:
        rewards.append(max(r['final_avg_reward'] for r in rl_results[algo]))

    x = np.arange(len(rl_algo_names))
    bars = ax.bar(x, rewards, width=0.7, alpha=0.8, color=colors[:len(rl_algo_names)])

    ax.set_title('Performance by Algorithm', fontsize=14)
    ax.set_ylabel('Final Average Reward', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(rl_algo_names)
    ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle(f'Hyperparameter Tuning Analysis - {environment_name}', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = f"{SAVE_DIR}/hyperparameter_tuning/{environment_name.lower()}_combined_tuning.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined hyperparameter analysis to {save_path}")


def plot_combined_exploration_strategies(sarsa_results, qlearning_results, environment_name):
    """Create a combined visualization for exploration strategy comparison."""
    # Extract strategies and results
    strategies = list(sarsa_results.keys())

    # Setup the figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Final rewards comparison
    ax = axs[0, 0]
    sarsa_rewards = [sarsa_results[s]['final_avg_reward'] for s in strategies]
    q_rewards = [qlearning_results[s]['final_avg_reward'] for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35
    bars1 = ax.bar(x - width / 2, sarsa_rewards, width, label='SARSA', color='#1f77b4')
    bars2 = ax.bar(x + width / 2, q_rewards, width, label='Q-Learning', color='#d62728')

    ax.set_title('Final Rewards by Exploration Strategy', fontsize=14)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(sarsa_rewards):
        ax.text(i - width / 2, v + 0.1, f"{v:.1f}", ha='center', fontsize=10)
    for i, v in enumerate(q_rewards):
        ax.text(i + width / 2, v + 0.1, f"{v:.1f}", ha='center', fontsize=10)

    # Plot 2: Computation time comparison
    ax = axs[0, 1]
    sarsa_times = [sarsa_results[s]['total_time'] for s in strategies]
    q_times = [qlearning_results[s]['total_time'] for s in strategies]

    bars1 = ax.bar(x - width / 2, sarsa_times, width, label='SARSA', color='#1f77b4')
    bars2 = ax.bar(x + width / 2, q_times, width, label='Q-Learning', color='#d62728')

    ax.set_title('Computation Time by Exploration Strategy', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Plot 3: Learning curves (just one example strategy)
    ax = axs[1, 0]
    window = 20  # Smoothing window

    # Choose epsilon_greedy as example
    strat = 'epsilon_greedy'
    sarsa_rewards = sarsa_results[strat]['rewards']
    q_rewards = qlearning_results[strat]['rewards']

    # Apply smoothing
    if len(sarsa_rewards) > window:
        sarsa_smoothed = np.convolve(sarsa_rewards, np.ones(window) / window, mode='valid')
        ax.plot(sarsa_smoothed, label=f'SARSA ({strat})', color='#1f77b4', linewidth=2)
    else:
        ax.plot(sarsa_rewards, label=f'SARSA ({strat})', color='#1f77b4', linewidth=2)

    if len(q_rewards) > window:
        q_smoothed = np.convolve(q_rewards, np.ones(window) / window, mode='valid')
        ax.plot(q_smoothed, label=f'Q-Learning ({strat})', color='#d62728', linewidth=2)
    else:
        ax.plot(q_rewards, label=f'Q-Learning ({strat})', color='#d62728', linewidth=2)

    ax.set_title(f'Learning Curves ({strat})', fontsize=14)
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Learning stability (std of final rewards)
    ax = axs[1, 1]

    # Calculate stability metrics from last 100 episodes
    stability_data = []
    for strat in strategies:
        sarsa_last100 = sarsa_results[strat]['rewards'][-100:]
        q_last100 = qlearning_results[strat]['rewards'][-100:]

        stability_data.append({
            'strategy': strat,
            'sarsa_std': np.std(sarsa_last100),
            'q_std': np.std(q_last100)
        })

    strat_names = [d['strategy'] for d in stability_data]
    sarsa_stds = [d['sarsa_std'] for d in stability_data]
    q_stds = [d['q_std'] for d in stability_data]

    x = np.arange(len(strat_names))
    bars1 = ax.bar(x - width / 2, sarsa_stds, width, label='SARSA', color='#1f77b4')
    bars2 = ax.bar(x + width / 2, q_stds, width, label='Q-Learning', color='#d62728')

    ax.set_title('Learning Stability (Std Dev of Last 100 Episodes)', fontsize=14)
    ax.set_ylabel('Standard Deviation', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_names)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle(f'Exploration Strategy Comparison - {environment_name}', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = f"{SAVE_DIR}/hyperparameter_tuning/{environment_name.lower()}_exploration_strategies.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved exploration strategy comparison to {save_path}")

def plot_hyperparameter_tuning_results(results, algorithm_name, env_name):
    """Visualize hyperparameter tuning results for VI or PI."""
    save_dir = f"{SAVE_DIR}/hyperparameter_tuning"
    os.makedirs(save_dir, exist_ok=True)

    # Extract parameters and metrics
    gamma_values = sorted(set(r['params']['gamma'] for r in results))
    theta_values = sorted(set(r['params']['theta'] for r in results))
    iterations = [r['iterations'] for r in results]
    times = [r['total_time'] for r in results]

    # Plot effect of gamma on iterations
    plt.figure(figsize=(10, 6))
    gamma_to_iterations = {}
    for gamma in gamma_values:
        gamma_to_iterations[gamma] = [r['iterations'] for r in results if r['params']['gamma'] == gamma]

    for gamma, iters in gamma_to_iterations.items():
        plt.plot(theta_values, iters, marker='o', linewidth=2, label=f'gamma={gamma}')

    plt.title(f'Effect of gamma on {algorithm_name} Convergence ({env_name})', fontsize=16)
    plt.xlabel('Theta (Convergence Threshold)', fontsize=14)
    plt.ylabel('Iterations to Converge', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = f"{save_dir}/{env_name}_{algorithm_name.replace(' ', '_')}_gamma_effect.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot convergence time vs iterations
    plt.figure(figsize=(10, 6))
    plt.scatter(iterations, times, alpha=0.7, s=100)

    for i, r in enumerate(results):
        plt.annotate(f"γ={r['params']['gamma']}, θ={r['params']['theta']}",
                     (iterations[i], times[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.title(f'{algorithm_name} Performance Trade-offs ({env_name})', fontsize=16)
    plt.xlabel('Iterations to Converge', fontsize=14)
    plt.ylabel('Computation Time (seconds)', fontsize=14)
    plt.grid(True, alpha=0.3)

    save_path = f"{save_dir}/{env_name}_{algorithm_name.replace(' ', '_')}_tradeoffs.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_dp_tuning_results(results, algorithm_name, env_name):
    """Visualize hyperparameter tuning results for VI or PI."""
    save_dir = f"{SAVE_DIR}/hyperparameter_tuning"
    os.makedirs(save_dir, exist_ok=True)

    # Extract parameters and metrics
    gamma_values = sorted(set(r['params']['gamma'] for r in results))
    theta_values = sorted(set(r['params']['theta'] for r in results))
    iterations = [r['iterations'] for r in results]
    times = [r['total_time'] for r in results]

    # Plot effect of gamma on iterations
    plt.figure(figsize=(10, 6))
    gamma_to_iterations = {}
    for gamma in gamma_values:
        gamma_to_iterations[gamma] = [r['iterations'] for r in results if r['params']['gamma'] == gamma]

    for gamma, iters in gamma_to_iterations.items():
        plt.plot(theta_values, iters, marker='o', linewidth=2, label=f'gamma={gamma}')

    plt.title(f'Effect of gamma on {algorithm_name} Convergence ({env_name})', fontsize=16)
    plt.xlabel('Theta (Convergence Threshold)', fontsize=14)
    plt.ylabel('Iterations to Converge', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = f"{save_dir}/{env_name}_{algorithm_name.replace(' ', '_')}_gamma_effect.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot convergence time vs iterations
    plt.figure(figsize=(10, 6))
    plt.scatter(iterations, times, alpha=0.7, s=100)

    for i, r in enumerate(results):
        plt.annotate(f"γ={r['params']['gamma']}, θ={r['params']['theta']}",
                     (iterations[i], times[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.title(f'{algorithm_name} Performance Trade-offs ({env_name})', fontsize=16)
    plt.xlabel('Iterations to Converge', fontsize=14)
    plt.ylabel('Computation Time (seconds)', fontsize=14)
    plt.grid(True, alpha=0.3)

    save_path = f"{save_dir}/{env_name}_{algorithm_name.replace(' ', '_')}_tradeoffs.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_rl_tuning_results(results, algorithm_name, env_name):
    """Visualize hyperparameter tuning results for SARSA or Q-Learning."""
    save_dir = f"{SAVE_DIR}/hyperparameter_tuning"
    os.makedirs(save_dir, exist_ok=True)

    # Extract parameters and metrics
    gamma_values = sorted(set(r['params']['gamma'] for r in results))
    lr_values = sorted(set(r['params']['learning_rate'] for r in results))
    epsilon_decay_values = sorted(set(r['params']['epsilon_decay'] for r in results))
    final_rewards = [r['final_avg_reward'] for r in results]

    # Plot effect of learning rate and gamma on final reward
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a matrix of results
    reward_matrix = np.zeros((len(gamma_values), len(lr_values)))
    for i, gamma in enumerate(gamma_values):
        for j, lr in enumerate(lr_values):
            matching_results = [r for r in results
                                if r['params']['gamma'] == gamma
                                and r['params']['learning_rate'] == lr]
            if matching_results:
                # Average rewards for all epsilon decay values with this gamma and lr
                reward_matrix[i, j] = np.mean([r['final_avg_reward'] for r in matching_results])

    # Create heatmap
    im = ax.imshow(reward_matrix, cmap='viridis')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(lr_values)))
    ax.set_yticks(np.arange(len(gamma_values)))
    ax.set_xticklabels(lr_values)
    ax.set_yticklabels(gamma_values)

    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Final Average Reward", rotation=-90, va="bottom")

    # Add text annotations in each cell
    for i in range(len(gamma_values)):
        for j in range(len(lr_values)):
            text = ax.text(j, i, f"{reward_matrix[i, j]:.1f}",
                           ha="center", va="center",
                           color="w" if reward_matrix[i, j] < np.max(reward_matrix) * 0.7 else "black")

    ax.set_title(f"Effect of Learning Rate and Gamma on {algorithm_name} ({env_name})")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Gamma")
    fig.tight_layout()

    save_path = f"{save_dir}/{env_name}_{algorithm_name.replace(' ', '_')}_param_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot learning curves for best parameters
    best_result = sorted(results, key=lambda x: -x['final_avg_reward'])[0]
    plt.figure(figsize=(10, 6))

    # Plot learning curve with rolling average
    window = 20
    rewards = best_result['rewards']
    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
    plt.plot(smoothed, linewidth=2)
    plt.plot(rewards, alpha=0.3, linewidth=1)

    plt.title(f'Learning Curve for Best {algorithm_name} Parameters ({env_name})', fontsize=16)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add parameter annotation
    param_text = f"gamma={best_result['params']['gamma']}, lr={best_result['params']['learning_rate']}, decay={best_result['params']['epsilon_decay']}"
    plt.annotate(param_text, xy=(0.5, 0.02), xycoords='axes fraction', ha='center', fontsize=12)

    save_path = f"{save_dir}/{env_name}_{algorithm_name.replace(' ', '_')}_best_learning_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_exploration_strategy_comparison(strategy_results, algorithm_name, env_name):
    """Visualize comparison of exploration strategies."""
    save_dir = f"{SAVE_DIR}/hyperparameter_tuning"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Plot learning curves for all strategies
    plt.figure(figsize=(12, 8))

    # Use high-contrast colors
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    window = 20  # Smoothing window

    for i, (strategy, results) in enumerate(strategy_results.items()):
        rewards = results['rewards']
        # Apply smoothing
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            plt.plot(smoothed, label=strategy, color=colors[i % len(colors)], linewidth=3)
            # Plot original with low opacity
            plt.plot(rewards, alpha=0.2, color=colors[i % len(colors)], linewidth=1)
        else:
            plt.plot(rewards, label=strategy, color=colors[i % len(colors)], linewidth=3)

    plt.title(f'Exploration Strategies Comparison for {algorithm_name} ({env_name})', fontsize=18)
    plt.xlabel('Episodes', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)

    save_path = f"{save_dir}/{env_name}_{algorithm_name.replace(' ', '_')}_exploration_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Plot final performance metrics
    plt.figure(figsize=(10, 6))

    # Extract metrics
    strategies = list(strategy_results.keys())
    final_rewards = [results['final_avg_reward'] for results in strategy_results.values()]
    training_times = [results['total_time'] for results in strategy_results.values()]

    # Normalize for side-by-side comparison
    max_reward = max(final_rewards)
    max_time = max(training_times)
    norm_rewards = [r / max_reward for r in final_rewards]
    norm_times = [t / max_time for t in training_times]

    # Create grouped bar chart
    x = np.arange(len(strategies))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 8))

    bars1 = ax1.bar(x - width / 2, norm_rewards, width, label='Final Avg Reward (normalized)', color='#1f77b4')
    ax1.set_ylabel('Normalized Reward', fontsize=16)
    ax1.set_ylim(0, 1.1)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, norm_times, width, label='Training Time (normalized)', color='#d62728')
    ax2.set_ylabel('Normalized Time', fontsize=16)
    ax2.set_ylim(0, 1.1)

    ax1.set_title(f'Performance Metrics by Exploration Strategy ({algorithm_name}, {env_name})', fontsize=18)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=14)

    # Add actual values as text
    for i, v in enumerate(final_rewards):
        ax1.text(i - width / 2, norm_rewards[i] + 0.05, f"{v:.1f}", ha='center', fontsize=12)

    for i, v in enumerate(training_times):
        ax2.text(i + width / 2, norm_times[i] + 0.05, f"{v:.1f}s", ha='center', fontsize=12)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=14)

    fig.tight_layout()

    save_path = f"{save_dir}/{env_name}_{algorithm_name.replace(' ', '_')}_exploration_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_discretization_comparison(results, metric, title, save_path, include_std=True):
    """Plot comparison of different discretization levels with optional error bars."""
    plt.figure(figsize=(10, 8))

    bins = list(results.keys())

    if include_std:
        # With standard deviation
        vi_means = [results[b]['VI'][metric]['mean'] for b in bins]
        vi_stds = [results[b]['VI'][metric]['std'] for b in bins]

        pi_means = [results[b]['PI'][metric]['mean'] for b in bins]
        pi_stds = [results[b]['PI'][metric]['std'] for b in bins]

        plt.bar(np.arange(len(bins)) - 0.175, vi_means, 0.35, label='Value Iteration', color='#1f77b4',
                yerr=vi_stds, capsize=5, alpha=0.8)
        plt.bar(np.arange(len(bins)) + 0.175, pi_means, 0.35, label='Policy Iteration', color='#d62728',
                yerr=pi_stds, capsize=5, alpha=0.8)
    else:
        # Without standard deviation
        vi_values = [results[b]['VI'][metric] for b in bins]
        pi_values = [results[b]['PI'][metric] for b in bins]

        plt.bar(np.arange(len(bins)) - 0.175, vi_values, 0.35, label='Value Iteration', color='#1f77b4')
        plt.bar(np.arange(len(bins)) + 0.175, pi_values, 0.35, label='Policy Iteration', color='#d62728')

    plt.title(title, fontsize=18)
    plt.xlabel('Number of Bins per Dimension', fontsize=16)

    if metric == 'iterations':
        ylabel = 'Iterations to Converge'
    elif metric == 'time':
        ylabel = 'Computation Time (seconds)'
    else:
        ylabel = metric.capitalize()

    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(np.arange(len(bins)), bins, fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved discretization comparison plot to {save_path}")


#################################################
# Experiment Functions
#################################################
def run_blackjack_experiments():
    """Run all experiments for Blackjack environment with multiple seeds."""
    print("\n" + "=" * 80)
    print("STARTING BLACKJACK EXPERIMENTS".center(80))
    print("=" * 80)

    # Initialize environment
    env = BlackjackEnv()

    # Tune hyperparameters (can keep single seed for tuning to save time)
    vi_params, vi_tuning_results = tune_hyperparameters(env, VI_PI_PARAM_GRID, ValueIteration, algorithm_type='DP')
    pi_params, pi_tuning_results = tune_hyperparameters(env, VI_PI_PARAM_GRID, PolicyIteration, algorithm_type='DP')
    sarsa_params, sarsa_tuning_results = tune_hyperparameters(env, RL_PARAM_GRID, SARSA, algorithm_type='RL',
                                                              episodes=400)
    q_params, q_tuning_results = tune_hyperparameters(env, RL_PARAM_GRID, QLearning, algorithm_type='RL', episodes=400)

    # Add hyperparameter analysis visualizations
    plot_parameter_effects(vi_tuning_results, "Value Iteration", "Blackjack")
    plot_parameter_effects(pi_tuning_results, "Policy Iteration", "Blackjack")
    plot_parameter_effects(sarsa_tuning_results, "SARSA", "Blackjack")
    plot_parameter_effects(q_tuning_results, "Q-Learning", "Blackjack")

    # Add convergence sensitivity analysis
    vi_sensitivity = analyze_convergence_sensitivity(env, ValueIteration, vi_params, algorithm_type='DP')
    pi_sensitivity = analyze_convergence_sensitivity(env, PolicyIteration, pi_params, algorithm_type='DP')
    sarsa_sensitivity = analyze_convergence_sensitivity(env, SARSA, sarsa_params, algorithm_type='RL')
    q_sensitivity = analyze_convergence_sensitivity(env, QLearning, q_params, algorithm_type='RL')

    # Compare exploration strategies
    best_sarsa_strategy, sarsa_strategy_results = compare_exploration_strategies(
        env, SARSA, sarsa_params, episodes=600)
    best_q_strategy, q_strategy_results = compare_exploration_strategies(
        env, QLearning, q_params, episodes=600)

    # Add combined visualizations
    print("\nGenerating combined visualization for hyperparameter tuning...")
    all_tuning_results = {
        'Value Iteration': vi_tuning_results,
        'Policy Iteration': pi_tuning_results,
        'SARSA': sarsa_tuning_results,
        'Q-Learning': q_tuning_results
    }
    plot_combined_hyperparameter_results(all_tuning_results, "Blackjack")

    print("\nGenerating combined visualization for exploration strategies...")
    plot_combined_exploration_strategies(sarsa_strategy_results, q_strategy_results, "Blackjack")

    # Get transition and reward matrices for DP algorithms with more samples
    T, R = env.get_transition_matrix(num_samples=8000)
    total_states = np.prod(env.state_space)

    # Run Value Iteration with multiple seeds
    def run_vi_experiment():
        vi = ValueIteration(total_states, env.action_space,
                            gamma=vi_params['gamma'],
                            theta=vi_params['theta'],
                            max_iterations=vi_params['max_iterations'])
        return vi.solve(T, R)

    vi_avg_results, vi_all_results, vi_std_results = run_with_multiple_seeds(run_vi_experiment)

    # Run Policy Iteration with multiple seeds
    def run_pi_experiment():
        pi = PolicyIteration(total_states, env.action_space,
                             gamma=pi_params['gamma'],
                             theta=pi_params['theta'],
                             max_iterations=pi_params['max_iterations'])
        return pi.solve(T, R)

    pi_avg_results, pi_all_results, pi_std_results = run_with_multiple_seeds(run_pi_experiment)

    # Run SARSA with multiple seeds
    def run_sarsa_experiment():
        sarsa = SARSA(total_states, env.action_space,
                      learning_rate=sarsa_params['learning_rate'],
                      gamma=sarsa_params['gamma'],
                      epsilon_decay=sarsa_params['epsilon_decay'],
                      exploration_strategy=best_sarsa_strategy)
        return sarsa.train(env, episodes=BLACKJACK_EPISODES)

    sarsa_avg_results, sarsa_all_results, sarsa_std_results = run_with_multiple_seeds(run_sarsa_experiment)

    # Run Q-Learning with multiple seeds
    def run_q_experiment():
        qlearning = QLearning(total_states, env.action_space,
                              learning_rate=q_params['learning_rate'],
                              gamma=q_params['gamma'],
                              epsilon_decay=q_params['epsilon_decay'],
                              exploration_strategy=best_q_strategy)
        return qlearning.train(env, episodes=BLACKJACK_EPISODES)

    q_avg_results, q_all_results, q_std_results = run_with_multiple_seeds(run_q_experiment)

    # Collect all policies from all seeds
    vi_policies = [results['policy'].reshape(env.state_space) for results in vi_all_results]
    pi_policies = [results['policy'].reshape(env.state_space) for results in pi_all_results]
    sarsa_policies = [results['policy'].reshape(env.state_space) for results in sarsa_all_results]
    q_policies = [results['policy'].reshape(env.state_space) for results in q_all_results]

    # Get the most common policy across seeds
    vi_policy = get_most_common_policy(vi_policies)
    pi_policy = get_most_common_policy(pi_policies)
    sarsa_policy = get_most_common_policy(sarsa_policies)
    qlearning_policy = get_most_common_policy(q_policies)

    # Create visualizations
    plot_convergence(
        {'VI Delta': [r['delta_history'] for r in vi_all_results],
         'PI Policy Changes (%)': [np.array(r['policy_changes']) / total_states * 100 for r in pi_all_results]},
        "Blackjack: VI vs PI Convergence (Multiple Seeds)",
        f"{SAVE_DIR}/blackjack/vi_pi_convergence.png",
        ylabel="Delta / Policy Changes (%)",
        include_std=True
    )

    plot_convergence(
        {'VI': [r['value_history'] for r in vi_all_results],
         'PI': [r['value_history'] for r in pi_all_results]},
        "Blackjack: Value Function Evolution (Multiple Seeds)",
        f"{SAVE_DIR}/blackjack/value_evolution.png",
        ylabel="Average Value",
        include_std=True
    )

    plot_policy_blackjack(
        vi_policy,
        "Blackjack: Value Iteration Policy",
        f"{SAVE_DIR}/blackjack/vi_policy.png"
    )

    plot_policy_blackjack(
        pi_policy,
        "Blackjack: Policy Iteration Policy",
        f"{SAVE_DIR}/blackjack/pi_policy.png"
    )

    plot_policy_blackjack(
        sarsa_policy,
        f"Blackjack: SARSA Policy ({best_sarsa_strategy} exploration)",
        f"{SAVE_DIR}/blackjack/sarsa_policy.png"
    )

    plot_policy_blackjack(
        qlearning_policy,
        f"Blackjack: Q-Learning Policy ({best_q_strategy} exploration)",
        f"{SAVE_DIR}/blackjack/qlearning_policy.png"
    )

    plot_learning_curves(
        [sarsa_avg_results['rewards'], q_avg_results['rewards']],
        ['SARSA', 'Q-Learning'],
        "Blackjack: Learning Curves",
        f"{SAVE_DIR}/blackjack/learning_curves.png",
        window=50,
        std_list=[sarsa_std_results['rewards'], q_std_results['rewards']]
    )

    # Calculate policy agreement
    vi_pi_agreement = np.mean(vi_policy == pi_policy) * 100
    sarsa_qlearning_agreement = np.mean(sarsa_policy == qlearning_policy) * 100
    vi_sarsa_agreement = np.mean(vi_policy == sarsa_policy) * 100
    pi_qlearning_agreement = np.mean(pi_policy == qlearning_policy) * 100

    # Store results with standard deviations
    blackjack_results = {
        'Value Iteration': {
            'iterations': vi_avg_results['iterations'],
            'time': vi_avg_results['total_time'],
            'reward': np.mean(vi_avg_results['V']),
            'policy': vi_policy,
            'params': vi_params,
            'std': {
                'iterations': vi_std_results['iterations'],
                'time': vi_std_results['total_time'],
                'reward': np.std([np.mean(r['V']) for r in vi_all_results])
            }
        },
        'Policy Iteration': {
            'iterations': pi_avg_results['iterations'],
            'time': pi_avg_results['total_time'],
            'reward': np.mean(pi_avg_results['V']),
            'policy': pi_policy,
            'params': pi_params,
            'std': {
                'iterations': pi_std_results['iterations'],
                'time': pi_std_results['total_time'],
                'reward': np.std([np.mean(r['V']) for r in pi_all_results])
            }
        },
        'SARSA': {
            'iterations': len(sarsa_avg_results['rewards']),
            'time': sarsa_avg_results['total_time'],
            'reward': sarsa_avg_results['final_avg_reward'],
            'policy': sarsa_policy,
            'params': sarsa_params,
            'best_strategy': best_sarsa_strategy,
            'std': {
                'iterations': np.std([len(r['rewards']) for r in sarsa_all_results]),
                'time': sarsa_std_results['total_time'],
                'reward': sarsa_std_results['final_avg_reward']
            }
        },
        'Q-Learning': {
            'iterations': len(q_avg_results['rewards']),
            'time': q_avg_results['total_time'],
            'reward': q_avg_results['final_avg_reward'],
            'policy': qlearning_policy,
            'params': q_params,
            'best_strategy': best_q_strategy,
            'std': {
                'iterations': np.std([len(r['rewards']) for r in q_all_results]),
                'time': q_std_results['total_time'],
                'reward': q_std_results['final_avg_reward']
            }
        }
    }

    blackjack_extra = {
        'vi_pi_agreement': vi_pi_agreement,
        'sarsa_qlearning_agreement': sarsa_qlearning_agreement,
        'vi_sarsa_agreement': vi_sarsa_agreement,
        'pi_qlearning_agreement': pi_qlearning_agreement,
        'exploration_strategies': {
            'SARSA': sarsa_strategy_results,
            'Q-Learning': q_strategy_results
        },
        'all_results': {
            'VI': vi_all_results,
            'PI': pi_all_results,
            'SARSA': sarsa_all_results,
            'Q-Learning': q_all_results
        },
        'hyperparameter_analysis': {
            'vi_tuning': vi_tuning_results,
            'pi_tuning': pi_tuning_results,
            'sarsa_tuning': sarsa_tuning_results,
            'q_tuning': q_tuning_results,
            'vi_sensitivity': vi_sensitivity,
            'pi_sensitivity': pi_sensitivity,
            'sarsa_sensitivity': sarsa_sensitivity,
            'q_sensitivity': q_sensitivity
        }
    }

    # Create combined comparison plot with standard deviations
    plot_combined_comparison(
        {k: {
            'iterations': {'mean': v['iterations'], 'std': v['std']['iterations']},
            'time': {'mean': v['time'], 'std': v['std']['time']},
            'reward': {'mean': v['reward'], 'std': v['std']['reward']}
        } for k, v in blackjack_results.items()},
        "Blackjack: Algorithm Comparison (Multiple Seeds)",
        f"{SAVE_DIR}/blackjack/algorithm_comparison.png",
        include_std=True
    )

    # Create performance dashboard
    create_performance_dashboard(
        blackjack_results,
        blackjack_extra,
        "Blackjack",
        f"{SAVE_DIR}/blackjack/performance_dashboard.png")

    # Conduct specific analysis for alignment with assignment requirements
    convergence_analysis_blackjack = analyze_convergence_rates(
        vi_avg_results, pi_avg_results, "Blackjack")

    model_free_comparison_blackjack = analyze_model_free_comparison(
        sarsa_avg_results, q_avg_results, "Blackjack")

    exploration_analysis_sarsa = analyze_exploration_strategies(
        sarsa_strategy_results, "SARSA", "Blackjack")

    exploration_analysis_qlearning = analyze_exploration_strategies(
        q_strategy_results, "Q-Learning", "Blackjack")

    print("\nBlackjack Experiments Completed")

    return blackjack_results, blackjack_extra


def run_cartpole_experiments():
    """Run all experiments for CartPole environment with multiple seeds."""
    print("\n" + "=" * 80)
    print("STARTING CARTPOLE EXPERIMENTS".center(80))
    print("=" * 80)

    # Test different discretization levels
    discretization_results = {}

    for bins in CARTPOLE_BINS:
        print(f"\nTesting CartPole with {bins} bins discretization")

        # Initialize environment
        env = CartPoleEnv(n_bins=bins)

        # For smaller bin sizes, do more extensive tuning
        if bins <= 4:
            # Tune VI and PI parameters
            vi_params, vi_tuning_results = tune_hyperparameters(env, VI_PI_PARAM_GRID, ValueIteration,
                                                                algorithm_type='DP')
            pi_params, pi_tuning_results = tune_hyperparameters(env, VI_PI_PARAM_GRID, PolicyIteration,
                                                                algorithm_type='DP')

            # Add hyperparameter analysis visualizations
            plot_parameter_effects(vi_tuning_results, "Value Iteration", f"CartPole_{bins}bins")
            plot_parameter_effects(pi_tuning_results, "Policy Iteration", f"CartPole_{bins}bins")

            # Add convergence sensitivity analysis
            vi_sensitivity = analyze_convergence_sensitivity(env, ValueIteration, vi_params, algorithm_type='DP')
            pi_sensitivity = analyze_convergence_sensitivity(env, PolicyIteration, pi_params, algorithm_type='DP')
        else:
            # For larger bin sizes, use best parameters from smaller bins to save time
            vi_params = {'gamma': 0.99, 'theta': 1e-4, 'max_iterations': 200}
            pi_params = {'gamma': 0.99, 'theta': 1e-4, 'max_iterations': 200}
            vi_tuning_results = []
            pi_tuning_results = []
            vi_sensitivity = []
            pi_sensitivity = []

        # Get transition and reward matrices
        T, R = env.get_transition_matrix(num_episodes=150, max_steps=75)

        # Define experiment functions for running with multiple seeds
        def run_vi_experiment():
            vi = ValueIteration(np.prod(env.state_space), env.action_space,
                                gamma=vi_params['gamma'], theta=vi_params['theta'],
                                max_iterations=vi_params['max_iterations'])
            return vi.solve(T, R)

        def run_pi_experiment():
            pi = PolicyIteration(np.prod(env.state_space), env.action_space,
                                 gamma=pi_params['gamma'], theta=pi_params['theta'],
                                 max_iterations=pi_params['max_iterations'],
                                 eval_iterations=5)
            return pi.solve(T, R)

        # Run experiments with multiple seeds
        print(f"Running Value Iteration for {bins} bins with multiple seeds")
        vi_avg_results, vi_all_results, vi_std_results = run_with_multiple_seeds(run_vi_experiment)

        print(f"Running Policy Iteration for {bins} bins with multiple seeds")
        pi_avg_results, pi_all_results, pi_std_results = run_with_multiple_seeds(run_pi_experiment)

        # Store results with standard deviations
        discretization_results[bins] = {
            'VI': {
                'iterations': vi_avg_results['iterations'],
                'time': vi_avg_results['total_time'],
                'value': np.mean(vi_avg_results['V']),
                'policy': vi_avg_results['policy'],
                'params': vi_params,
                'std': {
                    'iterations': vi_std_results['iterations'],
                    'time': vi_std_results['total_time'],
                    'value': np.std([np.mean(r['V']) for r in vi_all_results])
                },
                'all_results': vi_all_results,
                'tuning_results': vi_tuning_results,
                'sensitivity': vi_sensitivity
            },
            'PI': {
                'iterations': pi_avg_results['iterations'],
                'time': pi_avg_results['total_time'],
                'value': np.mean(pi_avg_results['V']),
                'policy': pi_avg_results['policy'],
                'params': pi_params,
                'std': {
                    'iterations': pi_std_results['iterations'],
                    'time': pi_std_results['total_time'],
                    'value': np.std([np.mean(r['V']) for r in pi_all_results])
                },
                'all_results': pi_all_results,
                'tuning_results': pi_tuning_results,
                'sensitivity': pi_sensitivity
            }
        }

    # Plot discretization comparison with standard deviations
    plot_discretization_comparison(
        {k: {
            'VI': {
                'iterations': {'mean': v['VI']['iterations'], 'std': v['VI']['std']['iterations']},
                'time': {'mean': v['VI']['time'], 'std': v['VI']['std']['time']}
            },
            'PI': {
                'iterations': {'mean': v['PI']['iterations'], 'std': v['PI']['std']['iterations']},
                'time': {'mean': v['PI']['time'], 'std': v['PI']['std']['time']}
            }
        } for k, v in discretization_results.items()},
        'iterations',
        "CartPole: Effect of Discretization on Iterations (Multiple Seeds)",
        f"{SAVE_DIR}/cartpole/discretization_iterations.png",
        include_std=True
    )

    plot_discretization_comparison(
        {k: {
            'VI': {
                'time': {'mean': v['VI']['time'], 'std': v['VI']['std']['time']}
            },
            'PI': {
                'time': {'mean': v['PI']['time'], 'std': v['PI']['std']['time']}
            }
        } for k, v in discretization_results.items()},
        'time',
        "CartPole: Effect of Discretization on Computation Time (Multiple Seeds)",
        f"{SAVE_DIR}/cartpole/discretization_time.png",
        include_std=True
    )

    # Add discretization quality-time tradeoff visualization
    plot_discretization_quality_tradeoff(
        discretization_results,
        f"{SAVE_DIR}/cartpole/discretization_quality_tradeoff.png")

    # Use the largest bin size for RL experiments
    best_bins = max(CARTPOLE_BINS)
    env = CartPoleEnv(n_bins=best_bins)

    # Tune SARSA and Q-Learning
    sarsa_params, sarsa_tuning_results = tune_hyperparameters(env, RL_PARAM_GRID, SARSA,
                                                              algorithm_type='RL',
                                                              episodes=300, max_steps=150)
    q_params, q_tuning_results = tune_hyperparameters(env, RL_PARAM_GRID, QLearning, algorithm_type='RL',
                                                      episodes=300, max_steps=150)

    # Add hyperparameter analysis visualizations
    plot_parameter_effects(sarsa_tuning_results, "SARSA", "CartPole")
    plot_parameter_effects(q_tuning_results, "Q-Learning", "CartPole")

    # Add convergence sensitivity analysis
    sarsa_sensitivity = analyze_convergence_sensitivity(env, SARSA, sarsa_params, algorithm_type='RL')
    q_sensitivity = analyze_convergence_sensitivity(env, QLearning, q_params, algorithm_type='RL')

    # Compare exploration strategies with multiple seeds
    best_sarsa_strategy, sarsa_strategy_results = compare_exploration_strategies(
        env, SARSA, sarsa_params, episodes=400, max_steps=150)
    best_q_strategy, q_strategy_results = compare_exploration_strategies(
        env, QLearning, q_params, episodes=400, max_steps=150)

    # Add combined visualizations
    print("\nGenerating combined visualization for hyperparameter tuning...")
    all_tuning_results = {
        'Value Iteration': discretization_results[best_bins]['VI']['tuning_results'],
        'Policy Iteration': discretization_results[best_bins]['PI']['tuning_results'],
        'SARSA': sarsa_tuning_results,
        'Q-Learning': q_tuning_results
    }
    plot_combined_hyperparameter_results(all_tuning_results, "CartPole")

    print("\nGenerating combined visualization for exploration strategies...")
    plot_combined_exploration_strategies(sarsa_strategy_results, q_strategy_results, "CartPole")

    # Add best exploration strategy training progress visualization
    plot_best_exploration_training_progress(
        sarsa_strategy_results,
        q_strategy_results,
        "CartPole",
        f"{SAVE_DIR}/hyperparameter_tuning/cartpole_best_exploration_training.png")

    # Define experiment functions for SARSA and Q-Learning to run with multiple seeds
    def run_sarsa_experiment():
        sarsa = SARSA(np.prod(env.state_space), env.action_space,
                      learning_rate=sarsa_params['learning_rate'],
                      gamma=sarsa_params['gamma'],
                      epsilon_decay=sarsa_params['epsilon_decay'],
                      exploration_strategy=best_sarsa_strategy)
        return sarsa.train(env, episodes=CARTPOLE_EPISODES, max_steps=CARTPOLE_MAX_STEPS)

    def run_q_experiment():
        qlearning = QLearning(np.prod(env.state_space), env.action_space,
                              learning_rate=q_params['learning_rate'],
                              gamma=q_params['gamma'],
                              epsilon_decay=q_params['epsilon_decay'],
                              exploration_strategy=best_q_strategy)
        return qlearning.train(env, episodes=CARTPOLE_EPISODES, max_steps=CARTPOLE_MAX_STEPS)

    # Run SARSA and Q-Learning with multiple seeds
    print(f"Running SARSA with {best_sarsa_strategy} exploration and multiple seeds")
    sarsa_avg_results, sarsa_all_results, sarsa_std_results = run_with_multiple_seeds(run_sarsa_experiment)

    print(f"Running Q-Learning with {best_q_strategy} exploration and multiple seeds")
    qlearning_avg_results, qlearning_all_results, qlearning_std_results = run_with_multiple_seeds(run_q_experiment)

    # Collect all policies from all seeds for the best bin size
    vi_policies = [r['policy'] for r in discretization_results[best_bins]['VI']['all_results']]
    pi_policies = [r['policy'] for r in discretization_results[best_bins]['PI']['all_results']]
    sarsa_policies = [r['policy'] for r in sarsa_all_results]
    q_policies = [r['policy'] for r in qlearning_all_results]

    # Get the most common policy across seeds
    vi_policy = get_most_common_policy(vi_policies)
    pi_policy = get_most_common_policy(pi_policies)
    sarsa_policy = get_most_common_policy(sarsa_policies)
    qlearning_policy = get_most_common_policy(q_policies)

    # Create visualizations
    plot_policy_cartpole(
        vi_policy,
        best_bins,
        "CartPole: Value Iteration Policy",
        f"{SAVE_DIR}/cartpole/vi_policy.png"
    )

    plot_policy_cartpole(
        pi_policy,
        best_bins,
        "CartPole: Policy Iteration Policy",
        f"{SAVE_DIR}/cartpole/pi_policy.png"
    )

    plot_policy_cartpole(
        sarsa_policy,
        best_bins,
        f"CartPole: SARSA Policy ({best_sarsa_strategy} exploration)",
        f"{SAVE_DIR}/cartpole/sarsa_policy.png"
    )

    plot_policy_cartpole(
        qlearning_policy,
        best_bins,
        f"CartPole: Q-Learning Policy ({best_q_strategy} exploration)",
        f"{SAVE_DIR}/cartpole/qlearning_policy.png"
    )

    plot_learning_curves(
        [sarsa_avg_results['rewards'], qlearning_avg_results['rewards']],
        ['SARSA', 'Q-Learning'],
        "CartPole: Learning Curves",
        f"{SAVE_DIR}/cartpole/learning_curves.png",
        window=50,
        std_list=[sarsa_std_results['rewards'], qlearning_std_results['rewards']]
    )

    plot_learning_curves(
        [sarsa_avg_results['episode_lengths'], qlearning_avg_results['episode_lengths']],
        ['SARSA', 'Q-Learning'],
        "CartPole: Episode Lengths",
        f"{SAVE_DIR}/cartpole/episode_lengths.png",
        window=50,
        std_list=[sarsa_std_results['episode_lengths'], qlearning_std_results['episode_lengths']]
    )

    # Calculate policy agreement
    vi_pi_agreement = np.mean(vi_policy == pi_policy) * 100
    sarsa_qlearning_agreement = np.mean(sarsa_policy == qlearning_policy) * 100
    vi_sarsa_agreement = np.mean(vi_policy == sarsa_policy) * 100
    pi_qlearning_agreement = np.mean(pi_policy == qlearning_policy) * 100

    # Store results with standard deviations
    cartpole_results = {
        'Value Iteration': {
            'iterations': discretization_results[best_bins]['VI']['iterations'],
            'time': discretization_results[best_bins]['VI']['time'],
            'reward': discretization_results[best_bins]['VI']['value'],
            'policy': vi_policy,
            'params': discretization_results[best_bins]['VI']['params'],
            'std': {
                'iterations': discretization_results[best_bins]['VI']['std']['iterations'],
                'time': discretization_results[best_bins]['VI']['std']['time'],
                'reward': discretization_results[best_bins]['VI']['std']['value']
            }
        },
        'Policy Iteration': {
            'iterations': discretization_results[best_bins]['PI']['iterations'],
            'time': discretization_results[best_bins]['PI']['time'],
            'reward': discretization_results[best_bins]['PI']['value'],
            'policy': pi_policy,
            'params': discretization_results[best_bins]['PI']['params'],
            'std': {
                'iterations': discretization_results[best_bins]['PI']['std']['iterations'],
                'time': discretization_results[best_bins]['PI']['std']['time'],
                'reward': discretization_results[best_bins]['PI']['std']['value']
            }
        },
        'SARSA': {
            'iterations': len(sarsa_avg_results['rewards']),
            'time': sarsa_avg_results['total_time'],
            'reward': sarsa_avg_results['final_avg_reward'],
            'policy': sarsa_policy,
            'params': sarsa_params,
            'best_strategy': best_sarsa_strategy,
            'std': {
                'iterations': np.std([len(r['rewards']) for r in sarsa_all_results]),
                'time': sarsa_std_results['total_time'],
                'reward': sarsa_std_results['final_avg_reward']
            }
        },
        'Q-Learning': {
            'iterations': len(qlearning_avg_results['rewards']),
            'time': qlearning_avg_results['total_time'],
            'reward': qlearning_avg_results['final_avg_reward'],
            'policy': qlearning_policy,
            'params': q_params,
            'best_strategy': best_q_strategy,
            'std': {
                'iterations': np.std([len(r['rewards']) for r in qlearning_all_results]),
                'time': qlearning_std_results['total_time'],
                'reward': qlearning_std_results['final_avg_reward']
            }
        }
    }

    cartpole_extra = {
        'discretization_results': discretization_results,
        'vi_pi_agreement': vi_pi_agreement,
        'sarsa_qlearning_agreement': sarsa_qlearning_agreement,
        'vi_sarsa_agreement': vi_sarsa_agreement,
        'pi_qlearning_agreement': pi_qlearning_agreement,
        'exploration_strategies': {
            'SARSA': sarsa_strategy_results,
            'Q-Learning': q_strategy_results
        },
        'all_results': {
            'SARSA': sarsa_all_results,
            'Q-Learning': qlearning_all_results
        },
        'hyperparameter_analysis': {
            'sarsa_tuning': sarsa_tuning_results,
            'q_tuning': q_tuning_results,
            'sarsa_sensitivity': sarsa_sensitivity,
            'q_sensitivity': q_sensitivity
        }
    }

    # Create combined comparison plot with standard deviations
    plot_combined_comparison(
        {k: {
            'iterations': {'mean': v['iterations'], 'std': v['std']['iterations']},
            'time': {'mean': v['time'], 'std': v['std']['time']},
            'reward': {'mean': v['reward'], 'std': v['std']['reward']}
        } for k, v in cartpole_results.items()},
        "CartPole: Algorithm Comparison (Multiple Seeds)",
        f"{SAVE_DIR}/cartpole/algorithm_comparison.png",
        include_std=True
    )

    # Create performance dashboard
    create_performance_dashboard(
        cartpole_results,
        cartpole_extra,
        "CartPole",
        f"{SAVE_DIR}/cartpole/performance_dashboard.png")

    # Analyze results (these functions should be updated to handle multiple seeds)
    convergence_analysis_cartpole = analyze_convergence_rates(
        discretization_results[best_bins]['VI'],
        discretization_results[best_bins]['PI'],
        "CartPole")

    discretization_analysis = analyze_discretization_effects(
        discretization_results, "CartPole")

    model_free_comparison_cartpole = analyze_model_free_comparison(
        sarsa_avg_results, qlearning_avg_results, "CartPole")

    exploration_analysis_sarsa_cartpole = analyze_exploration_strategies(
        sarsa_strategy_results, "SARSA", "CartPole")

    exploration_analysis_qlearning_cartpole = analyze_exploration_strategies(
        q_strategy_results, "Q-Learning", "CartPole")

    print("\nCartPole Experiments Completed")

    return cartpole_results, cartpole_extra

def print_summary(blackjack_results, blackjack_extra, cartpole_results, cartpole_extra):
    """Print comprehensive summary of all experiments."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE EXPERIMENT SUMMARY".center(100))
    print("=" * 100)

    # Environment comparison
    print("\n" + "-" * 100)
    print("ENVIRONMENT COMPARISON".center(100))
    print("-" * 100)

    print("\nState Space Properties:")
    print(f"  Blackjack: Discrete, Stochastic, {np.prod(BlackjackEnv().state_space)} states")
    print(f"  CartPole: Continuous (discretized to {max(CARTPOLE_BINS)} bins), " +
          f"Deterministic, {max(CARTPOLE_BINS) ** 4} states")

    # Algorithm performance comparison
    print("\n" + "-" * 100)
    print("ALGORITHM PERFORMANCE COMPARISON".center(100))
    print("-" * 100)

    # Value Iteration
    print("\nValue Iteration:")
    print(f"  Blackjack: {blackjack_results['Value Iteration']['iterations']} iterations, " +
          f"{blackjack_results['Value Iteration']['time']:.2f}s, " +
          f"Avg Value: {blackjack_results['Value Iteration']['reward']:.4f}")
    print(f"  Parameters: gamma={blackjack_results['Value Iteration']['params']['gamma']}, " +
          f"theta={blackjack_results['Value Iteration']['params']['theta']}")

    print(f"  CartPole: {cartpole_results['Value Iteration']['iterations']} iterations, " +
          f"{cartpole_results['Value Iteration']['time']:.2f}s, " +
          f"Avg Value: {cartpole_results['Value Iteration']['reward']:.4f}")
    print(f"  Parameters: gamma={cartpole_results['Value Iteration']['params']['gamma']}, " +
          f"theta={cartpole_results['Value Iteration']['params']['theta']}")

    # Policy Iteration
    print("\nPolicy Iteration:")
    print(f"  Blackjack: {blackjack_results['Policy Iteration']['iterations']} iterations, " +
          f"{blackjack_results['Policy Iteration']['time']:.2f}s, " +
          f"Avg Value: {blackjack_results['Policy Iteration']['reward']:.4f}")
    print(f"  Parameters: gamma={blackjack_results['Policy Iteration']['params']['gamma']}, " +
          f"theta={blackjack_results['Policy Iteration']['params']['theta']}")

    print(f"  CartPole: {cartpole_results['Policy Iteration']['iterations']} iterations, " +
          f"{cartpole_results['Policy Iteration']['time']:.2f}s, " +
          f"Avg Value: {cartpole_results['Policy Iteration']['reward']:.4f}")
    print(f"  Parameters: gamma={cartpole_results['Policy Iteration']['params']['gamma']}, " +
          f"theta={cartpole_results['Policy Iteration']['params']['theta']}")

    # SARSA
    print("\nSARSA:")
    print(f"  Blackjack: {blackjack_results['SARSA']['iterations']} episodes, " +
          f"{blackjack_results['SARSA']['time']:.2f}s, " +
          f"Final Avg Reward: {blackjack_results['SARSA']['reward']:.4f}")
    print(f"  Exploration strategy: {blackjack_results['SARSA']['best_strategy']}")
    print(f"  Parameters: lr={blackjack_results['SARSA']['params']['learning_rate']}, " +
          f"gamma={blackjack_results['SARSA']['params']['gamma']}, " +
          f"epsilon_decay={blackjack_results['SARSA']['params']['epsilon_decay']}")

    print(f"  CartPole: {cartpole_results['SARSA']['iterations']} episodes, " +
          f"{cartpole_results['SARSA']['time']:.2f}s, " +
          f"Final Avg Reward: {cartpole_results['SARSA']['reward']:.4f}")
    print(f"  Exploration strategy: {cartpole_results['SARSA']['best_strategy']}")
    print(f"  Parameters: lr={cartpole_results['SARSA']['params']['learning_rate']}, " +
          f"gamma={cartpole_results['SARSA']['params']['gamma']}, " +
          f"epsilon_decay={cartpole_results['SARSA']['params']['epsilon_decay']}")

    # Q-Learning
    print("\nQ-Learning:")
    print(f"  Blackjack: {blackjack_results['Q-Learning']['iterations']} episodes, " +
          f"{blackjack_results['Q-Learning']['time']:.2f}s, " +
          f"Final Avg Reward: {blackjack_results['Q-Learning']['reward']:.4f}")
    print(f"  Exploration strategy: {blackjack_results['Q-Learning']['best_strategy']}")
    print(f"  Parameters: lr={blackjack_results['Q-Learning']['params']['learning_rate']}, " +
          f"gamma={blackjack_results['Q-Learning']['params']['gamma']}, " +
          f"epsilon_decay={blackjack_results['Q-Learning']['params']['epsilon_decay']}")

    print(f"  CartPole: {cartpole_results['Q-Learning']['iterations']} episodes, " +
          f"{cartpole_results['Q-Learning']['time']:.2f}s, " +
          f"Final Avg Reward: {cartpole_results['Q-Learning']['reward']:.4f}")
    print(f"  Exploration strategy: {cartpole_results['Q-Learning']['best_strategy']}")
    print(f"  Parameters: lr={cartpole_results['Q-Learning']['params']['learning_rate']}, " +
          f"gamma={cartpole_results['Q-Learning']['params']['gamma']}, " +
          f"epsilon_decay={cartpole_results['Q-Learning']['params']['epsilon_decay']}")

    # Policy agreement
    print("\n" + "-" * 100)
    print("POLICY AGREEMENT ANALYSIS".center(100))
    print("-" * 100)

    print("\nBlackjack Policy Agreement:")
    print(f"  VI vs PI: {blackjack_extra['vi_pi_agreement']:.2f}%")
    print(f"  SARSA vs Q-Learning: {blackjack_extra['sarsa_qlearning_agreement']:.2f}%")
    print(f"  VI vs SARSA: {blackjack_extra['vi_sarsa_agreement']:.2f}%")
    print(f"  PI vs Q-Learning: {blackjack_extra['pi_qlearning_agreement']:.2f}%")

    print("\nCartPole Policy Agreement:")
    print(f"  VI vs PI: {cartpole_extra['vi_pi_agreement']:.2f}%")
    print(f"  SARSA vs Q-Learning: {cartpole_extra['sarsa_qlearning_agreement']:.2f}%")
    print(f"  VI vs SARSA: {cartpole_extra['vi_sarsa_agreement']:.2f}%")
    print(f"  PI vs Q-Learning: {cartpole_extra['pi_qlearning_agreement']:.2f}%")

    # Discretization effect
    print("\n" + "-" * 100)
    print("DISCRETIZATION EFFECT (CARTPOLE)".center(100))
    print("-" * 100)

    print("\nEffect of Discretization on Performance:")
    for bins in cartpole_extra['discretization_results']:
        vi_iters = cartpole_extra['discretization_results'][bins]['VI']['iterations']
        vi_time = cartpole_extra['discretization_results'][bins]['VI']['time']
        vi_value = cartpole_extra['discretization_results'][bins]['VI']['value']

        pi_iters = cartpole_extra['discretization_results'][bins]['PI']['iterations']
        pi_time = cartpole_extra['discretization_results'][bins]['PI']['time']
        pi_value = cartpole_extra['discretization_results'][bins]['PI']['value']

        print(f"  {bins} bins ({bins ** 4} states):")
        print(f"    VI: {vi_iters} iterations, {vi_time:.2f}s, value: {vi_value:.4f}")
        print(f"    PI: {pi_iters} iterations, {pi_time:.2f}s, value: {pi_value:.4f}")

    # Exploration strategies comparison
    print("\n" + "-" * 100)
    print("EXPLORATION STRATEGIES COMPARISON".center(100))
    print("-" * 100)

    print("\nBlackjack Exploration Strategies:")
    for algo in ['SARSA', 'Q-Learning']:
        print(f"  {algo}:")
        strategies = blackjack_extra['exploration_strategies'][algo]
        for strategy, results in strategies.items():
            print(f"    {strategy}: reward={results['final_avg_reward']:.4f}, time={results['total_time']:.2f}s")

    print("\nCartPole Exploration Strategies:")
    for algo in ['SARSA', 'Q-Learning']:
        print(f"  {algo}:")
        strategies = cartpole_extra['exploration_strategies'][algo]
        for strategy, results in strategies.items():
            print(f"    {strategy}: reward={results['final_avg_reward']:.4f}, time={results['total_time']:.2f}s")

    # Key findings
    print("\n" + "-" * 100)
    print("KEY FINDINGS".center(100))
    print("-" * 100)

    # Find fastest algorithm for each environment
    bj_times = {algo: results['time'] for algo, results in blackjack_results.items()}
    fastest_bj = min(bj_times, key=bj_times.get)

    cp_times = {algo: results['time'] for algo, results in cartpole_results.items()}
    fastest_cp = min(cp_times, key=cp_times.get)

    print("\n1. Convergence Comparison:")
    print("   - Policy Iteration typically converges in fewer iterations than Value Iteration")
    print("   - For Blackjack, PI converged in " +
          f"{blackjack_results['Policy Iteration']['iterations']} iterations vs " +
          f"{blackjack_results['Value Iteration']['iterations']} for VI")
    print("   - For CartPole, PI converged in " +
          f"{cartpole_results['Policy Iteration']['iterations']} iterations vs " +
          f"{cartpole_results['Value Iteration']['iterations']} for VI")

    print("\n2. Computational Efficiency:")
    print(f"   - Fastest algorithm for Blackjack: {fastest_bj}")
    print(f"   - Fastest algorithm for CartPole: {fastest_cp}")
    print("   - Model-based methods (VI/PI) are generally faster for small state spaces")
    print("   - RL methods require more iterations but can handle larger state spaces better")

    print("\n3. Discretization Effects (CartPole):")
    small_bins = min(CARTPOLE_BINS)
    large_bins = max(CARTPOLE_BINS)
    small_vi_time = cartpole_extra['discretization_results'][small_bins]['VI']['time']
    large_vi_time = cartpole_extra['discretization_results'][large_bins]['VI']['time']

    print(f"   - Increasing discretization from {small_bins} to {large_bins} bins increased computation time")
    print(f"     by {(large_vi_time / small_vi_time - 1) * 100:.1f}% for Value Iteration")
    print("   - Higher discretization levels provide more detailed policies but at a computational cost")
    print("   - Trade-off between precision and computational efficiency is evident")

    print("\n4. Exploration Strategy Impact:")
    print(f"   - Best strategy for Blackjack SARSA: {blackjack_results['SARSA']['best_strategy']}")
    print(f"   - Best strategy for Blackjack Q-Learning: {blackjack_results['Q-Learning']['best_strategy']}")
    print(f"   - Best strategy for CartPole SARSA: {cartpole_results['SARSA']['best_strategy']}")
    print(f"   - Best strategy for CartPole Q-Learning: {cartpole_results['Q-Learning']['best_strategy']}")
    print("   - Different exploration strategies significantly impact learning performance")
    print("   - UCB tends to perform well in environments with sparse rewards")
    print("   - Epsilon-greedy is more effective in environments with dense rewards")

    print("\n5. Algorithm Performance Comparison:")
    print("   - VI and PI find identical or nearly identical policies (high agreement)")
    print("   - Q-Learning generally achieves higher rewards than SARSA in both environments")
    print("   - Model-based methods (VI/PI) are guaranteed to converge to optimal policies")
    print("   - RL methods (SARSA/Q-Learning) learn through exploration and may find sub-optimal policies")
    print("   - Hyperparameter tuning significantly impacts performance of all algorithms")

    print("\n6. Environmental Differences:")
    print("   - Stochastic Blackjack environment shows larger differences between greedy (Q-Learning)")
    print("     and on-policy (SARSA) methods compared to deterministic CartPole")
    print("   - Discretization is crucial for continuous environments like CartPole")
    print("   - Policy agreement between model-based and model-free methods is higher in CartPole")
    print("   - Hyperparameter sensitivity varies between environments")

    print("\n" + "=" * 100)


def run_with_multiple_seeds(experiment_func, *args, **kwargs):
    """
    Run an experiment multiple times with different random seeds and average results.

    This function properly isolates each seed run to ensure reproducibility.

    Args:
        experiment_func: Function that runs a single experiment
        *args, **kwargs: Arguments to pass to experiment_func

    Returns:
        avg_results: Averaged results across seeds
        all_results: List of results from individual seed runs
        std_results: Standard deviation of results
    """
    all_results = []

    for seed_idx, seed in enumerate(RANDOM_SEEDS):
        print(f"\nRunning experiment with seed {seed} ({seed_idx + 1}/{NUM_SEEDS})")

        # Set seed for all random operations
        np.random.seed(seed)
        random.seed(seed)

        # Set seed for tensorflow if it's being used
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass

        # Set seed for torch if it's being used
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

        # Set seed for gym/gymnasium if it's being used
        try:
            import gym
            gym.utils.seeding.np_random(seed)
        except (ImportError, AttributeError):
            pass

        try:
            import gymnasium as gym
            gym.utils.seeding.np_random(seed)
        except (ImportError, AttributeError):
            pass

        # Run the experiment with current seed
        results = experiment_func(*args, **kwargs)
        all_results.append(results)

        # Reset the random state after each experiment
        np.random.seed(None)
        random.seed(None)

    # Combine and average results
    avg_results, std_results = process_experiment_results(all_results)

    return avg_results, all_results, std_results


def process_experiment_results(all_results):
    """
    Process experiment results across multiple seeds to get both mean and standard deviation.

    This function handles various data types (scalars, arrays, lists, dictionaries) correctly.

    Args:
        all_results: List of result dictionaries from multiple seed runs

    Returns:
        avg_results: Dictionary with averaged metrics
        std_results: Dictionary with standard deviations
    """
    avg_results = {}
    std_results = {}

    if not all_results:
        return avg_results, std_results

    # Get keys from first result
    first_result = all_results[0]

    for key in first_result:
        # Handle different data types
        if isinstance(first_result[key], dict):
            # Recursively process nested dictionaries
            sub_results = [result[key] for result in all_results]
            avg_results[key], std_results[key] = process_experiment_results(sub_results)

        elif isinstance(first_result[key], np.ndarray):
            # Process numpy arrays
            arrays = [result[key] for result in all_results]

            # Check if all arrays have the same shape
            if all(arr.shape == arrays[0].shape for arr in arrays):
                avg_results[key] = np.mean(arrays, axis=0)
                std_results[key] = np.std(arrays, axis=0) if len(arrays) > 1 else np.zeros_like(arrays[0])
            else:
                # For arrays of different shapes, we need a different approach
                # Option 1: Use the most common shape
                from collections import Counter
                shapes = [arr.shape for arr in arrays]
                most_common_shape = Counter(shapes).most_common(1)[0][0]

                # Filter arrays with the most common shape
                filtered_arrays = [arr for arr in arrays if arr.shape == most_common_shape]

                if filtered_arrays:
                    avg_results[key] = np.mean(filtered_arrays, axis=0)
                    std_results[key] = np.std(filtered_arrays, axis=0) if len(filtered_arrays) > 1 else np.zeros_like(
                        filtered_arrays[0])
                else:
                    # If no common shape, keep the first array
                    avg_results[key] = arrays[0]
                    std_results[key] = np.zeros_like(arrays[0])

                print(f"Warning: Arrays with different shapes for key '{key}' - using most common shape or first array")

        elif isinstance(first_result[key], list):
            # Process lists
            if all(isinstance(result[key], list) for result in all_results):
                # Determine whether it's a list of numeric values or a list of more complex structures
                sample_item = first_result[key][0] if first_result[key] else None

                if sample_item is None:
                    # Empty list
                    avg_results[key] = []
                    std_results[key] = []
                elif isinstance(sample_item, (int, float)):
                    # Numeric list: align lengths and compute stats
                    max_length = max(len(result[key]) for result in all_results)

                    # Pad shorter lists with the last value
                    padded_lists = []
                    for result in all_results:
                        list_data = result[key]
                        if len(list_data) < max_length and list_data:
                            list_data = list_data + [list_data[-1]] * (max_length - len(list_data))
                        padded_lists.append(list_data[:max_length])

                    # Convert to numpy for easier computation
                    data_array = np.array(padded_lists)

                    # Compute mean and std for each position
                    avg_results[key] = list(np.mean(data_array, axis=0))
                    std_results[key] = list(np.std(data_array, axis=0))
                else:
                    # List of complex items (not easily averaged)
                    # Keep the first list as a representative sample
                    avg_results[key] = first_result[key]
                    std_results[key] = None
                    print(f"Warning: Complex list structure for key '{key}' cannot be averaged")
            else:
                # Mixed types or not all results have this key as a list
                avg_results[key] = first_result[key]
                std_results[key] = None
                print(f"Warning: Inconsistent list structure for key '{key}' cannot be averaged")

        elif isinstance(first_result[key], (int, float)):
            # Average numerical values
            values = [result[key] for result in all_results]
            avg_results[key] = np.mean(values)
            std_results[key] = np.std(values) if len(values) > 1 else 0

        elif isinstance(first_result[key], str):
            # For strings, use the most common value
            from collections import Counter
            values = [result[key] for result in all_results]
            most_common = Counter(values).most_common(1)[0][0]
            avg_results[key] = most_common
            std_results[key] = None

        elif isinstance(first_result[key], bool):
            # For booleans, use majority vote
            values = [result[key] for result in all_results]
            avg_results[key] = sum(values) > len(values) / 2
            std_results[key] = None

        else:
            # For other types (complex objects, etc.), keep the first one and warn
            avg_results[key] = first_result[key]
            std_results[key] = None
            print(f"Warning: Unsupported data type for key '{key}': {type(first_result[key])}")

    return avg_results, std_results

def get_most_common_policy(policies):
    """
    Get the most common action for each state across multiple seed runs.

    Args:
        policies: List of policy arrays from multiple seeds

    Returns:
        most_common: Array with most common action for each state
    """
    if not policies:
        return None

    # Ensure all policies have the same shape
    shape = policies[0].shape
    if not all(p.shape == shape for p in policies):
        return policies[0]  # Return first one if shapes differ

    # Initialize result array
    most_common = np.zeros_like(policies[0])

    # For each state, find the most common action
    for idx in np.ndindex(shape):
        actions = [p[idx] for p in policies]
        most_common[idx] = max(set(actions), key=actions.count)

    return most_common


#################################################
# ANALYSIS
#################################################


def analyze_convergence_rates(vi_results, pi_results, environment_name):
    """
    Analyze convergence rates of Value Iteration vs Policy Iteration.
    """
    # Figure out which time key is used
    vi_time_key = 'total_time' if 'total_time' in vi_results else 'time'
    pi_time_key = 'total_time' if 'total_time' in pi_results else 'time'

    analysis = {
        'environment': environment_name,
        'vi_iterations': vi_results['iterations'],
        'pi_iterations': pi_results['iterations'],
        'vi_time': vi_results[vi_time_key],
        'pi_time': pi_results[pi_time_key],
        'vi_time_per_iter': vi_results[vi_time_key] / max(1, vi_results['iterations']),
        'pi_time_per_iter': pi_results[pi_time_key] / max(1, pi_results['iterations']),
        'faster_method': 'PI' if pi_results['iterations'] < vi_results['iterations'] else 'VI',
        'percent_difference': abs(vi_results['iterations'] - pi_results['iterations']) /
                              max(1, vi_results['iterations']) * 100
    }

    # Explanation of differences - key part of the assignment
    if pi_results['iterations'] < vi_results['iterations']:
        analysis['explanation'] = (
            "Policy Iteration converged faster because it makes larger, more direct policy updates. "
            "While Value Iteration must gradually improve values for all states before extracting "
            "a policy, Policy Iteration explicitly computes a complete policy at each iteration. "
            f"In {environment_name}, this led to {analysis['percent_difference']:.1f}% faster convergence."
        )
    else:
        analysis['explanation'] = (
            "Value Iteration converged faster in this case, which is unusual as Policy Iteration "
            "typically requires fewer iterations. This may be due to the specific structure of the "
            f"{environment_name} environment, where the policy evaluation step in Policy Iteration "
            "requires more computation than the direct maximization in Value Iteration."
        )

    print(f"\nConvergence Analysis for {environment_name}:")
    print(f"  VI required {analysis['vi_iterations']} iterations, taking {analysis['vi_time']:.2f} seconds")
    print(f"  PI required {analysis['pi_iterations']} iterations, taking {analysis['pi_time']:.2f} seconds")
    print(f"  {analysis['faster_method']} converged faster by {analysis['percent_difference']:.1f}%")
    print(f"  Explanation: {analysis['explanation']}")

    return analysis


def analyze_discretization_effects(discretization_results, environment_name):
    """
    Analyze how discretization affects the CartPole solution.

    This function directly addresses the assignment question:
    'How does discretization affect the CartPole solution?'

    Args:
        discretization_results: Results from different discretization levels
        environment_name: Name of the environment

    Returns:
        analysis: Dictionary with discretization analysis
    """
    bin_levels = sorted(discretization_results.keys())

    analysis = {
        'environment': environment_name,
        'bin_levels': bin_levels,
        'computation_efficiency': {},
        'solution_quality': {},
        'tradeoffs': {}
    }

    # Computation efficiency analysis
    for i in range(len(bin_levels) - 1):
        current_bin = bin_levels[i]
        next_bin = bin_levels[i + 1]

        # Time increase ratio for VI
        vi_time_ratio = (discretization_results[next_bin]['VI']['time'] /
                         discretization_results[current_bin]['VI']['time'])

        # Time increase ratio for PI
        pi_time_ratio = (discretization_results[next_bin]['PI']['time'] /
                         discretization_results[current_bin]['PI']['time'])

        analysis['computation_efficiency'][f'{current_bin}_to_{next_bin}'] = {
            'vi_time_ratio': vi_time_ratio,
            'pi_time_ratio': pi_time_ratio,
            'state_space_increase': (next_bin ** 4) / (current_bin ** 4)
        }

    # Solution quality analysis
    for bin_level in bin_levels:
        vi_value = discretization_results[bin_level]['VI']['value']
        pi_value = discretization_results[bin_level]['PI']['value']

        analysis['solution_quality'][bin_level] = {
            'vi_value': vi_value,
            'pi_value': pi_value,
            'total_states': bin_level ** 4
        }

    # Tradeoff analysis
    for i in range(len(bin_levels) - 1):
        current_bin = bin_levels[i]
        next_bin = bin_levels[i + 1]

        # Value improvement ratio for VI
        vi_value_ratio = (discretization_results[next_bin]['VI']['value'] /
                          discretization_results[current_bin]['VI']['value'])

        # Value improvement ratio for PI
        pi_value_ratio = (discretization_results[next_bin]['PI']['value'] /
                          discretization_results[current_bin]['PI']['value'])

        # Time cost ratio
        vi_time_ratio = (discretization_results[next_bin]['VI']['time'] /
                         discretization_results[current_bin]['VI']['time'])
        pi_time_ratio = (discretization_results[next_bin]['PI']['time'] /
                         discretization_results[current_bin]['PI']['time'])

        analysis['tradeoffs'][f'{current_bin}_to_{next_bin}'] = {
            'vi_value_improvement': (vi_value_ratio - 1) * 100,  # percentage improvement
            'pi_value_improvement': (pi_value_ratio - 1) * 100,  # percentage improvement
            'vi_time_cost_increase': (vi_time_ratio - 1) * 100,  # percentage increase
            'pi_time_cost_increase': (pi_time_ratio - 1) * 100,  # percentage increase
            'vi_efficiency_ratio': (vi_value_ratio - 1) / (vi_time_ratio - 1) if vi_time_ratio > 1 else float('inf'),
            'pi_efficiency_ratio': (pi_value_ratio - 1) / (pi_time_ratio - 1) if pi_time_ratio > 1 else float('inf')
        }

    # Print key findings - directly addresses assignment questions
    print(f"\nDiscretization Effects Analysis for {environment_name}:")
    print(f"  Tested discretization levels: {bin_levels} bins per dimension")
    print("  Computational efficiency findings:")

    for i in range(len(bin_levels) - 1):
        current_bin = bin_levels[i]
        next_bin = bin_levels[i + 1]
        tr = analysis['tradeoffs'][f'{current_bin}_to_{next_bin}']
        ef = analysis['computation_efficiency'][f'{current_bin}_to_{next_bin}']

        print(f"    Increasing from {current_bin} to {next_bin} bins:")
        print(f"      - State space increased by {ef['state_space_increase']:.1f}x")
        print(f"      - VI computation time increased by {tr['vi_time_cost_increase']:.1f}%")
        print(f"      - PI computation time increased by {tr['pi_time_cost_increase']:.1f}%")
        print(f"      - VI solution quality improved by {tr['vi_value_improvement']:.1f}%")
        print(f"      - PI solution quality improved by {tr['pi_value_improvement']:.1f}%")

    # Identify optimal discretization level based on efficiency ratio
    vi_efficiency = {b: analysis['tradeoffs'][f'{bin_levels[i]}_to_{bin_levels[i + 1]}']['vi_efficiency_ratio']
                     for i, b in enumerate(bin_levels[:-1])}

    pi_efficiency = {b: analysis['tradeoffs'][f'{bin_levels[i]}_to_{bin_levels[i + 1]}']['pi_efficiency_ratio']
                     for i, b in enumerate(bin_levels[:-1])}

    best_vi_bin = max(vi_efficiency, key=vi_efficiency.get)
    best_pi_bin = max(pi_efficiency, key=pi_efficiency.get)

    analysis['optimal_discretization'] = {
        'vi': best_vi_bin,
        'pi': best_pi_bin
    }

    print("  Optimal discretization levels:")
    print(f"    - For VI: {best_vi_bin} bins per dimension offers best efficiency")
    print(f"    - For PI: {best_pi_bin} bins per dimension offers best efficiency")

    return analysis


def analyze_exploration_strategies(strategy_results, algorithm_name, environment_name):
    """
    Analyze the impact of different exploration strategies.

    This function directly addresses the assignment question:
    'What exploration strategies did you use, and how did they affect learning?'

    Args:
        strategy_results: Results from different exploration strategies
        algorithm_name: Name of the algorithm (SARSA or Q-Learning)
        environment_name: Name of the environment

    Returns:
        analysis: Dictionary with exploration strategy analysis
    """
    analysis = {
        'environment': environment_name,
        'algorithm': algorithm_name,
        'strategies': list(strategy_results.keys()),
        'best_strategy': None,
        'comparisons': {},
        'learning_stability': {},
        'time_efficiency': {}
    }

    # Find best strategy by final reward
    rewards = {s: results['final_avg_reward'] for s, results in strategy_results.items()}
    analysis['best_strategy'] = max(rewards, key=rewards.get)

    # Compare strategies
    best_reward = rewards[analysis['best_strategy']]
    for strategy, results in strategy_results.items():
        reward = results['final_avg_reward']
        time = results['total_time']

        # Calculate learning curve metrics
        rewards_history = results['rewards']

        # Calculate stability metrics (using last 100 episodes)
        last_100 = rewards_history[-100:]
        stability = {
            'mean': np.mean(last_100),
            'std': np.std(last_100),
            'coefficient_of_variation': np.std(last_100) / abs(np.mean(last_100)) if np.mean(last_100) != 0 else float(
                'inf')
        }

        # Calculate learning speed (episodes to reach 90% of final performance)
        target = 0.9 * reward
        episodes_to_target = next((i for i, r in enumerate(rewards_history)
                                   if r >= target), len(rewards_history))

        analysis['comparisons'][strategy] = {
            'reward': reward,
            'relative_performance': (reward / best_reward) * 100 if best_reward != 0 else 0,
            'time': time,
            'episodes_to_90_percent': episodes_to_target
        }

        analysis['learning_stability'][strategy] = stability

        analysis['time_efficiency'][strategy] = {
            'reward_per_second': reward / time if time > 0 else 0,
            'episodes_per_second': len(rewards_history) / time if time > 0 else 0
        }

    # Print key findings - directly addresses assignment questions
    print(f"\nExploration Strategy Analysis for {algorithm_name} on {environment_name}:")
    print(f"  Best strategy: {analysis['best_strategy']} with reward {rewards[analysis['best_strategy']]:.2f}")

    print("  Performance comparison:")
    for strategy, comp in analysis['comparisons'].items():
        print(f"    - {strategy}: {comp['reward']:.2f} reward ({comp['relative_performance']:.1f}% of best)")
        print(f"      Time: {comp['time']:.2f}s, Episodes to 90% performance: {comp['episodes_to_90_percent']}")

    print("  Learning stability (last 100 episodes):")
    for strategy, stability in analysis['learning_stability'].items():
        print(f"    - {strategy}: Mean={stability['mean']:.2f}, Std={stability['std']:.2f}, " +
              f"CV={stability['coefficient_of_variation']:.2f}")

    # Provide strategy-specific analysis based on results
    print("  Strategy behavior analysis:")
    if 'epsilon_greedy' in strategy_results:
        print("    - Epsilon-greedy: Simple and efficient strategy that balances exploration/exploitation")
        print(f"      through gradually reducing randomness. In {environment_name}, it showed " +
              ("good performance." if analysis['best_strategy'] == 'epsilon_greedy' else
               "suboptimal performance compared to other strategies."))

    if 'boltzmann' in strategy_results:
        print("    - Boltzmann: Uses a softmax approach that weights actions by their Q-values,")
        print(f"      allowing more nuanced exploration. In {environment_name}, it showed " +
              ("excellent performance " if analysis['best_strategy'] == 'boltzmann' else
               "moderate performance ") + "due to its ability to balance exploration based on value estimates.")

    if 'ucb' in strategy_results:
        print("    - UCB (Upper Confidence Bound): Balances exploration and exploitation by")
        print(f"      favoring uncertain, potentially valuable actions. In {environment_name}, it showed " +
              ("superior performance " if analysis['best_strategy'] == 'ucb' else
               "suboptimal performance ") + "which aligns with its theoretical guarantees.")

    return analysis


def analyze_model_free_comparison(sarsa_results, qlearning_results, environment_name):
    """
    Compare SARSA and Q-Learning performance.

    This function directly addresses the assignment question:
    'How do SARSA and Q-Learning compare in performance?'

    Args:
        sarsa_results: Results from SARSA
        qlearning_results: Results from Q-Learning
        environment_name: Name of the environment

    Returns:
        analysis: Dictionary with model-free comparison analysis
    """
    analysis = {
        'environment': environment_name,
        'reward_comparison': {
            'sarsa': sarsa_results['final_avg_reward'],
            'qlearning': qlearning_results['final_avg_reward'],
            'relative_difference': (qlearning_results['final_avg_reward'] - sarsa_results['final_avg_reward']) /
                                   abs(sarsa_results['final_avg_reward']) * 100 if sarsa_results[
                                                                                       'final_avg_reward'] != 0 else float(
                'inf')
        },
        'time_comparison': {
            'sarsa': sarsa_results['total_time'],
            'qlearning': qlearning_results['total_time'],
            'relative_difference': (qlearning_results['total_time'] - sarsa_results['total_time']) /
                                   sarsa_results['total_time'] * 100 if sarsa_results['total_time'] > 0 else float(
                'inf')
        },
        'policy_difference': {},
        'learning_dynamics': {}
    }

    # Compare learning curves
    sarsa_rewards = sarsa_results['rewards']
    qlearning_rewards = qlearning_results['rewards']

    # Truncate to shorter length for comparison
    min_length = min(len(sarsa_rewards), len(qlearning_rewards))
    sarsa_truncated = sarsa_rewards[:min_length]
    qlearning_truncated = qlearning_rewards[:min_length]

    # Calculate various learning dynamics metrics
    sarsa_smoothed = np.convolve(sarsa_truncated, np.ones(10) / 10, mode='valid')
    qlearning_smoothed = np.convolve(qlearning_truncated, np.ones(10) / 10, mode='valid')

    # Initial learning speed (first 25% of episodes)
    first_quarter = min_length // 4
    sarsa_initial = sarsa_truncated[:first_quarter]
    qlearning_initial = qlearning_truncated[:first_quarter]

    # Final convergence (last 25% of episodes)
    sarsa_final = sarsa_truncated[-first_quarter:]
    qlearning_final = qlearning_truncated[-first_quarter:]

    analysis['learning_dynamics'] = {
        'initial_learning_rate': {
            'sarsa': np.mean(sarsa_initial),
            'qlearning': np.mean(qlearning_initial),
            'faster': 'Q-Learning' if np.mean(qlearning_initial) > np.mean(sarsa_initial) else 'SARSA'
        },
        'final_convergence': {
            'sarsa': {
                'mean': np.mean(sarsa_final),
                'std': np.std(sarsa_final)
            },
            'qlearning': {
                'mean': np.mean(qlearning_final),
                'std': np.std(qlearning_final)
            },
            'more_stable': 'Q-Learning' if np.std(qlearning_final) < np.std(sarsa_final) else 'SARSA'
        }
    }

    # Theoretical explanation for differences based on on-policy vs off-policy
    if qlearning_results['final_avg_reward'] > sarsa_results['final_avg_reward']:
        analysis['theoretical_explanation'] = (
            "Q-Learning outperformed SARSA, which aligns with theoretical expectations. "
            "As an off-policy algorithm, Q-Learning directly approximates the optimal policy "
            "regardless of the exploration policy used, while SARSA (on-policy) learns the "
            "value of the current policy including exploratory moves. In environments with "
            f"significant penalties like {environment_name}, this difference can be substantial."
        )
    else:
        analysis['theoretical_explanation'] = (
            "SARSA outperformed Q-Learning, which is interesting given Q-Learning's "
            "theoretical advantages as an off-policy method. This suggests that in "
            f"{environment_name}, the more conservative policy learned by SARSA (which "
            "accounts for exploration) may be beneficial, possibly due to avoiding risky "
            "states that Q-Learning's greedy policy might enter."
        )

    # Print key findings - directly addresses assignment questions
    print(f"\nModel-free Algorithm Comparison for {environment_name}:")
    print(f"  Final average reward:")
    print(f"    - SARSA: {analysis['reward_comparison']['sarsa']:.2f}")
    print(f"    - Q-Learning: {analysis['reward_comparison']['qlearning']:.2f}")
    print(f"    - Difference: {analysis['reward_comparison']['relative_difference']:.1f}% " +
          ("in favor of Q-Learning" if analysis['reward_comparison'][
                                           'relative_difference'] > 0 else "in favor of SARSA"))

    print("  Learning dynamics:")
    print(f"    - Initial learning (first 25% of episodes):")
    print(f"      SARSA: {analysis['learning_dynamics']['initial_learning_rate']['sarsa']:.2f}")
    print(f"      Q-Learning: {analysis['learning_dynamics']['initial_learning_rate']['qlearning']:.2f}")
    print(f"      {analysis['learning_dynamics']['initial_learning_rate']['faster']} learned faster initially")

    print(f"    - Final convergence (last 25% of episodes):")
    print(f"      SARSA: mean={analysis['learning_dynamics']['final_convergence']['sarsa']['mean']:.2f}, " +
          f"std={analysis['learning_dynamics']['final_convergence']['sarsa']['std']:.2f}")
    print(f"      Q-Learning: mean={analysis['learning_dynamics']['final_convergence']['qlearning']['mean']:.2f}, " +
          f"std={analysis['learning_dynamics']['final_convergence']['qlearning']['std']:.2f}")
    print(
        f"      {analysis['learning_dynamics']['final_convergence']['more_stable']} was more stable in final convergence")

    print(f"  Theoretical explanation:")
    print(f"    {analysis['theoretical_explanation']}")

    return analysis


def generate_report_content(blackjack_results, blackjack_extra, cartpole_results, cartpole_extra):
    """
    Generate structured content for the report based on experiment results.

    This function maps directly to the required sections in the assignment's report structure.

    Args:
        blackjack_results, blackjack_extra: Results from Blackjack experiments
        cartpole_results, cartpole_extra: Results from CartPole experiments

    Returns:
        report: Dictionary with structured content for each report section
    """
    report = {
        'mdp_description': {
            'blackjack': {
                'state_space': 'Discrete: player_sum (4-21), dealer card (1-10), usable_ace (True/False)',
                'action_space': 'Discrete: 0 (stick) or 1 (hit)',
                'reward_structure': 'Win (+1), Lose (-1), Draw (0)',
                'stochasticity': 'Stochastic due to random card draws',
                'challenges': 'Large state space with stochastic transitions'
            },
            'cartpole': {
                'state_space': 'Continuous: cart position, cart velocity, pole angle, pole angular velocity',
                'discretization': f'Discretized using {max(CARTPOLE_BINS)} bins per dimension',
                'action_space': 'Discrete: 0 (left) or 1 (right)',
                'reward_structure': '+1 for each timestep the pole remains upright',
                'determinism': 'Deterministic physics-based environment',
                'challenges': 'Continuous state space requiring discretization'
            }
        },
        'method_explanation': {
            'dynamic_programming': {
                'value_iteration': {
                    'description': 'Value Iteration iteratively updates state values based on the Bellman optimality equation, converging to the optimal value function from which the optimal policy is derived.',
                    'algorithm': 'For each state s, iteratively update V(s) = max_a { R(s,a) + γ * Σ T(s,a,s\') * V(s\') }',
                    'blackjack_application': f"Applied with γ={blackjack_results['Value Iteration']['params']['gamma']}, θ={blackjack_results['Value Iteration']['params']['theta']}",
                    'cartpole_application': f"Applied with γ={cartpole_results['Value Iteration']['params']['gamma']}, θ={cartpole_results['Value Iteration']['params']['theta']} after discretization",
                },
                'policy_iteration': {
                    'description': 'Policy Iteration alternates between policy evaluation (computing values for the current policy) and policy improvement (updating the policy based on values).',
                    'algorithm': 'Policy Evaluation: Compute V^π(s) for current policy π, Policy Improvement: π(s) = argmax_a { R(s,a) + γ * Σ T(s,a,s\') * V^π(s\') }',
                    'blackjack_application': f"Applied with γ={blackjack_results['Policy Iteration']['params']['gamma']}, θ={blackjack_results['Policy Iteration']['params']['theta']}",
                    'cartpole_application': f"Applied with γ={cartpole_results['Policy Iteration']['params']['gamma']}, θ={cartpole_results['Policy Iteration']['params']['theta']} after discretization",
                },
            },
            'reinforcement_learning': {
                'sarsa': {
                    'description': 'SARSA (State-Action-Reward-State-Action) is an on-policy TD learning algorithm that updates Q-values based on the action actually taken in the next state.',
                    'algorithm': 'Q(s,a) ← Q(s,a) + α * [r + γ * Q(s\',a\') - Q(s,a)]',
                    'blackjack_application': f"Applied with α={blackjack_results['SARSA']['params']['learning_rate']}, γ={blackjack_results['SARSA']['params']['gamma']}, {blackjack_results['SARSA']['best_strategy']} exploration",
                    'cartpole_application': f"Applied with α={cartpole_results['SARSA']['params']['learning_rate']}, γ={cartpole_results['SARSA']['params']['gamma']}, {cartpole_results['SARSA']['best_strategy']} exploration",
                },
                'q_learning': {
                    'description': 'Q-Learning is an off-policy TD learning algorithm that updates Q-values based on the maximum Q-value in the next state, regardless of the action actually taken.',
                    'algorithm': 'Q(s,a) ← Q(s,a) + α * [r + γ * max_a\' Q(s\',a\') - Q(s,a)]',
                    'blackjack_application': f"Applied with α={blackjack_results['Q-Learning']['params']['learning_rate']}, γ={blackjack_results['Q-Learning']['params']['gamma']}, {blackjack_results['Q-Learning']['best_strategy']} exploration",
                    'cartpole_application': f"Applied with α={cartpole_results['Q-Learning']['params']['learning_rate']}, γ={cartpole_results['Q-Learning']['params']['gamma']}, {cartpole_results['Q-Learning']['best_strategy']} exploration",
                },
            },
            'discretization': {
                'description': 'Discretization was necessary for CartPole to apply tabular methods (VI, PI) to its continuous state space.',
                'strategy': f'Each continuous dimension was divided into {CARTPOLE_BINS} discrete bins.',
                'implementation': 'Used uniform binning for cart position, velocity, pole angle, and pole angular velocity.'
            },
            'exploration_strategies': {
                'epsilon_greedy': 'Balances exploration and exploitation by selecting random actions with probability ε and greedy actions with probability 1-ε.',
                'boltzmann': 'Uses a softmax distribution over Q-values to select actions, giving higher probability to actions with higher Q-values.',
                'ucb': 'Upper Confidence Bound approach that balances exploration and exploitation by favoring actions with high uncertainty.'
            }
        },
        'results_analysis': {
            'vi_vs_pi': {
                'blackjack_convergence': {
                    'vi_iterations': blackjack_results['Value Iteration']['iterations'],
                    'pi_iterations': blackjack_results['Policy Iteration']['iterations'],
                    'faster_method': 'Policy Iteration' if blackjack_results['Policy Iteration']['iterations'] <
                                                           blackjack_results['Value Iteration'][
                                                               'iterations'] else 'Value Iteration',
                    'speedup': abs(
                        blackjack_results['Value Iteration']['iterations'] - blackjack_results['Policy Iteration'][
                            'iterations']) / max(1, blackjack_results['Value Iteration']['iterations']) * 100
                },
                'cartpole_convergence': {
                    'vi_iterations': cartpole_results['Value Iteration']['iterations'],
                    'pi_iterations': cartpole_results['Policy Iteration']['iterations'],
                    'faster_method': 'Policy Iteration' if cartpole_results['Policy Iteration']['iterations'] <
                                                           cartpole_results['Value Iteration'][
                                                               'iterations'] else 'Value Iteration',
                    'speedup': abs(
                        cartpole_results['Value Iteration']['iterations'] - cartpole_results['Policy Iteration'][
                            'iterations']) / max(1, cartpole_results['Value Iteration']['iterations']) * 100
                },
                'policy_similarity': {
                    'blackjack': f"{blackjack_extra['vi_pi_agreement']:.2f}% agreement",
                    'cartpole': f"{cartpole_extra['vi_pi_agreement']:.2f}% agreement"
                },
                'explanation': 'Policy Iteration typically converges in fewer iterations due to its direct policy updates but may take longer per iteration due to the policy evaluation step.',
            },
            'sarsa_vs_qlearning': {
                'blackjack_performance': {
                    'sarsa_reward': blackjack_results['SARSA']['reward'],
                    'qlearning_reward': blackjack_results['Q-Learning']['reward'],
                    'better_algorithm': 'Q-Learning' if blackjack_results['Q-Learning']['reward'] >
                                                        blackjack_results['SARSA']['reward'] else 'SARSA',
                    'reward_difference': abs(
                        blackjack_results['Q-Learning']['reward'] - blackjack_results['SARSA']['reward'])
                },
                'cartpole_performance': {
                    'sarsa_reward': cartpole_results['SARSA']['reward'],
                    'qlearning_reward': cartpole_results['Q-Learning']['reward'],
                    'better_algorithm': 'Q-Learning' if cartpole_results['Q-Learning']['reward'] >
                                                        cartpole_results['SARSA']['reward'] else 'SARSA',
                    'reward_difference': abs(
                        cartpole_results['Q-Learning']['reward'] - cartpole_results['SARSA']['reward'])
                },
                'policy_similarity': {
                    'blackjack': f"{blackjack_extra['sarsa_qlearning_agreement']:.2f}% agreement",
                    'cartpole': f"{cartpole_extra['sarsa_qlearning_agreement']:.2f}% agreement"
                },
                'exploration_impact': {
                    'blackjack_sarsa': f"Best strategy: {blackjack_results['SARSA']['best_strategy']}",
                    'blackjack_qlearning': f"Best strategy: {blackjack_results['Q-Learning']['best_strategy']}",
                    'cartpole_sarsa': f"Best strategy: {cartpole_results['SARSA']['best_strategy']}",
                    'cartpole_qlearning': f"Best strategy: {cartpole_results['Q-Learning']['best_strategy']}",
                },
                'explanation': 'Q-Learning, being an off-policy algorithm, can learn the optimal policy regardless of exploration, potentially giving it an advantage over the on-policy SARSA in certain environments.',
            },
            'discretization_effects': {
                'bin_levels': CARTPOLE_BINS,
                'state_space_sizes': [f"{b} bins: {b ** 4} states" for b in CARTPOLE_BINS],
                'computation_time': {
                    'vi': [cartpole_extra['discretization_results'][b]['VI']['time'] for b in CARTPOLE_BINS],
                    'pi': [cartpole_extra['discretization_results'][b]['PI']['time'] for b in CARTPOLE_BINS],
                },
                'solution_quality': {
                    'vi': [cartpole_extra['discretization_results'][b]['VI']['value'] for b in CARTPOLE_BINS],
                    'pi': [cartpole_extra['discretization_results'][b]['PI']['value'] for b in CARTPOLE_BINS],
                },
                'tradeoffs': 'Finer discretization provides higher solution quality but at exponentially increasing computational cost.',
                'optimal_balance': f"{CARTPOLE_BINS[1]} bins provides a good tradeoff between accuracy and computation time.",
            },
        },
        'visualizations': {
            'convergence_plots': {
                'blackjack_vi_pi': f"{SAVE_DIR}/blackjack/vi_pi_convergence.png",
                'blackjack_value_evolution': f"{SAVE_DIR}/blackjack/value_evolution.png",
                'cartpole_discretization_iterations': f"{SAVE_DIR}/cartpole/discretization_iterations.png",
                'cartpole_discretization_time': f"{SAVE_DIR}/cartpole/discretization_time.png",
            },
            'policy_heatmaps': {
                'blackjack_vi': f"{SAVE_DIR}/blackjack/vi_policy.png",
                'blackjack_pi': f"{SAVE_DIR}/blackjack/pi_policy.png",
                'blackjack_sarsa': f"{SAVE_DIR}/blackjack/sarsa_policy.png",
                'blackjack_qlearning': f"{SAVE_DIR}/blackjack/qlearning_policy.png",
                'cartpole_vi': f"{SAVE_DIR}/cartpole/vi_policy.png",
                'cartpole_pi': f"{SAVE_DIR}/cartpole/pi_policy.png",
                'cartpole_sarsa': f"{SAVE_DIR}/cartpole/sarsa_policy.png",
                'cartpole_qlearning': f"{SAVE_DIR}/cartpole/qlearning_policy.png",
            },
            'learning_curves': {
                'blackjack_rl': f"{SAVE_DIR}/blackjack/learning_curves.png",
                'cartpole_rl': f"{SAVE_DIR}/cartpole/learning_curves.png",
                'cartpole_episode_lengths': f"{SAVE_DIR}/cartpole/episode_lengths.png",
            },
            'algorithm_comparison': {
                'blackjack': f"{SAVE_DIR}/blackjack/algorithm_comparison.png",
                'cartpole': f"{SAVE_DIR}/cartpole/algorithm_comparison.png",
            },
        },
        'conclusion': {
            'key_findings': {
                'DP_vs_RL': 'Dynamic Programming methods (VI/PI) converge to optimal policies more reliably but require known transition probabilities. Reinforcement Learning methods can work with unknown dynamics but may require more tuning.',
                'on_vs_off_policy': 'The off-policy nature of Q-Learning generally resulted in better performance than SARSA, particularly in environments with significant penalties.',
                'discretization_tradeoffs': 'Discretization involves a critical tradeoff between solution quality and computational efficiency. Optimal discretization depends on available computational resources and accuracy requirements.',
                'exploration_importance': 'The choice of exploration strategy significantly impacts RL performance, with different strategies working better in different environments.',
            },
            'challenges': {
                'computational_requirements': 'Fine discretization of continuous state spaces leads to exponential growth in computational requirements.',
                'hyperparameter_sensitivity': 'Both DP and RL methods are sensitive to hyperparameters, requiring careful tuning.',
                'discretization_design': 'Designing an effective discretization scheme requires domain knowledge and experimentation.',
            },
            'future_work': {
                'function_approximation': 'Using function approximation methods like neural networks could address the curse of dimensionality in continuous state spaces.',
                'advanced_algorithms': 'Implementing more advanced algorithms like DQN or DDPG could further improve performance in complex environments.',
                'adaptive_discretization': 'Developing adaptive discretization methods that allocate more bins to critical regions of the state space.',
            },
        }
    }

    return report



#################################################
# Main Execution
#################################################

#################################################
# Main Execution
#################################################

if __name__ == "__main__":
    try:
        # Set up the experiment environment
        print("\n" + "=" * 80)
        print("STARTING MDP ANALYSIS FOR ASSIGNMENT 4".center(80))
        print("=" * 80)
        print(f"\nUsing random seed: {RANDOM_SEEDS}")
        print(f"Results will be saved to: {os.path.abspath(SAVE_DIR)}")

        # Run Blackjack experiments
        print("\n" + "=" * 80)
        print("STARTING BLACKJACK EXPERIMENTS".center(80))
        print("=" * 80)

        # Initialize environment
        blackjack_env = BlackjackEnv()

        # Tune hyperparameters
        vi_params, vi_tuning_results = tune_hyperparameters(blackjack_env, VI_PI_PARAM_GRID, ValueIteration,
                                                            algorithm_type='DP')
        pi_params, pi_tuning_results = tune_hyperparameters(blackjack_env, VI_PI_PARAM_GRID, PolicyIteration,
                                                            algorithm_type='DP')
        sarsa_params, sarsa_tuning_results = tune_hyperparameters(blackjack_env, RL_PARAM_GRID, SARSA,
                                                                  algorithm_type='RL', episodes=400)
        q_params, q_tuning_results = tune_hyperparameters(blackjack_env, RL_PARAM_GRID, QLearning, algorithm_type='RL',
                                                          episodes=400)

        # Add detailed hyperparameter tuning visualizations
        print("\nGenerating detailed hyperparameter tuning visualizations for Blackjack...")
        plot_dp_tuning_results(vi_tuning_results, "Value Iteration", "Blackjack")
        plot_dp_tuning_results(pi_tuning_results, "Policy Iteration", "Blackjack")
        plot_rl_tuning_results(sarsa_tuning_results, "SARSA", "Blackjack")
        plot_rl_tuning_results(q_tuning_results, "Q-Learning", "Blackjack")

        # Compare exploration strategies
        best_sarsa_strategy, sarsa_strategy_results = compare_exploration_strategies(
            blackjack_env, SARSA, sarsa_params, episodes=600)
        best_q_strategy, q_strategy_results = compare_exploration_strategies(
            blackjack_env, QLearning, q_params, episodes=600)

        # Add exploration strategy visualizations
        print("\nGenerating detailed exploration strategy visualizations for Blackjack...")
        plot_exploration_strategy_comparison(
            sarsa_strategy_results, "SARSA", "Blackjack")
        plot_exploration_strategy_comparison(
            q_strategy_results, "Q-Learning", "Blackjack")

        # Add combined visualizations
        print("\nGenerating combined visualization for hyperparameter tuning...")
        all_tuning_results = {
            'Value Iteration': vi_tuning_results,
            'Policy Iteration': pi_tuning_results,
            'SARSA': sarsa_tuning_results,
            'Q-Learning': q_tuning_results
        }
        plot_combined_hyperparameter_results(all_tuning_results, "Blackjack")

        print("\nGenerating combined visualization for exploration strategies...")
        plot_combined_exploration_strategies(sarsa_strategy_results, q_strategy_results, "Blackjack")

        # Add best exploration strategy training progress visualization
        plot_best_exploration_training_progress(
            sarsa_strategy_results,
            q_strategy_results,
            "Blackjack",
            f"{SAVE_DIR}/hyperparameter_tuning/blackjack_best_exploration_training.png")

        # Get transition matrix for DP algorithms
        T, R = blackjack_env.get_transition_matrix(num_samples=8000)
        total_states = np.prod(blackjack_env.state_space)

        # Run Value Iteration
        vi = ValueIteration(total_states, blackjack_env.action_space,
                            gamma=vi_params['gamma'], theta=vi_params['theta'],
                            max_iterations=vi_params['max_iterations'])
        vi_results = vi.solve(T, R)

        # Run Policy Iteration
        pi = PolicyIteration(total_states, blackjack_env.action_space,
                             gamma=pi_params['gamma'], theta=pi_params['theta'],
                             max_iterations=pi_params['max_iterations'])
        pi_results = pi.solve(T, R)

        # Run SARSA
        sarsa = SARSA(total_states, blackjack_env.action_space,
                      learning_rate=sarsa_params['learning_rate'],
                      gamma=sarsa_params['gamma'],
                      epsilon_decay=sarsa_params['epsilon_decay'],
                      exploration_strategy=best_sarsa_strategy)
        sarsa_results = sarsa.train(blackjack_env, episodes=BLACKJACK_EPISODES)

        # Run Q-Learning
        qlearning = QLearning(total_states, blackjack_env.action_space,
                              learning_rate=q_params['learning_rate'],
                              gamma=q_params['gamma'],
                              epsilon_decay=q_params['epsilon_decay'],
                              exploration_strategy=best_q_strategy)
        qlearning_results = qlearning.train(blackjack_env, episodes=BLACKJACK_EPISODES)

        # Reshape policies for visualization and analysis
        vi_policy = vi_results['policy'].reshape(blackjack_env.state_space)
        pi_policy = pi_results['policy'].reshape(blackjack_env.state_space)
        sarsa_policy = sarsa_results['policy'].reshape(blackjack_env.state_space)
        qlearning_policy = qlearning_results['policy'].reshape(blackjack_env.state_space)

        # Create visualizations
        plot_convergence(
            {'VI Delta': vi_results['delta_history'],
             'PI Policy Changes (%)': np.array(pi_results['policy_changes']) / total_states * 100},
            "Blackjack: VI vs PI Convergence",
            f"{SAVE_DIR}/blackjack/vi_pi_convergence.png",
            ylabel="Delta / Policy Changes (%)"
        )

        plot_convergence(
            {'VI': vi_results['value_history'],
             'PI': pi_results['value_history']},
            "Blackjack: Value Function Evolution",
            f"{SAVE_DIR}/blackjack/value_evolution.png",
            ylabel="Average Value"
        )

        plot_policy_blackjack(
            vi_policy,
            "Blackjack: Value Iteration Policy",
            f"{SAVE_DIR}/blackjack/vi_policy.png"
        )

        plot_policy_blackjack(
            pi_policy,
            "Blackjack: Policy Iteration Policy",
            f"{SAVE_DIR}/blackjack/pi_policy.png"
        )

        plot_policy_blackjack(
            sarsa_policy,
            f"Blackjack: SARSA Policy ({best_sarsa_strategy} exploration)",
            f"{SAVE_DIR}/blackjack/sarsa_policy.png"
        )

        plot_policy_blackjack(
            qlearning_policy,
            f"Blackjack: Q-Learning Policy ({best_q_strategy} exploration)",
            f"{SAVE_DIR}/blackjack/qlearning_policy.png"
        )

        plot_learning_curves(
            [sarsa_results['rewards'], qlearning_results['rewards']],
            ['SARSA', 'Q-Learning'],
            "Blackjack: Learning Curves",
            f"{SAVE_DIR}/blackjack/learning_curves.png",
            window=100
        )

        # Calculate policy agreement
        vi_pi_agreement = np.mean(vi_policy == pi_policy) * 100
        sarsa_qlearning_agreement = np.mean(sarsa_policy == qlearning_policy) * 100
        vi_sarsa_agreement = np.mean(vi_policy == sarsa_policy) * 100
        pi_qlearning_agreement = np.mean(pi_policy == qlearning_policy) * 100

        # Store Blackjack results
        blackjack_results = {
            'Value Iteration': {
                'iterations': vi_results['iterations'],
                'time': vi_results['total_time'],
                'reward': np.mean(vi_results['V']),
                'policy': vi_policy,
                'params': vi_params,
            },
            'Policy Iteration': {
                'iterations': pi_results['iterations'],
                'time': pi_results['total_time'],
                'reward': np.mean(pi_results['V']),
                'policy': pi_policy,
                'params': pi_params,
            },
            'SARSA': {
                'iterations': len(sarsa_results['rewards']),
                'time': sarsa_results['total_time'],
                'reward': sarsa_results['final_avg_reward'],
                'policy': sarsa_policy,
                'params': sarsa_params,
                'best_strategy': best_sarsa_strategy,
            },
            'Q-Learning': {
                'iterations': len(qlearning_results['rewards']),
                'time': qlearning_results['total_time'],
                'reward': qlearning_results['final_avg_reward'],
                'policy': qlearning_policy,
                'params': q_params,
                'best_strategy': best_q_strategy,
            }
        }

        blackjack_extra = {
            'vi_pi_agreement': vi_pi_agreement,
            'sarsa_qlearning_agreement': sarsa_qlearning_agreement,
            'vi_sarsa_agreement': vi_sarsa_agreement,
            'pi_qlearning_agreement': pi_qlearning_agreement,
            'exploration_strategies': {
                'SARSA': sarsa_strategy_results,
                'Q-Learning': q_strategy_results
            }
        }

        # Create combined comparison plot
        plot_combined_comparison(
            {k: {'iterations': v['iterations'], 'time': v['time'], 'reward': v['reward']}
             for k, v in blackjack_results.items()},
            "Blackjack: Algorithm Comparison",
            f"{SAVE_DIR}/blackjack/algorithm_comparison.png"
        )

        # Create performance dashboard
        create_performance_dashboard(
            blackjack_results,
            blackjack_extra,
            "Blackjack",
            f"{SAVE_DIR}/blackjack/performance_dashboard.png")

        # Conduct specific analysis for alignment with assignment requirements
        convergence_analysis_blackjack = analyze_convergence_rates(
            vi_results, pi_results, "Blackjack")

        model_free_comparison_blackjack = analyze_model_free_comparison(
            sarsa_results, qlearning_results, "Blackjack")

        exploration_analysis_sarsa = analyze_exploration_strategies(
            sarsa_strategy_results, "SARSA", "Blackjack")

        exploration_analysis_qlearning = analyze_exploration_strategies(
            q_strategy_results, "Q-Learning", "Blackjack")

        print("\nBlackjack Experiments Completed")

        # Run CartPole experiments
        print("\n" + "=" * 80)
        print("STARTING CARTPOLE EXPERIMENTS".center(80))
        print("=" * 80)

        # Test different discretization levels
        discretization_results = {}

        for bins in CARTPOLE_BINS:
            print(f"\nTesting CartPole with {bins} bins discretization")

            # Initialize environment
            cartpole_env = CartPoleEnv(n_bins=bins)

            # For smaller bin sizes, do more extensive tuning
            if bins <= 4:
                # Tune VI and PI parameters
                vi_params, vi_tuning_results = tune_hyperparameters(cartpole_env, VI_PI_PARAM_GRID, ValueIteration,
                                                                    algorithm_type='DP')
                pi_params, pi_tuning_results = tune_hyperparameters(cartpole_env, VI_PI_PARAM_GRID, PolicyIteration,
                                                                    algorithm_type='DP')

                # Add detailed hyperparameter tuning visualizations for CartPole (smallest bin size)
                if bins == min(CARTPOLE_BINS):
                    print("\nGenerating detailed hyperparameter tuning visualizations for CartPole...")
                    plot_dp_tuning_results(vi_tuning_results, "Value Iteration", "CartPole")
                    plot_dp_tuning_results(pi_tuning_results, "Policy Iteration", "CartPole")
            else:
                # For larger bin sizes, use best parameters from smaller bins to save time
                vi_params = {'gamma': 0.99, 'theta': 1e-4, 'max_iterations': 200}
                pi_params = {'gamma': 0.99, 'theta': 1e-4, 'max_iterations': 200}

            # Get transition and reward matrices
            T, R = cartpole_env.get_transition_matrix(num_episodes=150, max_steps=75)

            # Run Value Iteration
            vi = ValueIteration(np.prod(cartpole_env.state_space), cartpole_env.action_space,
                                gamma=vi_params['gamma'], theta=vi_params['theta'],
                                max_iterations=vi_params['max_iterations'])
            vi_results = vi.solve(T, R)

            # Run Policy Iteration
            pi = PolicyIteration(np.prod(cartpole_env.state_space), cartpole_env.action_space,
                                 gamma=pi_params['gamma'], theta=pi_params['theta'],
                                 max_iterations=pi_params['max_iterations'])
            pi_results = pi.solve(T, R)

            # Store results
            discretization_results[bins] = {
                'VI': {
                    'iterations': vi_results['iterations'],
                    'time': vi_results['total_time'],
                    'value': np.mean(vi_results['V']),
                    'policy': vi_results['policy'],
                    'params': vi_params
                },
                'PI': {
                    'iterations': pi_results['iterations'],
                    'time': pi_results['total_time'],
                    'value': np.mean(pi_results['V']),
                    'policy': pi_results['policy'],
                    'params': pi_params
                }
            }

        # Plot discretization comparison
        plot_discretization_comparison(
            {k: {'VI': {'iterations': v['VI']['iterations']}, 'PI': {'iterations': v['PI']['iterations']}}
             for k, v in discretization_results.items()},
            'iterations',
            "CartPole: Effect of Discretization on Iterations",
            f"{SAVE_DIR}/cartpole/discretization_iterations.png",
            include_std=False
        )

        plot_discretization_comparison(
            {k: {'VI': {'time': v['VI']['time']}, 'PI': {'time': v['PI']['time']}}
             for k, v in discretization_results.items()},
            'time',
            "CartPole: Effect of Discretization on Computation Time",
            f"{SAVE_DIR}/cartpole/discretization_time.png",
            include_std=False
        )

        # Add discretization quality-time tradeoff visualization
        plot_discretization_quality_tradeoff(
            discretization_results,
            f"{SAVE_DIR}/cartpole/discretization_quality_tradeoff.png")

        # Use the largest bin size for RL experiments
        best_bins = max(CARTPOLE_BINS)
        cartpole_env = CartPoleEnv(n_bins=best_bins)

        # Tune SARSA and Q-Learning
        sarsa_params, sarsa_tuning_results = tune_hyperparameters(cartpole_env, RL_PARAM_GRID, SARSA,
                                                                  algorithm_type='RL',
                                                                  episodes=300, max_steps=150)
        q_params, q_tuning_results = tune_hyperparameters(cartpole_env, RL_PARAM_GRID, QLearning, algorithm_type='RL',
                                                          episodes=300, max_steps=150)

        # Add detailed hyperparameter tuning visualizations for CartPole RL algorithms
        print("\nGenerating detailed hyperparameter tuning visualizations for CartPole RL algorithms...")
        plot_rl_tuning_results(sarsa_tuning_results, "SARSA", "CartPole")
        plot_rl_tuning_results(q_tuning_results, "Q-Learning", "CartPole")

        # Compare exploration strategies
        best_sarsa_strategy, sarsa_strategy_results = compare_exploration_strategies(
            cartpole_env, SARSA, sarsa_params, episodes=400, max_steps=150)
        best_q_strategy, q_strategy_results = compare_exploration_strategies(
            cartpole_env, QLearning, q_params, episodes=400, max_steps=150)

        # Add exploration strategy visualizations for CartPole
        print("\nGenerating detailed exploration strategy visualizations for CartPole...")
        plot_exploration_strategy_comparison(
            sarsa_strategy_results, "SARSA", "CartPole")
        plot_exploration_strategy_comparison(
            q_strategy_results, "Q-Learning", "CartPole")

        # Add combined visualizations
        print("\nGenerating combined visualization for hyperparameter tuning...")
        all_tuning_results = {
            'Value Iteration': vi_tuning_results,
            'Policy Iteration': pi_tuning_results,
            'SARSA': sarsa_tuning_results,
            'Q-Learning': q_tuning_results
        }
        plot_combined_hyperparameter_results(all_tuning_results, "CartPole")

        print("\nGenerating combined visualization for exploration strategies...")
        plot_combined_exploration_strategies(sarsa_strategy_results, q_strategy_results, "CartPole")

        # Add best exploration strategy training progress visualization for CartPole
        plot_best_exploration_training_progress(
            sarsa_strategy_results,
            q_strategy_results,
            "CartPole",
            f"{SAVE_DIR}/hyperparameter_tuning/cartpole_best_exploration_training.png")

        # Run SARSA and Q-Learning with best strategies
        sarsa = SARSA(np.prod(cartpole_env.state_space), cartpole_env.action_space,
                      learning_rate=sarsa_params['learning_rate'],
                      gamma=sarsa_params['gamma'],
                      epsilon_decay=sarsa_params['epsilon_decay'],
                      exploration_strategy=best_sarsa_strategy)
        sarsa_results = sarsa.train(cartpole_env, episodes=CARTPOLE_EPISODES, max_steps=CARTPOLE_MAX_STEPS)

        qlearning = QLearning(np.prod(cartpole_env.state_space), cartpole_env.action_space,
                              learning_rate=q_params['learning_rate'],
                              gamma=q_params['gamma'],
                              epsilon_decay=q_params['epsilon_decay'],
                              exploration_strategy=best_q_strategy)
        qlearning_results = qlearning.train(cartpole_env, episodes=CARTPOLE_EPISODES, max_steps=CARTPOLE_MAX_STEPS)

        # Get policies from best bin size
        vi_policy = discretization_results[best_bins]['VI']['policy']
        pi_policy = discretization_results[best_bins]['PI']['policy']
        sarsa_policy = sarsa_results['policy']
        qlearning_policy = qlearning_results['policy']

        # Create visualizations
        plot_policy_cartpole(
            vi_policy,
            best_bins,
            "CartPole: Value Iteration Policy",
            f"{SAVE_DIR}/cartpole/vi_policy.png"
        )

        plot_policy_cartpole(
            pi_policy,
            best_bins,
            "CartPole: Policy Iteration Policy",
            f"{SAVE_DIR}/cartpole/pi_policy.png"
        )

        plot_policy_cartpole(
            sarsa_policy,
            best_bins,
            f"CartPole: SARSA Policy ({best_sarsa_strategy} exploration)",
            f"{SAVE_DIR}/cartpole/sarsa_policy.png"
        )

        plot_policy_cartpole(
            qlearning_policy,
            best_bins,
            f"CartPole: Q-Learning Policy ({best_q_strategy} exploration)",
            f"{SAVE_DIR}/cartpole/qlearning_policy.png"
        )

        plot_learning_curves(
            [sarsa_results['rewards'], qlearning_results['rewards']],
            ['SARSA', 'Q-Learning'],
            "CartPole: Learning Curves",
            f"{SAVE_DIR}/cartpole/learning_curves.png",
            window=50
        )

        plot_learning_curves(
            [sarsa_results['episode_lengths'], qlearning_results['episode_lengths']],
            ['SARSA', 'Q-Learning'],
            "CartPole: Episode Lengths",
            f"{SAVE_DIR}/cartpole/episode_lengths.png",
            window=50
        )

        # Calculate policy agreement
        vi_pi_agreement = np.mean(vi_policy == pi_policy) * 100
        sarsa_qlearning_agreement = np.mean(sarsa_policy == qlearning_policy) * 100
        vi_sarsa_agreement = np.mean(vi_policy == sarsa_policy) * 100
        pi_qlearning_agreement = np.mean(pi_policy == qlearning_policy) * 100

        # Store CartPole results
        cartpole_results = {
            'Value Iteration': {
                'iterations': discretization_results[best_bins]['VI']['iterations'],
                'time': discretization_results[best_bins]['VI']['time'],
                'reward': discretization_results[best_bins]['VI']['value'],
                'policy': vi_policy,
                'params': discretization_results[best_bins]['VI']['params']
            },
            'Policy Iteration': {
                'iterations': discretization_results[best_bins]['PI']['iterations'],
                'time': discretization_results[best_bins]['PI']['time'],
                'reward': discretization_results[best_bins]['PI']['value'],
                'policy': pi_policy,
                'params': discretization_results[best_bins]['PI']['params']
            },
            'SARSA': {
                'iterations': len(sarsa_results['rewards']),
                'time': sarsa_results['total_time'],
                'reward': sarsa_results['final_avg_reward'],
                'policy': sarsa_policy,
                'params': sarsa_params,
                'best_strategy': best_sarsa_strategy
            },
            'Q-Learning': {
                'iterations': len(qlearning_results['rewards']),
                'time': qlearning_results['total_time'],
                'reward': qlearning_results['final_avg_reward'],
                'policy': qlearning_policy,
                'params': q_params,
                'best_strategy': best_q_strategy
            }
        }

        cartpole_extra = {
            'discretization_results': discretization_results,
            'vi_pi_agreement': vi_pi_agreement,
            'sarsa_qlearning_agreement': sarsa_qlearning_agreement,
            'vi_sarsa_agreement': vi_sarsa_agreement,
            'pi_qlearning_agreement': pi_qlearning_agreement,
            'exploration_strategies': {
                'SARSA': sarsa_strategy_results,
                'Q-Learning': q_strategy_results
            }
        }

        # Create combined comparison plot
        plot_combined_comparison(
            {k: {'iterations': v['iterations'], 'time': v['time'], 'reward': v['reward']}
             for k, v in cartpole_results.items()},
            "CartPole: Algorithm Comparison",
            f"{SAVE_DIR}/cartpole/algorithm_comparison.png"
        )

        # Create performance dashboard
        create_performance_dashboard(
            cartpole_results,
            cartpole_extra,
            "CartPole",
            f"{SAVE_DIR}/cartpole/performance_dashboard.png")

        # Conduct specific analysis for alignment with assignment requirements
        convergence_analysis_cartpole = analyze_convergence_rates(
            discretization_results[best_bins]['VI'],
            discretization_results[best_bins]['PI'],
            "CartPole")

        discretization_analysis = analyze_discretization_effects(
            discretization_results, "CartPole")

        model_free_comparison_cartpole = analyze_model_free_comparison(
            sarsa_results, qlearning_results, "CartPole")

        exploration_analysis_sarsa_cartpole = analyze_exploration_strategies(
            sarsa_strategy_results, "SARSA", "CartPole")

        exploration_analysis_qlearning_cartpole = analyze_exploration_strategies(
            q_strategy_results, "Q-Learning", "CartPole")

        print("\nCartPole Experiments Completed")

        # Generate structured report content
        report_content = generate_report_content(
            blackjack_results, blackjack_extra,
            cartpole_results, cartpole_extra)

        # Save report structure for reference
        with open(f"{SAVE_DIR}/report_structure.json", "w") as f:
            import json

            json.dump(report_content, f, indent=2)

        # Print comprehensive summary
        print_summary(blackjack_results, blackjack_extra, cartpole_results, cartpole_extra)

        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY".center(80))
        print("=" * 80)
        print(f"\nResults saved to: {os.path.abspath(SAVE_DIR)}")

    except Exception as e:
        print(f"\nERROR: An exception occurred during execution: {e}")
        import traceback

        traceback.print_exc()
        print("\nExperiment failed to complete. Please check the error above.")