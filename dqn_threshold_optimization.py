import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import random
from sklearn.metrics import precision_score, recall_score, f1_score
import json

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("\n✅ Libraries loaded!")


# Import TSSGC model
import sys
sys.path.append('..')
from model import TSSGC

# Load data
data = torch.load('data/processed/fraud_graph_with_splits.pt', weights_only=False).to(device)

# Load trained model
model = TSSGC(
    input_dim=data.num_node_features,
    hidden_dim=64,
    output_dim=2,
    num_layers=3,
    dropout=0.5,
    temporal_dim=32,
    num_node_types=2
).to(device)

model.load_state_dict(torch.load('results/model_tssgc.pt', weights_only=False))
model.eval()

print("✅ Trained TSSGC model loaded!")
print(f"Data: {data}")



# Get fraud probabilities from trained model
with torch.no_grad():
    out = model(data.x, data.edge_index, data.time, data.amount)
    probs = torch.exp(out)[:, 1]  # Probability of fraud class

print("✅ Model predictions generated!")
print(f"Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
print(f"Mean probability: {probs.mean():.4f}")

# Analyze probability distribution
fraud_probs = probs[data.y == 1].cpu().numpy()
normal_probs = probs[data.y == 0].cpu().numpy()

print(f"\nFraud transactions - Mean prob: {fraud_probs.mean():.4f}")
print(f"Normal transactions - Mean prob: {normal_probs.mean():.4f}")



class DQNAgent(nn.Module):
    """
    Deep Q-Network for threshold optimization
    State: Current fraud probabilities statistics
    Action: Fraud detection threshold (0.0 to 1.0)
    Reward: F1 score - False Alarm Rate penalty
    """
    def __init__(self, state_dim=10, hidden_dim=128, action_dim=100):
        super(DQNAgent, self).__init__()
        
        # Q-Network
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Action space: 100 threshold values from 0.01 to 1.0
        self.action_space = torch.linspace(0.01, 1.0, action_dim)
        
    def forward(self, state):
        """Get Q-values for all actions"""
        return self.network(state)
    
    def select_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            # Explore: random action
            action_idx = random.randint(0, len(self.action_space) - 1)
        else:
            # Exploit: best action
            with torch.no_grad():
                q_values = self.forward(state)
                action_idx = q_values.argmax().item()
        
        threshold = self.action_space[action_idx].item()
        return action_idx, threshold


class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float)
        )
    
    def __len__(self):
        return len(self.buffer)


print("✅ DQN Agent defined!")


class ThresholdEnvironment:
    """
    Environment for threshold optimization
    """
    def __init__(self, probs, labels, mask):
        self.probs = probs[mask].cpu().numpy()
        self.labels = labels[mask].cpu().numpy()
        self.current_step = 0
        self.max_steps = 100
        
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        return self.get_state()
    
    def get_state(self):
        """
        State: Statistics of fraud probabilities
        - Mean, std, min, max, quartiles
        """
        state = torch.tensor([
            self.probs.mean(),
            self.probs.std(),
            self.probs.min(),
            self.probs.max(),
            np.percentile(self.probs, 25),
            np.percentile(self.probs, 50),
            np.percentile(self.probs, 75),
            np.sum(self.labels),  # Number of frauds
            len(self.labels),  # Total transactions
            self.current_step / self.max_steps  # Progress
        ], dtype=torch.float)
        return state
    
    def step(self, threshold):
        """
        Take action (set threshold) and get reward
        """
        self.current_step += 1
        
        # Make predictions based on threshold
        predictions = (self.probs >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(self.labels, predictions, zero_division=0)
        recall = recall_score(self.labels, predictions, zero_division=0)
        f1 = f1_score(self.labels, predictions, zero_division=0)
        
        # False alarm rate
        tn = np.sum((predictions == 0) & (self.labels == 0))
        fp = np.sum((predictions == 1) & (self.labels == 0))
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Reward function: Balance F1 score and FAR
        # Goal: High F1, Low FAR
        reward = f1 - 0.5 * far  # Penalize false alarms
        
        # Bonus for good recall (catching frauds is critical)
        if recall > 0.85:
            reward += 0.2
        
        # Done if max steps reached
        done = self.current_step >= self.max_steps
        
        next_state = self.get_state()
        
        info = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'far': far,
            'threshold': threshold
        }
        
        return next_state, reward, done, info


print("✅ Environment defined!")


# Create environment (use validation set)
env = ThresholdEnvironment(probs, data.y, data.val_mask)

# Create DQN agent
state_dim = 10
action_dim = 100  # 100 possible thresholds
dqn_agent = DQNAgent(state_dim, hidden_dim=128, action_dim=action_dim).to(device)
target_agent = DQNAgent(state_dim, hidden_dim=128, action_dim=action_dim).to(device)
target_agent.load_state_dict(dqn_agent.state_dict())

# Replay buffer
replay_buffer = ReplayBuffer(capacity=1000)

# Optimizer
optimizer = torch.optim.Adam(dqn_agent.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training parameters
NUM_EPISODES = 200
BATCH_SIZE = 32
GAMMA = 0.95  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

epsilon = EPSILON_START
best_reward = -float('inf')
best_threshold = 0.5

# Training history
history = {
    'episode_rewards': [],
    'episode_f1': [],
    'episode_far': [],
    'episode_thresholds': [],
    'losses': []
}

print("\n" + "="*70)
print("TRAINING DQN AGENT FOR THRESHOLD OPTIMIZATION")
print("="*70 + "\n")

for episode in range(NUM_EPISODES):
    state = env.reset()
    episode_reward = 0
    episode_loss = 0
    step_count = 0
    
    best_f1_in_episode = 0
    best_threshold_in_episode = 0.5
    
    for step in range(env.max_steps):
        # Select action
        state_tensor = state.to(device)
        action_idx, threshold = dqn_agent.select_action(state_tensor, epsilon)
        
        # Take step
        next_state, reward, done, info = env.step(threshold)
        
        # Store transition
        replay_buffer.push(state, action_idx, reward, next_state, done)
        
        # Update state
        state = next_state
        episode_reward += reward
        step_count += 1
        
        # Track best threshold
        if info['f1'] > best_f1_in_episode:
            best_f1_in_episode = info['f1']
            best_threshold_in_episode = threshold
        
        # Train DQN
        if len(replay_buffer) >= BATCH_SIZE:
            # Sample batch
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            states = states.to(device)
            next_states = next_states.to(device)
            
            # Current Q-values
            current_q = dqn_agent(states).gather(1, actions.unsqueeze(1).to(device))
            
            # Target Q-values
            with torch.no_grad():
                next_q = target_agent(next_states).max(1)[0]
                target_q = rewards.to(device) + GAMMA * next_q * (1 - dones.to(device))
            
            # Compute loss
            loss = criterion(current_q.squeeze(), target_q)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            episode_loss += loss.item()
        
        if done:
            break
    
    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    # Update target network
    if episode % TARGET_UPDATE == 0:
        target_agent.load_state_dict(dqn_agent.state_dict())
    
    # Store history
    history['episode_rewards'].append(episode_reward)
    history['episode_f1'].append(best_f1_in_episode)
    history['episode_far'].append(info['far'])
    history['episode_thresholds'].append(best_threshold_in_episode)
    if episode_loss > 0:
        history['losses'].append(episode_loss / step_count)
    
    # Track best
    if episode_reward > best_reward:
        best_reward = episode_reward
        best_threshold = best_threshold_in_episode
    
    # Print progress
    if episode % 20 == 0 or episode == NUM_EPISODES - 1:
        print(f"Episode {episode:3d} | "
              f"Reward: {episode_reward:>7.3f} | "
              f"F1: {best_f1_in_episode:.4f} | "
              f"FAR: {info['far']:.4f} | "
              f"Best Threshold: {best_threshold_in_episode:.4f} | "
              f"Epsilon: {epsilon:.3f}")

print(f"\n✅ DQN Training Complete!")
print(f"Best Reward: {best_reward:.3f}")
print(f"Best Threshold: {best_threshold:.4f}")

# Save DQN agent
torch.save(dqn_agent.state_dict(), '../results/dqn_agent.pt')
print("✅ DQN agent saved!")


print("\n" + "="*70)
print("TEST SET EVALUATION WITH OPTIMIZED THRESHOLD")
print("="*70)

# Get test predictions
test_probs = probs[data.test_mask].cpu().numpy()
test_labels = data.y[data.test_mask].cpu().numpy()

# Apply optimized threshold
test_predictions = (test_probs >= best_threshold).astype(int)

# Calculate metrics
from sklearn.metrics import confusion_matrix, classification_report

test_precision = precision_score(test_labels, test_predictions, zero_division=0)
test_recall = recall_score(test_labels, test_predictions, zero_division=0)
test_f1 = f1_score(test_labels, test_predictions, zero_division=0)
test_cm = confusion_matrix(test_labels, test_predictions)

tn, fp, fn, tp = test_cm.ravel()
test_far = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\n📊 TSSGC + DQN Results (Optimized Threshold: {best_threshold:.4f}):")
print(f"{'='*70}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print(f"FAR:       {test_far:.4f} ({test_far*100:.2f}%)")

print(f"\n📊 Confusion Matrix:")
print(test_cm)

print(f"\nTrue Negatives:  {tn:>6d}")
print(f"False Positives: {fp:>6d}")
print(f"False Negatives: {fn:>6d}")
print(f"True Positives:  {tp:>6d}")

# Save results
results_dqn = {
    'model': 'TSSGC + DQN',
    'optimized_threshold': float(best_threshold),
    'test_metrics': {
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1': float(test_f1),
        'far': float(test_far)
    },
    'confusion_matrix': {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }
}

with open('results/results_tssgc_dqn.json', 'w') as f:
    json.dump(results_dqn, f, indent=4)

print("\n✅ Results saved to: results/results_tssgc_dqn.json")


fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Rewards
axes[0, 0].plot(history['episode_rewards'], linewidth=2, color='#2E86AB')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].set_title('DQN Training Rewards')
axes[0, 0].grid(True, alpha=0.3)

# F1 Score
axes[0, 1].plot(history['episode_f1'], linewidth=2, color='#06A77D')
axes[0, 1].axhline(y=test_f1, color='red', linestyle='--', label=f'Final Test F1: {test_f1:.4f}')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('F1 Score')
axes[0, 1].set_title('F1 Score Evolution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# False Alarm Rate
axes[1, 0].plot(history['episode_far'], linewidth=2, color='#F24236')
axes[1, 0].axhline(y=test_far, color='green', linestyle='--', label=f'Final FAR: {test_far:.4f}')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('False Alarm Rate')
axes[1, 0].set_title('False Alarm Rate Evolution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Threshold
axes[1, 1].plot(history['episode_thresholds'], linewidth=2, color='#F18F01')
axes[1, 1].axhline(y=best_threshold, color='red', linestyle='--', label=f'Best: {best_threshold:.4f}')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Threshold')
axes[1, 1].set_title('Threshold Evolution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/dqn_training.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ DQN training plots saved!")
