import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import torch
import numpy as np

# --- Network Architecture Hourly --- 
class EnhancedDQN_hourly(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128) # input layer, 120 inputs, 128 outputs
        self.bn1 = nn.BatchNorm1d(128) # batch normalization
        self.fc2 = nn.Linear(128, 64) # hidden layer, 128 inputs, 64 outputs
        self.bn2 = nn.BatchNorm1d(64) # batch normalization
        self.fc3 = nn.Linear(64, action_size) # output layer, 64 inputs, 3 outputs (actions)
        self.dropout = nn.Dropout(0.25) # Drop 25% of neurons to avoid overfitting
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0) # Add batch dimension if needed
        x = torch.relu(self.bn1(self.fc1(x))) # First transformation
        x = self.dropout(x) # Drop ~32 neurons (25% of 128) to avoid overfitting
        x = torch.relu(self.bn2(self.fc2(x))) # Hidden layer
        return self.fc3(x) # [1,64] → [1,3] (Q-values for sell/hold/buy)

# --- Network Architecture 5m ---
class EnhancedDQN_5m(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256) # input layer, 120 inputs, 128 outputs
        self.bn1 = nn.BatchNorm1d(256) # batch normalization
        self.fc2 = nn.Linear(256, 128) # hidden layer, 256 inputs, 128 outputs
        self.bn2 = nn.BatchNorm1d(128) # batch normalization
        self.fc3 = nn.Linear(128, action_size) # output layer, 128 inputs, 3 outputs (actions)
        self.dropout = nn.Dropout(0.5) # Drop 50% of neurons to avoid overfitting

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0) # Add batch dimension if needed
        x = torch.relu(self.bn1(self.fc1(x))) # First transformation
        x = self.dropout(x) # Drop ~32 neurons (25% of 128) to avoid overfitting
        x = torch.relu(self.bn2(self.fc2(x))) # Hidden layer
        return self.fc3(x) # [1,128] → [1,3] (Q-values for sell/hold/buy)

# --- DQN Agent hourly---
class EnhancedDQNAgent:
    """
        Double DQN: Separates action selection and evaluation

        Experience Replay: Memory of 20,000 transitions

        Target Network: Separate network for stable calculations

        Soft Updates: Progressive update of the target network
    """
    def __init__(self, state_size, action_size, time_cycle, device='cpu', gamma=0.99, lr=0.0005, batch_size=64):
        self.hyperparams = {  # Save all hyperparameters
            'gamma': gamma, # Discount factor for future rewards
            'lr': lr,
            'batch_size': batch_size, # Mini-batch size
            'epsilon': 1.0,  # Initial exploration probability (100%)
            'epsilon_min': 0.05, # Minimum exploration allowed (5%)
            'epsilon_decay': 0.998, # Epsilon decay rate
            'tau': 0.005, # For soft update of the target network
            'update_every': 4 # Update frequency
        }
        self.state_size = state_size # window_size * features
        self.action_size = action_size # 3 actions: sell, hold, buy
        self.device = device # Device (CPU or GPU)
        self.memory = deque(maxlen=20000) # Experience buffer
        self.gamma = self.hyperparams['gamma']
        self.epsilon = self.hyperparams['epsilon']
        self.epsilon_min = self.hyperparams['epsilon_min']
        self.epsilon_decay = self.hyperparams['epsilon_decay']
        if time_cycle == 'hourly':
            self.model = EnhancedDQN_hourly(state_size, action_size).to(device)  # Main network
            self.target_model = EnhancedDQN_hourly(state_size, action_size).to(device) # Target network
        elif time_cycle == '5m':
            self.model = EnhancedDQN_5m(state_size, action_size).to(device)  # Main network
            self.target_model = EnhancedDQN_5m(state_size, action_size).to(device) # Target network
        self.target_model.load_state_dict(self.model.state_dict()) # Identical initialization
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'], weight_decay=1e-5)
        self.batch_size = self.hyperparams['batch_size'] 
        self.tau = self.hyperparams['tau']
        self.update_every = self.hyperparams['update_every']

    def remember(self, state, action, reward, next_state, done):
        """
            Function: Stores experiences (state, action, reward, next_state, done)
            Capacity: 20,000 samples (removes oldest ones when exceeding this limit)
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state): # Decision making
        """
            ε-greedy: Balance between exploration (random actions) and exploitation (using the model)
            Processing:
                - Converts state to tensor
                - Adds batch dimension (unsqueeze)
                - Moves to GPU if available
        """
        if np.random.rand() <= self.epsilon: # Exploration (random action)
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval() # Evaluation mode
        with torch.no_grad(): # Disable gradients for evaluation
            q_values = self.model(state) # Shape: [1, 3] (Q-values for each action)
        self.model.train() # Switch back to training mode
        return torch.argmax(q_values).item() # Exploitation (best action), return action with highest Q-value
    
    def replay(self): # Training (replay) - When memory ≥ batch_size
        """
            Samples 64 random experiences from the buffer.

            Calculates Q-targets (using Double DQN).

            Performs backpropagation and updates the network weights.

            Soft updates the target network (target_model).
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sampling Experiences
        minibatch = random.sample(self.memory, self.batch_size) # selects 64 random experiences from past experiences
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(self.device)
        
        # Double DQN # Uses model to select action, but target_model to evaluate its Q-value.
        next_actions = self.model(next_states).max(1)[1] # Selection with main network
        next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)) # Evaluation with target network
        target = rewards + (1 - dones) * self.gamma * next_q.squeeze() # Target calculation, adjusted Bellman formula
        
        # Update weights # Backpropagation
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        loss = nn.MSELoss()(current_q.squeeze(), target.detach())
        self.optimizer.zero_grad()
        loss.backward() # Calculates gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Avoids exploding gradients
        self.optimizer.step() # Updates weights

        # Soft update of the target network
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        # Decrease epsilon (exploration) to increase exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, save_path, e, best_score, mean_fold_results, std_fold_results, fold_results, train_rewards):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparams': self.hyperparams,  # Save all hyperparameters
            'episode': e,
            'best_score': best_score,
            'mean_fold_results': mean_fold_results,
            'std_fold_results': std_fold_results,
            'fold_results': fold_results,
            'train_rewards': train_rewards,
            'epsilon': self.epsilon  # Save separately in case it changes during training
        }, save_path)
        print(f"💾 Model saved to {save_path} (Episode {e}, ε={self.epsilon:.4f})")

    def load_model(self, saved_path):
        try:
            # 1. Load the checkpoint with safety handling
            checkpoint = torch.load(saved_path, map_location=self.device, weights_only=False)

            # Load hyperparameters (with default values if they don't exist)
            self.hyperparams = checkpoint.get('hyperparams', {
                'gamma': 0.99,
                'lr': 0.0005,
                'batch_size': 64,
                'epsilon_min': 0.05,
                'epsilon_decay': 0.998,
                'tau': 0.005,
                'update_every': 4
            })

            # Apply hyperparameters
            self.gamma = self.hyperparams['gamma']
            self.batch_size = self.hyperparams['batch_size']
            self.epsilon = checkpoint.get('epsilon', self.hyperparams.get('epsilon', 1.0))

            # Recreate optimizer with the saved lr
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'])

            # 2. Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])

            # 3. Move models to the correct device
            self.model.to(self.device)
            self.target_model.to(self.device)

            # 4. Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Move optimizer tensors to the correct device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

            # 5. Restore training parameters
            best_score = checkpoint.get('best_score', -np.inf)
            episode = checkpoint.get('episode', 0)
            mean_fold_results = checkpoint.get('mean_fold_results', -np.inf)
            std_fold_results = checkpoint.get('std_fold_results', 0.0)
            fold_results = checkpoint.get('fold_results', [])
            train_rewards = checkpoint.get('train_rewards', [])


            print(f"✅ Model loaded successfully with hyperparameters:")
            print(f"gamma={self.hyperparams['gamma']}, lr={self.hyperparams['lr']}, batch={self.hyperparams['batch_size']}")
            print(f"| ε: {self.epsilon:.4f} | Average Fold results: {mean_fold_results:.2f}% | Best Score: {best_score:.2f}% |")

            return self, best_score, episode, mean_fold_results, std_fold_results, fold_results, train_rewards
        
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise

