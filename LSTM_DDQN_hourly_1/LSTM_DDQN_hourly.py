import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import random
import matplotlib.pyplot as plt
import logging
import pandas_ta as ta
import time


# Initial configuration
start_time = time.time()
logging.basicConfig(filename='trading_bot.log', level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# --- Load and preprocess data ---
def load_and_preprocess_data(filepath):
    """Loads and preprocesses historical data with improved normalization"""
    df = pd.read_csv(filepath, index_col='close_time')
    df.index = pd.to_datetime(df.index)
    df = df.replace([np.inf, -np.inf], np.nan).ffill()

    # --- Improved technical indicators ---
    df['OBV'] = ta.obv(df['close_price'], df['close_volume'])
    df['VWAP'] = ta.vwap(df['high_price'], df['low_price'], df['close_price'], df['close_volume'])
    df['MA_24h'] = df['close_price'].rolling(window=24).mean()
    df['MA_168h'] = df['close_price'].rolling(window=168).mean()
    df['hourly_return'] = df['close_price'].pct_change()
    df['RSI_14h'] = ta.rsi(df['close_price'], length=14)
    df['EMA_12'] = df['close_price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close_price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['ATR_24h'] = ta.atr(df['high_price'], df['low_price'], df['close_price'], length=24)
    df['Momentum_24h'] = df['close_price'].pct_change(24)

    # Special normalization for bounded indicators
    df['RSI_14h'] = df['RSI_14h'] / 100  # Normalize RSI between 0-1
    df['MACD'] = np.tanh(df['MACD'].values * 0.1)  # Safer version

    # Improved normalization
    df['OBV'] = np.tanh(df['OBV'].values * 1e-7)
    df['VWAP'] = (df['VWAP'] - df['close_price'].mean()) / df['close_price'].std()

    # Standard normalization for other features
    scaler = MinMaxScaler()
    features_to_scale = ['open_price', 'high_price', 'low_price', 'close_price', 
                         'close_volume', 'MA_24h', 'MA_168h', 'hourly_return',
                         'ATR_24h', 'Momentum_24h']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    return df.values, df, scaler

# --- Enhanced Trading Environment ---
class EnhancedTradingEnvironment:
    def __init__(self, data, window_size=24):  
        self.data = data   
        self.window_size = window_size 
        self.current_step = window_size 
        self.max_steps = len(data) - 1 
        self.action_space = 3  
        self.state_size = window_size * data.shape[1] 
        self.position = 0  
        self.commission = 0.001  
        self.max_position_size = 0.1  

    def reset(self): 
        self.current_step = self.window_size  
        self.position = 0                     
        return self._get_state()    
    
    def _get_state(self):
        """
        Takes the last window_size hours of data (e.g., 24 rows).
        flatten(): Converts the 2D matrix (24h x 8 features) into a 1D vector (for the neural network).
        """
        return self.data[self.current_step - self.window_size : self.current_step].flatten()
    
    def step(self, action):
        current_price = self.data[self.current_step, 3]  # Current closing price (column 3)
        next_price = self.data[self.current_step + 1, 3] if self.current_step < self.max_steps else current_price # Next hour's price
        price_change = (next_price - current_price) / current_price if current_price != 0 else 0

        # Validate action -- Prevent impossible actions (e.g., selling without holding).
        valid_action = action
        if action == 2 and self.position == 1:   # Wants to buy but already invested
            valid_action = 1                     # Force to hold
        elif action == 0 and self.position != 1: # Wants to sell without holding
            valid_action = 1                     # Force to hold


        # Reward system with momentum
        momentum = self.data[self.current_step, 10]  # Momentum index
        obv = self.data[self.current_step, 5]       # Normalized OBV
        rsi = self.data[self.current_step, 7]  # Normalized RSI (0-1)

        if valid_action == 2:  # Buy
            reward = price_change * (6.0 + 2.0 * momentum)  # Positive reinforcement with momentum
            if rsi > 0.7:
                reward *= 0.2  # Stronger penalty
            elif obv < -0.5:   # If money flow is negative
                reward *= 0.3
            self.position = 1

        elif valid_action == 0:  # Sell
            reward = -price_change * (2.0 - 1.0 * momentum)  # Smaller reward
            if rsi < 0.3:
                reward *= 0.3
            elif obv > 0.5:     # If money flow is positive
                reward *= 0.5
            self.position = 0

        else:  # Hold
            reward = 0.1 * (1 + obv)  # Reward based on money flow


        # Penalty for over-operating
        if valid_action != 1 and abs(price_change) < 0.005:  # Small movements
            reward -= 0.1

        # Limit rewards
        reward = np.clip(reward, -2.0, 2.0)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps 
        next_state = self._get_state()
        
        """
        next_state: New state (1 hour sliding window).

        reward: Reward/penalty.

        done: True if the episode finished.

        info: Useful metadata (current price, valid action).
        """
        return next_state, reward, done, {"price": current_price, "valid_action": valid_action}

# --- Neural Network ---
class EnhancedDQN(nn.Module):
    def __init__(self, state_size, action_size, window_size=24):
        self.window_size = window_size
        self.state_size = state_size
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_size//self.window_size, 
                           hidden_size=64, 
                           num_layers=2,
                           batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(64, 512),
            nn.SiLU(),  
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  
            nn.SiLU(),
            nn.Linear(256, action_size)
        )
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(-1, self.window_size, self.state_size//self.window_size)  
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  
        return self.net(x)

class EnhancedDQNAgent:
    """
        Double DQN: Separate action selection and evaluation

        Experience Replay: Memory of 50,000 transitions

        Target Network: Separate network for stable calculations

        Soft Updates: Progressive update of the target network
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size # 192
        self.action_size = action_size # 3
        self.memory = deque(maxlen=50000)  # Experience replay buffer
        self.gamma = 0.98  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration probability (100%)
        self.epsilon_min = 0.05  # Minimum exploration probability (5%)
        self.epsilon_decay = 0.98 # Very slow decay rate for epsilon
        self.model = EnhancedDQN(state_size, action_size).to(device) # Main network
        self.target_model = EnhancedDQN(state_size, action_size).to(device) # Target network
        self.target_model.load_state_dict(self.model.state_dict())  # Identical initialization
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=1e-5, amsgrad=True)
        self.batch_size = 512  # Mini-batch size
        self.tau = 0.005  # For soft update of the target network
        self.update_every = 5  # Update frequency

    def remember(self, state, action, reward, next_state, done):
        """
            Function: Stores experiences (state, action, reward, next_state, done)
            Capacity: 50,000 samples (deletes the oldest ones when this limit is exceeded)
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state): # Decision making
        """
            ε-greedy: Balance between exploration (random actions) and exploitation (using the model)
            Processing:
                - Converts the state to tensor
                - Adds batch dimension (unsqueeze)
                - Moves to GPU if available
        """
        if np.random.rand() <= self.epsilon:  # Exploration (random action)
            return random.randrange(self.action_size)

        # Exploitation using the model
        state = torch.FloatTensor(state).to(device)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension if necessary

        self.model.eval()  # Evaluation mode
        with torch.no_grad():  # Disable gradients for evaluation
            q_values = self.model(state)  # Shape: [1, 3] (Q-values for each action)
        self.model.train()  # Switch back to training mode
        return torch.argmax(q_values).item()  # Exploitation (best action), return action with highest Q-value

    def replay(self):  # Training (replay) - When memory ≥ batch_size
        """
            Samples 64 random experiences from the buffer.

            Calculates the Q-targets (using Double DQN).

            Performs backpropagation and updates the network weights.

            Soft updates the target network (target_model).
        """
        if len(self.memory) < self.batch_size:
            return

        # Experience Sampling
        minibatch = random.sample(self.memory, self.batch_size)  # Selects 64 random experiences from past experiences
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(device)
        #print("Dimensiones de states:", states.shape)  # Should be [batch_size, state_size]
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(device)

        # Double DQN # Uses model to select action, but target_model to evaluate its Q-value.
        next_actions = self.model(next_states).max(1)[1]  # Selection with main network
        next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))  # Evaluation with target network
        target = rewards + (1 - dones) * self.gamma * next_q.squeeze()  # Target calculation, adjusted Bellman formula

        # Weight Update # Backpropagation
        current_q = self.model(states).gather(1, actions.unsqueeze(1))  # Current predictions
        loss = nn.MSELoss()(current_q.squeeze(), target.detach())  # Loss calculation (MSE)

        # Optimization
        self.optimizer.zero_grad()
        loss.backward()  # Calculate gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Prevent exploding gradients
        self.optimizer.step()  # Update weights

        # Soft update of the target network
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # Decrease epsilon (exploration) to increase exploitation
        # if agent.epsilon > agent.epsilon_min:
        #     agent.epsilon *= agent.epsilon_decay

    def save_model(self, save_path, e, best_score, train_rewards):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),  # Important for Double DQN
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': e,  # Current episode
            'best_score': best_score,
            'train_rewards': train_rewards  # Reward history
        }, save_path)
        print(f"💾 Model saved to {save_path} (Episode {e}, ε={self.epsilon:.4f})")

    def load_model(self, saved_path,device):
        try:
            # 1. Load the checkpoint with safety handling
            checkpoint = torch.load(saved_path, 
                                map_location=device,
                                weights_only=False)

            # 2. Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])

            # 3. Move models to the correct device
            self.model.to(device)
            self.target_model.to(device)

            # 4. Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Move optimizer tensors to the device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)

            # 5. Restore training parameters
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            best_score = checkpoint.get('best_score', -np.inf)
            train_rewards = checkpoint.get('train_rewards', [])

            print(f"✅ Model loaded successfully to {device}")
            print(f"| ε: {self.epsilon:.4f} | Best Score: {best_score:.2f}% |")

            return self, best_score, train_rewards
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise

# --- Evaluation Function  ---
def evaluate(agent, env, df, initial_balance=10000, return_full_history=False):
    """
    Main Purpose
    Evaluate how your trading strategy would perform in the real world, using:

    Historical data (hourly prices).

    Policy learned by the agent (buy/sell/hold).

    Conservative capital management (1% investment per trade).
    
    """
    state = env.reset()               # Reset the environment (first state)
    portfolio = initial_balance       # E.g., $10,000 initial USD
    positions = 0                     # Amount of assets (ETH) held
    portfolio_history = [portfolio]   # Record portfolio value at each step
    price_history = []                # Store denormalized prices
    actions_history = []              # Record actions taken
    done = False

    # Price denormalizer
    price_min = df['close_price'].min()
    price_range = df['close_price'].max() - price_min
    
    while not done:
        action = agent.act(state)
        next_state, _, done, info = env.step(action)

        # Denormalize price
        current_price = info['price'] * price_range + price_min

        # Execute trades
        if action == 2 and portfolio > 0:  # Buy
            buy_amount = portfolio * 0.1  # 10% of the portfolio
            positions += (buy_amount * (1 - env.commission)) / current_price
            portfolio -= buy_amount
        elif action == 0 and positions > 0:  # Sell
            sell_amount = positions * 0.1  # Sell 10% of the position
            portfolio += (sell_amount * current_price) * (1 - env.commission)
            positions -= sell_amount

        # Record values
        current_value = portfolio + positions * current_price
        portfolio_history.append(current_value)
        price_history.append(current_price)
        actions_history.append(action)
        state = next_state

    # Calculate percentage return
    final_return = (portfolio_history[-1] / initial_balance - 1) * 100
    
    if return_full_history:
        return final_return, portfolio_history, price_history, actions_history
    return final_return, portfolio_history

""" ----------------------------------------------------- Main Implementation ------------------------------------------------------ """
if __name__ == "__main__":
    # Load data
    data_array, df, scaler = load_and_preprocess_data('C:\\Andrey\\Kakua_Projets\\Trading\\Bot_RL_v1\\Datasets\\historical_01-01-2019_to_01-01-2025_ETHUSDT.csv')

    # --- Data division (80% train, 20% test) ---
    train_size = int(0.8 * len(data_array))
    train_data = data_array[:train_size]
    test_data = data_array[train_size:]

    print(f"\nData division:")
    print(f"Total: {len(data_array)}")
    print(f"Training: {len(train_data)}")
    print(f"Evaluation: {len(test_data)}")

    # --- Environment creation ---
    window_size = 24  # Changed to 24 hours
    train_env = EnhancedTradingEnvironment(train_data, window_size)
    test_env = EnhancedTradingEnvironment(test_data, window_size)
    state_size = window_size * train_data.shape[1]  # features * time window
    print(f"State size calculated: {state_size}")
    action_size = train_env.action_space

    # --- Training configuration ---
    agent = EnhancedDQNAgent(state_size, action_size)
    episodes = 150 
    save_path = 'best_trading_model.pth'
    best_score = -np.inf
    no_improve = 0
    patience = max(15, int(episodes * 0.01))

    # Dimension verification
    print(f"Number of features: {train_data.shape[1]}")
    print(f"Window size: {window_size}")
    print(f"State size calculated: {window_size * train_data.shape[1]}")

    # --- Training Phases
    print("\nStarting training...")
    train_rewards = []

    # Each episode is a complete pass through the training data
    for e in range(episodes):
        state = train_env.reset() # Reset to the start of the data
        total_reward = 0
        done = False

        # Adjust parameters by phase
        if e < int(episodes*0.4):
            agent.epsilon = max(0.6, agent.epsilon)
            train_env.commission = 0.0003
        elif e < int(episodes*0.8):
            agent.epsilon = max(0.25, agent.epsilon)
            train_env.commission = 0.0005
        else:
            agent.epsilon = max(0.1, agent.epsilon)
            train_env.commission = 0.0008

        while not done: # Until reaching the end of the training data
            action = agent.act(state) # Decide to buy/sell/hold (ε-greedy)
            next_state, reward, done, _ = train_env.step(action) # Apply action
            agent.remember(state, action, reward, next_state, done) # Store experiences for later learning
            state = next_state # Move to the next state
            total_reward += reward

            if len(agent.memory) > agent.batch_size: # If there are enough stored experiences (batch_size=64)
                agent.replay() # backpropagation, Train the neural network with mini-batches
            #print(f'epsilon: {agent.epsilon}')
        
        train_rewards.append(total_reward)

        # Evaluation and saving
        if e % 10 == 0:
            val_return, _, _, actions = evaluate(agent, test_env, df, return_full_history=True)
            elapsed = (time.time() - start_time) / 3600

            # Calculate action distribution
            actions_dist = pd.Series(actions).value_counts(normalize=True)

            print(f"Episode: {e+1}/{episodes}, Reward: {total_reward:.2f}, "
                  f"Val Return: {val_return:.2f}%, ε: {agent.epsilon:.3f}, "
                  f"Time: {elapsed:.2f}h")
            print(f"Actions: Buy={actions_dist.get(2, 0):.1%}, "
                  f"Sell={actions_dist.get(0, 0):.1%}, "
                  f"Hold={actions_dist.get(1, 0):.1%}")

            # Early stopping
            if val_return > best_score:
                best_score = val_return
                no_improve = 0
                # Save ALL agent status 
                agent.save_model(save_path,e,best_score,train_rewards) 

            else:
                no_improve += 1
                if no_improve >= patience: # If no improvement in "patience" evaluations
                    print(f"Early stopping at episode {e}")
                    break

        # Decrease epsilon (exploration) to increase exploitation
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        if e % 2 == 0:
            print('e: ',agent.epsilon)
    """------------------------- Final Evaluation and Visualization -------------------------"""
    # Load the saved model
    agent, best_score, train_rewards = agent.load_model(save_path, device)

    # --- Final Evaluation ---
    print("\nEvaluating on test data...")
    final_return, portfolio_history, price_history, actions_history = evaluate(
        agent, test_env, df, return_full_history=True)

    # Metric calculations
    final_value = portfolio_history[-1]
    returns = np.diff(portfolio_history) / portfolio_history[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365)
    max_drawdown = (np.maximum.accumulate(portfolio_history) - portfolio_history).max()
    buy_hold_return = (price_history[-1] / price_history[0] - 1) * 100
    actions_dist = pd.Series(actions_history).value_counts(normalize=True)
    print("\n--- Final Results ---")
    print(f"Initial Value: $10,000.00")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Strategy Return: {(final_value/10000-1)*100:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: ${max_drawdown:,.2f} ({max_drawdown/10000:.2%})")
    print(f"Actions: Buy={actions_dist.get(2, 0):.1%}, "
          f"Sell={actions_dist.get(0, 0):.1%}, "
          f"Hold={actions_dist.get(1, 0):.1%}")
    print(f"Total Time: {(time.time() - start_time)/3600:.2f} hours")

    # --- Visualization ---
    plt.figure(figsize=(16, 10))

    # Plot 1: Price
    plt.subplot(2, 2, 1)
    plt.plot(price_history, label='ETH Price', color='blue', alpha=0.6)
    plt.xlabel('Time Step (hours)')
    plt.ylabel('Price (USD)')
    plt.title('Price during Evaluation')
    plt.grid(True)

    # Plot 2: Portfolio
    plt.subplot(2, 2, 2)
    plt.plot(portfolio_history, label='Portfolio Value', color='green')
    plt.axhline(y=10000, color='red', linestyle='--', label='Initial Investment')
    plt.xlabel('Time Step (hours)')
    plt.ylabel('Value (USD)')
    plt.title('Portfolio Performance')
    plt.legend()
    plt.grid(True)

    # Plot 3: Actions
    plt.subplot(2, 2, 3)
    plt.plot(actions_history, 'o', markersize=2, alpha=0.6)
    plt.yticks([0, 1, 2], ['Sell', 'Hold', 'Buy'])
    plt.xlabel('Time Step (hours)')
    plt.ylabel('Action')
    plt.title('Action Distribution')
    plt.grid(True)

    # Plot 4: Rewards
    plt.subplot(2, 2, 4)
    plt.plot(train_rewards, label='Reward', color='purple')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Progression')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Mark the end
    end_time = time.time()
    # Calculate elapsed time in hours
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time/3600:.4f} Hours")