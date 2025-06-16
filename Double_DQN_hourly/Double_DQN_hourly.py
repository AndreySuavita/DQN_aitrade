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
import time
import pickle
import json
"""
    This model is designed to trade ETH/USDT on Binance using hourly data.
"""
# --- Loading and preprocessing data ---
def load_and_preprocess_data(filepath,load_scaler=False):
    """Loads and preprocesses historical data
    
    Args:
        filepath (str): Path to the CSV file with historical data
        load_scaler (bool): If True, loads and uses the previously saved scaler.
            If False, creates and saves a new scaler.
                           
    Returns:
        tuple: (normalized_data, original_data, scaler)
    """
    df = pd.read_csv(filepath, index_col='close_time')
    # Data cleaning
    df = df.replace([np.inf, -np.inf], np.nan).ffill()  
    # Add basic technical indicators
    df['MA_10'] = df['close_price'].rolling(window=10).mean()
    df['MA_50'] = df['close_price'].rolling(window=50).mean()
    df['hourly_return'] = df['close_price'].pct_change()
    
    # Select features
    selected_features = ['open_price', 'high_price', 'low_price', 'close_price', 
                        'close_volume', 'MA_10', 'MA_50', 'hourly_return']
    df = df[selected_features].dropna()
    
    # Normalize data
    if load_scaler:
        try:
            # Load existing scaler
            with open('eth_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            # Ensure columns match the original scaler
            if set(df.columns) != set(scaler.feature_names_in_):
                raise ValueError("Features in data don't match scaler features")
            df_normalized = pd.DataFrame(scaler.transform(df), columns=df.columns)
            print('Loaded existing scaler')
            
        except FileNotFoundError:
            raise FileNotFoundError("Scaler file not found. Run without load_scaler first")
        except Exception as e:
            raise ValueError(f"Error loading scaler: {str(e)}")
    else:
        # Create and save new scaler
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        # Save the scaler
        with open('eth_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print('Scaler saved')

        # Save scaler metadata
        scaler_metadata = {
        'feature_names': df.columns.tolist(),
        'data_min': scaler.data_min_.tolist(),
        'data_max': scaler.data_max_.tolist(),
        'scale': scaler.scale_.tolist(),
        'min': scaler.min_.tolist()
        }
        with open('scaler_metadata.json', 'w') as f:
            json.dump(scaler_metadata, f)
        print('Created and saved new scaler')

    return df_normalized.values, df, scaler

# --- Trading Environment ---
class EnhancedTradingEnvironment:
    def __init__(self, data, window_size=15):
        self.data = data   # Normalized data
        self.window_size = window_size # Visible history (15 steps)
        self.current_step = window_size # Current step, starts after having enough history
        self.max_steps = len(data) - 1 # Last possible step
        self.action_space = 3  # Possible actions: 0=sell, 1=hold, 2=buy
        self.state_size = window_size * data.shape[1] # Flattened state size (window * features) (8*15)
        self.position = 0  # 0=not invested, 1=invested (in ETH)
        self.commission = 0.001  # Commission of 0.1% per operation
        # print('--state_size--')
        # print(self.state_size)
        # exit()
    def reset(self):
        self.current_step = self.window_size
        self.position = 0
        return self._get_state()
    
    def _get_state(self):
        """
        Takes the data from the last window_size hours (e.g., 15 rows).
        flatten(): Converts the 2D matrix (15h x 8 features) into a 1D vector (for the neural network).
        """
        return self.data[self.current_step - self.window_size : self.current_step].flatten() # flattened data window

    def step(self, action):
        # Price Calculation
        current_price = self.data[self.current_step, 3]  # Current closing price
        next_price = self.data[self.current_step + 1, 3] if self.current_step < self.max_steps else current_price # Percentage change the increase or decrease of the price
        # print('--data--')
        # print(type(self.data))
        # print('--current_price--')
        # print(current_price)
        # print('--next_price--')
        # print(next_price)
        # exit()

        # Safe handling of price_change calculation, division by 0
        try:
            price_change = (next_price - current_price) / current_price if current_price != 0 else 0
        except Exception as e:
            logging.warning(f"Error calculating price_change: {e}")
            price_change = 0

        # Validate action based on current position
        valid_action = action
        if action == 2 and self.position == 1:  # Wants to buy but is already invested
            valid_action = 1  # Force hold
        elif action == 0 and self.position != 1: # Wants to sell without having a position
            valid_action = 1  # Force hold

        # Reward system
        if valid_action == 0:  # Sell
            reward = -price_change * 2.5 # Punish selling before rises
            self.position = 0
        elif valid_action == 2:  # Buy
            reward = price_change * 2.0  # Reward successful buys
            self.position = 1
        else:  # Hold
            reward = 0.2 if abs(price_change) < 0.01 else -0.1  # Reward holding in sideways markets

        # Apply commission
        if valid_action != 1:  # If not holding
            reward -= self.commission * 2  # Penalize commission (round trip)

        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_state = self._get_state()
        
        """
        next_state: New state (sliding window 1 hour).

        reward: Reward/Penalty.

        done: True if the episode ended.

        info: Useful metadata (current price, valid action).
        """
        return next_state, reward, done, {"price": current_price, "valid_action": valid_action}


# --- Neural Network ---
class EnhancedDQN(nn.Module):
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


class EnhancedDQNAgent:
    """
        Double DQN: Separates action selection and evaluation

        Experience Replay: Memory of 20,000 transitions

        Target Network: Separate network for stable calculations

        Soft Updates: Progressive update of the target network
    """
    def __init__(self, state_size, action_size, device='cpu'):	
        self.state_size = state_size # 120
        self.action_size = action_size # 3
        self.device = device # Device (CPU or GPU)
        self.memory = deque(maxlen=20000)  # Experience buffer
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration probability (100%)
        self.epsilon_min = 0.05  # Minimum exploration allowed (5%)
        self.epsilon_decay = 0.998  # Epsilon decay rate
        self.model = EnhancedDQN(state_size, action_size).to(self.device)  # Main network
        self.target_model = EnhancedDQN(state_size, action_size).to(self.device)  # Target network
        self.target_model.load_state_dict(self.model.state_dict())  # Identical initialization
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=1e-5)
        self.batch_size = 64  # Mini-batch size
        self.tau = 0.005  # For soft update of the target network
        self.update_every = 4  # Update frequency

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
        if np.random.rand() <= self.epsilon:  # Exploration (random action)
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()  # Evaluation mode
        with torch.no_grad():  # Disable gradients for evaluation
            q_values = self.model(state)  # Shape: [1, 3] (Q-values for each action)
        self.model.train()  # Switch back to training mode
        return torch.argmax(q_values).item()  # Exploitation (best action), return action with highest Q-value

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
        minibatch = random.sample(self.memory, self.batch_size)  # selects 64 random experiences from past experiences
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(self.device)

        # Double DQN # Uses model to select action, but target_model to evaluate its Q-value.
        next_actions = self.model(next_states).max(1)[1]  # Selection with main network
        next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))  # Evaluation with target network
        target = rewards + (1 - dones) * self.gamma * next_q.squeeze()  # Target calculation, adjusted Bellman formula

        # Update weights # Backpropagation
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        loss = nn.MSELoss()(current_q.squeeze(), target.detach())
        self.optimizer.zero_grad()
        loss.backward()  # Calculates gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Avoids exploding gradients
        self.optimizer.step()  # Updates weights

        # Soft update of the target network
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # Decrease epsilon (exploration) to increase exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, save_path, e, best_score, train_rewards):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),  # Important for Double DQN
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': e,  # Current episode
            'best_score': best_score,
            'train_rewards': train_rewards  # Training rewards history
        }, save_path)
        print(f"💾 Model saved to {save_path} (Episode {e}, ε={self.epsilon:.4f})")

    def load_model(self, saved_path):
        try:
            # 1. Load the checkpoint with safety handling
            checkpoint = torch.load(saved_path, 
                                map_location=self.device,
                                weights_only=False)  

            # 2. Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])

            # 3. Move models to the correct device
            self.model.to(self.device)
            self.target_model.to(self.device)

            # 4. Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Move optimizer tensors to the device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

            # 5. Restore training parameters
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            best_score = checkpoint.get('best_score', -np.inf)
            train_rewards = checkpoint.get('train_rewards', [])

            print(f"✅ Model loaded successfully to {self.device}")
            print(f"| ε: {self.epsilon:.4f} | Best Score: {best_score:.2f}% |")

            return self, best_score, train_rewards
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
        

# --- Improved Evaluation Function ---
def evaluate(agent, env, scaler, initial_balance=10000):
    '''
    The evaluate function has two key objectives:
    Assess the agent's performance with a conservative trading strategy
    Simulate real trades with capital management (only invests 10% of the portfolio in each trade)

    Input parameters:
    - agent: The DQN agent we have trained (contains the neural network model)
    - env: The trading environment (can be training or testing)
    - scaler: The MinMaxScaler used to normalize the data
    - initial_balance: Initial capital for the simulation (default $10,000)
    '''
    state = env.reset()               # Reset the environment (first state)
    portfolio = initial_balance       # E.g: $10,000 initial USD
    positions = 0                     # Amount of assets (ETH) held
    portfolio_history = [portfolio]   # Record portfolio value at each step
    price_history = []                # Store denormalized prices
    actions_history = []              # Record actions taken
    done = False                      # Simulation end flag

    # Prepare an empty array for scaling investment
    temp_array = np.zeros((1, len(scaler.feature_names_in_)))  # Use the same dimensionality as the scaler

    while not done:
        action = agent.act(state) # Choose action (0, 1, 2)
        next_state, reward, done, info = env.step(action) # Execute action in the environment

        # Manual denormalization of price
        # price_range = df['close_price'].max() - df['close_price'].min()
        # current_price = info['price'] * price_range + df['close_price'].min()

        # Denormalization using the scaler - INDEX 3 FOR close_price
        temp_array.fill(0)  # Clear the temporary array
        temp_array[0, 3] = info['price']  # Place the normalized value in position 3 (close_price)
        denormalized = scaler.inverse_transform(temp_array)
        current_price = denormalized[0, 3]  # Get the denormalized close_price

        # Conservative trading logic
        if action == 2 and portfolio > 0: #buy
            buy_amount = portfolio * 0.1 # invest 10% of the portfolio
            positions += (buy_amount * (1 - env.commission)) / current_price  # Buy ETH
            portfolio -= buy_amount # Reduce cash
        elif action == 0 and positions > 0: # sell
            sell_amount = positions * 0.1 # Sell 10% of the ETH held
            portfolio += (sell_amount * current_price) * (1 - env.commission) # Convert to USD
            positions -= sell_amount  # Reduce the position in ETH

        current_value = portfolio + positions * current_price  # Total value (USD + ETH)
        portfolio_history.append(current_value)  # Record current value
        price_history.append(current_price)     # Record price history
        actions_history.append(action)          # Record action
        state = next_state  # Move to the next state

    final_return = (portfolio_history[-1] / initial_balance - 1) * 100 # Percentage return
    

    return final_return, portfolio_history, price_history, actions_history

# --- Metric Calculation ---
def metrics(portfolio_history, final_return, price_history, actions_history, initial_balance):
    # Metric calculations
    final_value = portfolio_history[-1]
    returns = np.diff(portfolio_history) / portfolio_history[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365)
    max_drawdown = (np.maximum.accumulate(portfolio_history) - portfolio_history).max()
    buy_hold_return = (price_history[-1] / price_history[0] - 1) * 100
    actions_dist = pd.Series(actions_history).value_counts(normalize=True)
    print("\n--- Final Results ---")
    print(f"Initial Value: ${initial_balance:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Test Performance (%): {final_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: ${max_drawdown:,.2f} ({max_drawdown/initial_balance:.2%})")
    print(f"Actions: Buy={actions_dist.get(2, 0):.1%}, "
          f"Sell={actions_dist.get(0, 0):.1%}, "
          f"Hold={actions_dist.get(1, 0):.1%}")
    

# --- Results Display ---
def plot_results(portfolio_history, price_history, actions_history, train_rewards, initial_balance):
    
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
    plt.axhline(y=initial_balance, color='red', linestyle='--', label='Initial Investment')
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
    plt.plot(train_rewards, label='Rewards', color='purple')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Progression')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

"""------------------------------ MAIN EXECUTION ------------------------------"""
if __name__ == "__main__":
    # Initial setup
    start_time = time.time()
    logging.basicConfig(filename='trading_bot.log', level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # --- Load data ---
    data_array, df, scaler = load_and_preprocess_data('C:\\Andrey\\Kakua_Projets\\Trading\\Bot_RL_v1\\Datasets\\historical_01-01-2019_to_01-01-2025_ETHUSDT.csv')

    # --- Split data (80% train, 20% test) ---
    train_size = int(0.8 * len(data_array))
    train_data = data_array[:train_size]
    test_data = data_array[train_size:]

    print(f"\nData split:")
    print(f"Total: {len(data_array)}")
    print(f"Training: {len(train_data)}")
    print(f"Evaluation: {len(test_data)}")

    # --- Create environments ---
    window_size = 15
    initial_balance = 10000 # USD
    train_env = EnhancedTradingEnvironment(train_data, window_size)
    test_env = EnhancedTradingEnvironment(test_data, window_size)
    state_size = train_env.state_size
    action_size = train_env.action_space
    # print('--state_size--')
    # print(state_size)
    # print('--action_size--')
    # print(action_size)
    # exit()

    # --- Training configuration ---
    agent = EnhancedDQNAgent(state_size, action_size, device)
    episodes = 200
    save_path = 'best_trading_model.pth'
    best_score = -np.inf
    no_improve = 0
    patience = 15
    #patience = max(15, int(episodes * 0.01))

    # Dimension verification
    print(f"Number of features: {train_data.shape[1]}")
    print(f"Window size: {window_size}")
    print(f"State size Calculated: {window_size * train_data.shape[1]}")

    # --- Training ---
    print("\nStarting training...")
    train_rewards = []

    # Each episode is a complete pass through the training data
    for e in range(episodes):
        state = train_env.reset() # Reset to the beginning of the data
        total_reward = 0
        done = False

        while not done: # Until reaching the end of the training data
            action = agent.act(state)  # Decide to buy/sell/hold (ε-greedy)
            next_state, reward, done, _ = train_env.step(action) # Apply action
            agent.remember(state, action, reward, next_state, done) # Store experiences for later learning
            state = next_state # Move to the next state
            total_reward += reward

            if len(agent.memory) > agent.batch_size: # If there are enough stored experiences (batch_size=64)
                agent.replay() # backpropagation, Train the neural network with mini-batches
        # End of the episode
        train_rewards.append(total_reward)

        # Evaluation and saving
        if e % 5 == 0:
            val_return, _, _ , actions= evaluate(agent, test_env, scaler)
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
                # Save the entire state of the agent
                agent.save_model(save_path,e,best_score,train_rewards)
            else:
                no_improve += 1
                if no_improve >= patience: # If no improvement in "patience" evaluations
                    print(f"Early stopping in episode {e}")
                    break
        

    """ --- Final Evaluation using the evaluate function --- """
    # load the saved model
    agent, best_score, train_rewards = agent.load_model(save_path, device)

    print("\nEvaluating on test data...")
    final_return, portfolio_history, price_history, actions_history = evaluate(
        agent, test_env, scaler, initial_balance
    )

    # 5. Final results and metrics
    metrics(portfolio_history, final_return, price_history, actions_history, initial_balance)
    
    print(f"\nTotal Time: {(time.time() - start_time)/3600:.2f} hours")

    # --- Visualization ---
    plot_results(
        portfolio_history=portfolio_history,
        price_history=price_history,
        actions_history=actions_history,
        train_rewards=train_rewards,
        initial_balance=initial_balance
    )
    
    
    

    