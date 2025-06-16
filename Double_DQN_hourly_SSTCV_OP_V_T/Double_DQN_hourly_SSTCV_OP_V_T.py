import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from collections import deque
import random
import matplotlib.pyplot as plt
import logging
import time
from datetime import datetime, timedelta, timezone
import pickle
import json
import os
import sys
sys.path.append("../")
from binance_actions import binance_actions
"""
    Model created using Split Series Time Cross Validation (SSTCV) with Double DQN.
    This model is designed to trade ETH/USDT on Binance using hourly data.
    SSTCV is used to optimize the model's performance by training on different time splits independently using a separate agent for each fold.
    the final training is done on the entire dataset using the best parameters found during the optimization.
"""

def wait_until_next_hour_with_offset(offset_seconds=15):
    """Wait until the next time cycle with an offset"""
    now = datetime.now(timezone.utc)  # Modern and recommended way
    # Calculate the next hour start
    next_hour = (now.replace(minute=0, second=0, microsecond=0) 
                 + timedelta(hours=1))
    # Add the offset
    target_time = next_hour + timedelta(seconds=offset_seconds)
    # Calculate how long to wait
    wait_seconds = (target_time - now).total_seconds()
    if wait_seconds > 0:
        time.sleep(wait_seconds)

# --- Load and preprocess data (same as before) ---
def load_and_preprocess_data(window_size, filepath='', load_scaler=False, binance_on=False):
    """Loads and preprocesses historical data
    
    Args:
        filepath (str): Path to the CSV file with historical data
        load_scaler (bool): If True, loads and uses the previously saved scaler.
            If False, creates and saves a new scaler.
        binance_on (bool): If True, uses the Binance API to fetch data.
    
    Returns:
        tuple: (normalized_data, original_data, scaler)
    """
    window_1 = 10
    window_2 = 50
    max_window = max(window_1, window_2)
    if binance_on:
        binance = binance_actions()
        data=binance.get_klines(symbol="ETHUSDT", interval='1h', limit=max_window - 1 + window_size)
        df = data.set_index('close_time')
        # print(df.shape)
        # exit()
    else:
        df = pd.read_csv(filepath, index_col='close_time')
    # Data cleaning
    df = df.replace([np.inf, -np.inf], np.nan).ffill()  
    # Add basic technical indicators
    df[f'MA_{window_1}'] = df['close_price'].rolling(window=window_1).mean()
    df[f'MA_{window_2}'] = df['close_price'].rolling(window=window_2).mean()
    df['Percentage_return'] = df['close_price'].pct_change()
    
    # Select features
    selected_features = ['open_price', 'high_price', 'low_price', 'close_price', 
                        'close_volume', f'MA_{window_1}', f'MA_{window_2}', 'Percentage_return']
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
        print('Saved new scaler')

    return df_normalized.values, df, scaler
# --- Trading Environment ---
class EnhancedTradingEnvironment:
    def __init__(self, data, window_size=15, binance_on=False):
        self.data = data # Normalized data
        self.window_size = window_size # Visible history 
        self.binance_on = binance_on # If True, use Binance API for real-time data
        self.current_step = window_size # Current step, starts after having enough history
        self.max_steps = len(data) - 1 # Last possible step
        self.action_space = 3 # Possible actions: 0=sell, 1=hold, 2=buy
        self.state_size = window_size * self.data.shape[1] # Flattened state size (window * features) 
        self.position = 0 # 0=not invested, 1=invested (in ETH)
        self.commission = 0.001 # Commission of 0.1% per operation
    
    def reset(self):
        if self.binance_on:
            self.data, _, _ = load_and_preprocess_data(window_size=self.window_size, load_scaler=True, binance_on=self.binance_on)
            data_flattened = self.data.flatten()  # Flatten the data
            # print(self.data.shape)
            # print(data_flattened.shape)
            # exit()
            return data_flattened # Flattened data window
        else:
            self.current_step = self.window_size
            self.position = 0
            return self._get_state()
    
    def _get_state(self):
        """
        Takes the data from the last window_size hours (e.g., 15 rows).
        flatten(): Converts the 2D matrix (15h x 8 features) into a 1D vector (for the neural network).
        """
        return self.data[self.current_step - self.window_size : self.current_step].flatten()
    
    def step(self, action):
        # If Binance is connected, return the current price
        if self.binance_on:
            current_price = self.data[self.window_size-1, 3]  # Current close price
            return {"price": current_price}
        else:
            current_price = self.data[self.current_step, 3]
            next_price = self.data[self.current_step + 1, 3] if self.current_step < self.max_steps else current_price
            
            # Safe handling of price_change calculation, division by 0
            try:
                price_change = (next_price - current_price) / current_price if current_price != 0 else 0
            except Exception as e:
                logging.warning(f"Error calculating price_change: {e}")
                price_change = 0
            
            # Validate action based on current position
            valid_action = action
            if action == 2 and self.position == 1: # Wants to buy but is already invested
                valid_action = 1 # Force hold
            elif action == 0 and self.position != 1: # Wants to sell without having a position
                valid_action = 1 # Force hold
            
            # Reward system
            if valid_action == 0: # Sell
                reward = -price_change * 2.5 # Punish selling before rises
                self.position = 0
            elif valid_action == 2: # Buy
                reward = price_change * 2.0 # Reward successful buys
                self.position = 1
            else: # Hold
                reward = 0.2 if abs(price_change) < 0.01 else -0.1 # Reward holding in sideways markets
            
            # Apply commission, action different from 1 (hold)
            if valid_action != 1:
                reward -= self.commission * 2
            
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
    
# --- Network Architecture --- 
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

# --- DQN Agent ---
class EnhancedDQNAgent:
    """
        Double DQN: Separates action selection and evaluation

        Experience Replay: Memory of 20,000 transitions

        Target Network: Separate network for stable calculations

        Soft Updates: Progressive update of the target network
    """
    def __init__(self, state_size, action_size, device='cpu', gamma=0.99, lr=0.0005, batch_size=64):
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
        self.model = EnhancedDQN(state_size, action_size).to(device)  # Main network
        self.target_model = EnhancedDQN(state_size, action_size).to(device) # Target network
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
            print(f"γ={self.hyperparams['gamma']}, lr={self.hyperparams['lr']}, batch={self.hyperparams['batch_size']}")
            print(f"| ε: {self.epsilon:.4f} | Average Fold results: {mean_fold_results:.2f}% | Best Score: {best_score:.2f}% |")

            return self, best_score, episode, mean_fold_results, std_fold_results, fold_results, train_rewards
        
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise


# --- Enhanced Evaluation Function ---
def evaluate(agent, env, scaler, initial_balance=10000, binance_on=False):
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
    state = env.reset()
    portfolio = initial_balance       # Example: $10,000 initial USD
    positions = 0                     # Amount of assets (ETH) held
    portfolio_history = [portfolio]   # Record portfolio value at each step
    price_history = []                # Store denormalized prices
    actions_history = []              # Record actions taken  
    done = False                      # Simulation end flag
    counter = 0                       # Counter for Binance connection    

    # Prepare an empty array for scaling investment
    temp_array = np.zeros((1, len(scaler.feature_names_in_))) # Use the same dimensionality as the scaler

    while not done:     
        # If Binance is connected, use its step method
        if binance_on:
            # Wait until 10 seconds after the next UTC hour
            wait_until_next_hour_with_offset() 
            #time.sleep(5)
            state = env.reset()
            action = agent.act(state) # Choose action (0, 1, 2)  
            info = env.step(action) # Execute action in the environment
        else:
            action = agent.act(state) # Choose action (0, 1, 2)
            next_state, reward, done, info = env.step(action) # Execute action in the environment
        # print(scaler.feature_names_in_) check column order

        # Denormalization using the scaler - INDEX 3 FOR close_price
        temp_array.fill(0) # Clear the temporary array
        temp_array[0, 3] = info['price'] # Place the normalized value in position 3 (close_price)
        denormalized = scaler.inverse_transform(temp_array)
        current_price = denormalized[0, 3] # Get the denormalized close_price

        # Conservative trading logic
        if action == 2 and portfolio > 0: #buy
            buy_amount = portfolio * 0.1 # invest 10% of the portfolio
            positions += (buy_amount * (1 - env.commission)) / current_price # Buy ETH
            portfolio -= buy_amount # Reduce cash
        elif action == 0 and positions > 0: # sell
            sell_amount = positions * 0.1 # Sell 10% of the ETH held
            portfolio += (sell_amount * current_price) * (1 - env.commission) # Convert to USD
            positions -= sell_amount # Reduce ETH position
        
        current_value = portfolio + positions * current_price # Total value (USD + ETH)
        portfolio_history.append(current_value)
        price_history.append(current_price)
        actions_history.append(action)
        

        # If Binance is connected
        if binance_on:
            print('[-] Current value Portafolio:',current_value)
            print('[-] Current price',current_price)
            if action == 2:
                print('[-] Action: Buy',action)
            elif action == 0:
                print('[-] Action: Sell',action)
            else:
                print('[-] Action: Hold',action)
            counter += 1
            print(f'[-] Step_number: {counter}')
            print('[+]------------------------------------------------')
            # wait number_steps steps to finish the evaluation
            if counter >= binance_on:
                done = True
        else:
            state = next_state

    final_return = (portfolio_history[-1] / initial_balance - 1) * 100 # Percentage return
    
    return final_return, portfolio_history, price_history, actions_history

# --- Time Series CV Training (modified) ---
def time_series_cv_train(agent, best_params, full_train_data, window_size, initial_balance, episodes=200, n_splits=5, patience=5):
    """Enhanced training with Time Series CV and model selection"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    global_best_score = -np.inf
    all_train_rewards = []
    
    # Create directory for models if it doesn't exist
    os.makedirs('cv_models', exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(full_train_data)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        
        # Clone the agent to ensure independent training for each fold
        fold_agent = EnhancedDQNAgent(
            state_size=agent.state_size,
            action_size=agent.action_size,
            device=agent.device,
            **best_params  
        )
        
        train_fold = full_train_data[train_idx]
        val_fold = full_train_data[val_idx]
        
        train_env = EnhancedTradingEnvironment(train_fold, window_size)
        val_env = EnhancedTradingEnvironment(val_fold, window_size)
        
        fold_best_score = -np.inf
        no_improve = 0
        fold_train_rewards = []
        
        for e in range(episodes):
            state = train_env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = fold_agent.act(state)
                next_state, reward, done, _ = train_env.step(action)
                fold_agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if len(fold_agent.memory) > fold_agent.batch_size:
                    fold_agent.replay()
            
            fold_train_rewards.append(total_reward)
            
            # Validation every 5 episodes
            if e % 5 == 0:
                val_return, _, _, _ = evaluate(fold_agent, val_env, scaler, initial_balance)
                
                # Update best model for this fold
                if val_return > fold_best_score:
                    fold_best_score = val_return
                    no_improve = 0
                    torch.save(fold_agent.model.state_dict(), f'cv_models/best_fold_{fold}.pth')
                
                # Update global best model
                if val_return > global_best_score:
                    global_best_score = val_return
                    # Update the main agent with the best weights
                    agent.model.load_state_dict(fold_agent.model.state_dict())
                    agent.target_model.load_state_dict(fold_agent.target_model.state_dict())
                
                print(f"Ep {e+1}/{episodes} | Train R: {total_reward:.2f} | Val R: {val_return:.2f}% | ε: {fold_agent.epsilon:.3f}")
                
                # Early stopping
                no_improve += 1
                if no_improve >= patience:
                    print(f"⏹ Early stopping in episode {e+1}")
                    break
        
        # Final evaluation of this fold
        fold_agent.model.load_state_dict(torch.load(f'cv_models/best_fold_{fold}.pth'))
        final_return, _, _, _ = evaluate(fold_agent, val_env, scaler, initial_balance)
        fold_results.append(final_return)
        all_train_rewards.extend(fold_train_rewards)
        
        print(f"Fold {fold+1} completed. Return: {final_return:.2f}%")
    
    return np.mean(fold_results), np.std(fold_results), fold_results, agent, all_train_rewards

# --- Hyperparameter Optimization (modified) ---
def optimize_hyperparams(full_train_data, window_size, initial_balance, param_grid, n_splits=3, episodes=50, patience=3):
    """Optimize hyperparameters using Time Series CV with separate agents for each fold"""
    best_params = {}
    best_score = -np.inf
    results = []
    
    # Generate parameter combinations
    from itertools import product
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    for params in param_combinations:
        print(f"\n🔍 Testing parameters: {params}")
        fold_scores = []
        
        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(full_train_data)):
            print(f"\n=== Fold {fold + 1}/{n_splits} ===")
            # Create new agent for each fold to ensure clean evaluation
            agent = EnhancedDQNAgent(
                state_size=window_size * full_train_data.shape[1],
                action_size=3,
                device=device,
                **params
            )
            
            train_fold = full_train_data[train_idx]
            val_fold = full_train_data[val_idx]
            
            train_env = EnhancedTradingEnvironment(train_fold, window_size)
            val_env = EnhancedTradingEnvironment(val_fold, window_size)
            
            # Train with early stopping
            best_fold_score = -np.inf
            no_improve = 0
            
            for e in range(episodes):
                state = train_env.reset()
                done = False
                
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, _ = train_env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    
                    if len(agent.memory) > agent.batch_size:
                        agent.replay()
                
                # Validation every 5 episodes
                if e % 5 == 0:
                    val_return, _, _, _ = evaluate(agent, val_env, scaler, initial_balance)
                    
                    if val_return > best_fold_score:
                        best_fold_score = val_return
                        no_improve = 0
                    else:
                        no_improve += 1
                        
                    if no_improve >= patience:
                        break
            
            fold_scores.append(best_fold_score)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        results.append((params, mean_score, std_score))
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            print(f"🔥 New best score: {best_score:.2f}% ± {std_score:.2f}")
    
    return best_params, results

# --- Metric Calculation ---
def metrics(portfolio_history, test_return, price_history, actions_history, initial_balance, mean_fold_results, std_fold_results):
        final_value = portfolio_history[-1]
        returns = np.diff(portfolio_history) / portfolio_history[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        max_drawdown = (np.maximum.accumulate(portfolio_history) - portfolio_history).max()
        buy_hold_return = (price_history[-1] / price_history[0] - 1) * 100
        actions_dist = pd.Series(actions_history).value_counts(normalize=True)

        print("\n--- Final Results ---")
        print(f"CV Performance: {mean_fold_results:.2f}% ± {std_fold_results:.2f}")
        print(f"Initial Value: ${initial_balance:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Test Performance (%): {test_return:.2f}%")
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: ${max_drawdown:,.2f} ({max_drawdown/initial_balance:.2%})")
        print(f"Actions: Buy={actions_dist.get(2, 0):.1%}, "
                f"Sell={actions_dist.get(0, 0):.1%}, "
                f"Hold={actions_dist.get(1, 0):.1%}")

def plot_results(fold_results, train_rewards, initial_balance, actions_history, test_results, save_img=False, eval_img_path='Current_evaluation.png'):
    plt.figure(figsize=(20, 12))  # Tamaño más grande para 6 gráficas

    # --- Gráfica 1: Price vs Portfolio Value ---
    plt.subplot(2, 3, 1)  # Fila 1, Col 1
    plt.plot(test_results['price_history'], label='ETH Price', color='blue', alpha=0.6)
    plt.ylabel('Price (USD)')
    plt.legend(loc='upper left')
    plt.grid(True)
    ax2 = plt.gca().twinx()
    ax2.plot(test_results['portfolio_history'], label='Portfolio Value', color='green')
    ax2.axhline(y=initial_balance, color='red', linestyle='--', label='Initial Investment')
    ax2.set_ylabel('Value (USD)')
    ax2.legend(loc='upper right')
    plt.title('Price vs Portfolio Value in Test')

    # --- Gráfica 2: Performance by fold ---
    plt.subplot(2, 3, 2)  # Fila 2, Col 1
    plt.bar(range(1, len(fold_results)+1), fold_results, color='skyblue')
    plt.axhline(y=np.mean(fold_results), color='r', linestyle='--', label='Average')
    plt.title('Performance by Fold in SSTCV')
    plt.xlabel('Fold')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True)

    # --- Gráfica 3: Training rewards ---
    plt.subplot(2, 3, 3)  # Fila 1, Col 2
    plt.plot(train_rewards, label='Reward', color='purple')
    plt.title('Rewards During Training')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)

    # --- Gráfica 4: Action Timeline ---
    plt.subplot(2, 3, 4)  # Fila 1, Col 3
    plt.plot(actions_history, 'o', markersize=2, alpha=0.6)
    plt.yticks([0, 1, 2], ['Sell', 'Hold', 'Buy'])
    plt.xlabel('Time Step (hours)')
    plt.ylabel('Action')
    plt.title('Action Timeline')
    plt.grid(True)

    # --- Gráfica 5: Actions Distribution ---
    plt.subplot(2, 3, 5)  # Fila 2, Col 2
    actions_dist = test_results['actions_dist']
    plt.bar(['Sell', 'Hold', 'Buy'], 
            [actions_dist.get(0, 0), actions_dist.get(1, 0), actions_dist.get(2, 0)])
    plt.title('Actions Distribution in Test')
    plt.ylabel('Proportion')

    # --- Gráfica 6: Evaluation Returns ---
    plt.subplot(2, 3, 6)  # Fila 2, Col 3
    try:
        img = plt.imread(eval_img_path)
        plt.imshow(img)
        plt.axis('off')
        #plt.title('Training Evaluation Returns')
    except FileNotFoundError:
        plt.text(0.5, 0.5, 'Evaluation plot not found', ha='center')
        plt.axis('off')

    plt.tight_layout(pad=3.0)  # Space between subplots
    
    if save_img:
        # Save image with high resolution
        plt.savefig(save_img, bbox_inches='tight', dpi=300)
        print(f"✓ Graph saved in {save_img}")
        
    plt.show()
    plt.close()

def plot_evaluation_py(episode_counts, eval_rewards, filename='Current_evaluation.png', show_result=False):
    # Tamaño proporcional a 1/6 del tamaño total de la figura principal (20x12)
    subplot_width = 20 / 3  # Ancho por columna (3 columnas)
    subplot_height = 12 / 2  # Alto por fila (2 filas)
    
    plt.figure(figsize=(subplot_width, subplot_height))  # Tamaño individual
    
    plt.plot(episode_counts, eval_rewards, label='Val Return', color='purple')  
    plt.title('Training Evaluation Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return (USD)')
    plt.grid(True)
    plt.legend()
    
    # Guardar con alta resolución y ajustando bordes
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()
    
    if show_result:
        img = plt.imread(filename)
        plt.figure(figsize=(subplot_width, subplot_height))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

# --- Main Execution  ---
if __name__ == "__main__":
    # --- Log and device configuration ---
    start_time = time.time()
    logging.basicConfig(filename='trading_bot.log', level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # initial configuration for live plotting
    episode_counts = []
    eval_rewards = []

    # --- Parameters ---
    window_size = 15
    initial_balance = 10000 # USD
    episodes = 600
    
    # 1. Load and prepare data
    data_array, df, scaler = load_and_preprocess_data(window_size=window_size, filepath='C:\\Andrey\\Kakua_Projets\\Trading\\Bot_RL_v1\\Datasets\\historical_01-01-2019_to_01-01-2025_ETHUSDT.csv')
    train_size = int(0.8 * len(data_array))
    full_train_data = data_array[:train_size]
    test_data = data_array[train_size:]

    print(f"\nData Split:")
    print(f"Training: {len(full_train_data)} samples")
    print(f"Test: {len(test_data)} samples")

    # 2. Hyperparameter Optimization (optional)
    # param_grid = {
    #     'gamma': [0.95, 0.99],
    #     'lr': [0.0001, 0.0005],
    #     'batch_size': [32, 64]
    # }

    # print("\nStarting hyperparameter optimization...")
    # best_params, opt_results = optimize_hyperparams(
    #     full_train_data, window_size, initial_balance, param_grid,
    #     n_splits=3, episodes=50, patience=3
    # )

    # print(f"\nBest parameters found: {best_params}")

    best_params = {
        'gamma': 0.95,
        'lr': 0.0001,
        'batch_size': 32
    }
    
    # 3. Final training with best parameters
    print("\nSplit time series cross-validation with best parameters...")
    agent = EnhancedDQNAgent(
        state_size=window_size * full_train_data.shape[1],
        action_size=3,
        device=device,
        **best_params
    )
    
    ## Commented out the cross-validation training for now
    # mean_fold_results, std_fold_results, fold_results, trained_agent, train_rewards_cv = time_series_cv_train(
    #     agent, best_params, full_train_data, window_size, initial_balance,
    #     episodes=episodes, n_splits=5, patience=5
    # )

    mean_fold_results = 89.84351506364882
    std_fold_results = 150.12635790298214
    fold_results = [384.4632589509047, 55.16628321984469, -34.54628107093532, 25.688069424624803, 18.4462447938053]
    
    # 4. Final training on entire dataset
    
    # --- Create environments ---
    train_env = EnhancedTradingEnvironment(full_train_data, window_size)
    test_env = EnhancedTradingEnvironment(test_data, window_size)

    # using best model from CV (optional)
    # agent = trained_agent

    save_path = 'best_trading_model.pth'
    best_score = -np.inf
    patience = 50
    no_improve = 0

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
            val_return, _, _ , actions= evaluate(agent, test_env, scaler, initial_balance)
            elapsed = (time.time() - start_time) / 3600

            # Print return vs episode:
            episode_counts.append(e)
            eval_rewards.append(val_return)      
            plot_evaluation_py(episode_counts, eval_rewards)

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
                agent.save_model(
                    save_path=save_path,
                    e=episodes,
                    best_score=best_score,
                    mean_fold_results=mean_fold_results,
                    std_fold_results=std_fold_results,
                    fold_results=fold_results,
                    train_rewards=train_rewards
                )
            else:
                no_improve += 1
                if no_improve >= patience: # If no improvement in "patience" evaluations
                    print(f"Early stopping in episode {e}")
                    break
    
    # save the final model
    agent.save_model(
        save_path='last_trading_model.pth',
        e=episodes,
        mean_fold_results=mean_fold_results,
        std_fold_results=std_fold_results,
        fold_results=fold_results,
        train_rewards=train_rewards
    )

    # Print last return vs episode:
    plot_evaluation_py(episode_counts, eval_rewards)

    # 5. Evaluation on test set

    # load the saved model
    agent, best_score ,episode, mean_fold_results, std_fold_results, fold_results, train_rewards = agent.load_model(save_path)
    
    print("\nEvaluating on test set...")

    test_return, portfolio_history, price_history, actions_history = evaluate(
        agent, test_env, scaler, initial_balance
    )

    # 6. Final results and metrics
    metrics(portfolio_history, test_return, price_history, actions_history, initial_balance, mean_fold_results, std_fold_results)

    print(f"\nTotal execution time: {(time.time() - start_time)/3600:.2f} hours")
    
    # 7. Results visualization
    test_results = {
        'final_return': test_return,
        'portfolio_history': portfolio_history,
        'price_history': price_history,
        'actions_dist': pd.Series(actions_history).value_counts(normalize=True).to_dict()
    }

    plot_results(
        fold_results=fold_results,  
        train_rewards=train_rewards,
        initial_balance=initial_balance,
        actions_history=actions_history,
        test_results=test_results,
        save_img='Final_test_result.png'
    )
