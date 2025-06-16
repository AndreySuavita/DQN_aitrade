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

# Initial configuration
start_time = time.time()
logging.basicConfig(filename='trading_bot_optimized.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# --- Optimized Hyperparameters ---
class Hyperparameters:
    WINDOW_SIZE = 15                  
    HIDDEN_SIZE = 128                 
    NUM_LAYERS = 2                    
    BATCH_SIZE = 256                  
    GAMMA = 0.98                      
    EPS_START = 1.0
    EPS_END = 0.1                    
    EPS_DECAY = 0.99                 
    TAU = 0.005                        
    LR = 0.005                        
    TRAIN_EPISODES = 150              
    PATIENCE = 15                     

""" --- Loading and preprocessing data --- """
def load_and_preprocess_data(filepath):
    """Loads and preprocesses historical data with more technical indicators"""
    df = pd.read_csv(filepath, index_col='close_time', parse_dates=True)

    # Data cleaning
    df = df.replace([np.inf, -np.inf], np.nan).ffill()

    # Add more technical indicators
    df['MA_10'] = df['close_price'].rolling(window=10).mean()
    df['MA_50'] = df['close_price'].rolling(window=50).mean()
    df['MA_200'] = df['close_price'].rolling(window=200).mean()
    df['hourly_return'] = df['close_price'].pct_change()
    df['RSI'] = compute_rsi(df['close_price'], 14)
    df['MACD'] = compute_macd(df['close_price'])
    df['Bollinger_Upper'], df['Bollinger_Lower'] = compute_bollinger_bands(df['close_price'])

    # Select features
    selected_features = ['open_price', 'high_price', 'low_price', 'close_price',
                        'close_volume', 'MA_10', 'MA_50', 'MA_200', 'hourly_return',
                        'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
    df = df[selected_features].dropna()

    # Normalize data
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return df_normalized.values, df, scaler

# Functions for technical indicators
def compute_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd

def compute_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper = rolling_mean + (rolling_std * num_std)
    lower = rolling_mean - (rolling_std * num_std)
    return upper, lower


# --- Trading Environment ---
class EnhancedTradingEnvironment:
    def __init__(self, data, window_size=Hyperparameters.WINDOW_SIZE):
        self.data = data
        self.window_size = window_size
        self.current_step = window_size
        self.max_steps = len(data) - 1
        self.action_space = 3  # 0=sell, 1=hold, 2=buy
        self.state_size = window_size * data.shape[1]
        self.position = 0
        self.commission = 0.001
        self.initial_balance = 10000
        self.current_balance = self.initial_balance
        self.positions = 0

    def reset(self):
        self.current_step = self.window_size
        self.position = 0
        self.current_balance = self.initial_balance
        self.positions = 0
        return self._get_state()
    
    def _get_state(self):
        return self.data[self.current_step - self.window_size : self.current_step].flatten()
    
    def step(self, action):
        current_price = self.data[self.current_step, 3]  # Current price
        next_price = self.data[self.current_step + 1, 3] if self.current_step < self.max_steps else current_price
        current_price = max(current_price, 1e-8)  # Avoid division by zero

        # Validate action
        valid_action = action
        if action == 2 and self.position == 1:   # Wants to buy but already invested
            valid_action = 1                     # Force to hold
        elif action == 0 and self.position != 1: # Wants to sell without holding
            valid_action = 1                     # Force to hold

        # --- Trading logic (buy/sell) ---
        if valid_action == 2:  # Buy
            buy_amount = min(self.current_balance * 0.1, self.current_balance)
            if current_price > 1e-8:
                self.positions += (buy_amount * (1 - self.commission)) / current_price
                self.current_balance -= buy_amount
                self.position = 1
        elif valid_action == 0:  # Sell
            sell_amount = min(self.positions * 0.1, self.positions)
            self.positions -= sell_amount
            self.current_balance += (sell_amount * current_price) * (1 - self.commission)
            self.position = 0

        # --- Reward calculation (for all actions, including "Hold") ---
        price_change = (next_price - current_price) / current_price if current_price != 0 else 0
        reward = 0  # Default value

        # Reward system
        if valid_action == 0:  # Sell
            reward = -price_change * 2.5  # Punish selling before rises
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
        
        return next_state, reward, done, {
            "price": current_price,
            "valid_action": valid_action,
            "portfolio_value": self.current_balance + self.positions * current_price
        }
# --- Red Neuronal LSTM ---
class EnhancedLSTMDQN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, action_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer with recurrent dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Fully connected layers with improved initialization
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

        # Dropout and activations
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()

        # Weight initialization
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
    
    def forward(self, x, hidden=None):
        # Reorganize data for LSTM
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.size(0)
        seq_len = Hyperparameters.WINDOW_SIZE
        input_size = x.size(1) // seq_len
        x = x.view(batch_size, seq_len, input_size)
        
        # Forward pass LSTM
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            hidden = (h0, c0)
        
        out, hidden = self.lstm(x, hidden)
        
        # Take only the last time step
        out = out[:, -1, :]

        # Fully connected layers
        out = self.activation(self.bn1(self.fc1(out)))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, hidden

# --- DQN Agent Enhanced with LSTM  ---
class LSTMDQNAgent:
    def __init__(self, input_size, hidden_size, num_layers, action_size):
        self.memory = deque(maxlen=50000)  
        self.gamma = Hyperparameters.GAMMA
        self.epsilon = Hyperparameters.EPS_START
        self.epsilon_min = Hyperparameters.EPS_END
        self.epsilon_decay = Hyperparameters.EPS_DECAY
        self.input_size = input_size
        self.action_size = action_size
        
        # LSTM Networks
        self.model = EnhancedLSTMDQN(input_size, hidden_size, num_layers, action_size).to(device)
        self.target_model = EnhancedLSTMDQN(input_size, hidden_size, num_layers, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(),
                                   lr=Hyperparameters.LR,
                                   weight_decay=1e-5)
        
        self.batch_size = Hyperparameters.BATCH_SIZE
        self.tau = Hyperparameters.TAU
        self.update_every = 4
        self.steps_done = 0
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, hidden=None, eval_mode=False):
        if not eval_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), hidden
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            q_values, new_hidden = self.model(state, hidden)
        self.model.train()
        return torch.argmax(q_values).item(), new_hidden
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Experience sampling
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(device)

        # Double DQN with LSTM
        current_q, _ = self.model(states)
        current_q = current_q.gather(1, actions.unsqueeze(1))
        
        next_actions, _ = self.model(next_states)
        next_actions = next_actions.max(1)[1]
        
        next_q, _ = self.target_model(next_states)
        next_q = next_q.gather(1, next_actions.unsqueeze(1))
        
        target = rewards + (1 - dones) * self.gamma * next_q.squeeze()

        # Loss calculation
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target.detach())
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()

        # Soft update of the target network
        self.soft_update_target_network()

        # # Epsilon decay
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

        self.steps_done += 1
    
    def soft_update_target_network(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def save_model(self, save_path, e, best_score, train_rewards):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),  
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': e,  
            'best_score': best_score,
            'train_rewards': train_rewards  # Reward history
        }, save_path)
        print(f"💾 Model saved at {save_path} (Episode {e}, ε={self.epsilon:.4f})")

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

            print(f"✅ Model loaded successfully on {device}")
            print(f"| ε: {self.epsilon:.4f} | Best Score: {best_score:.2f}% |")

            return self, best_score, train_rewards
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")

            raise

# --- Evaluation Function ---
def evaluate(agent, env, df, initial_balance=10000, return_full_history=False):
    state = env.reset()
    hidden = None
    portfolio_history = [initial_balance]
    price_history = []
    actions_history = []
    rewards_history = []
    done = False
    
    while not done:
        action, hidden = agent.act(state, hidden, eval_mode=True)
        next_state, reward, done, info = env.step(action)
        
        # Denormalize price
        price_range = df['close_price'].max() - df['close_price'].min()
        current_price = info['price'] * price_range + df['close_price'].min()

        # Log information
        portfolio_history.append(info['portfolio_value'])
        price_history.append(current_price)
        actions_history.append(action)
        rewards_history.append(reward)
        
        state = next_state
    
    final_return = (portfolio_history[-1] / initial_balance - 1) * 100
    
    if return_full_history:
        return final_return, portfolio_history, price_history, actions_history, rewards_history
    else:
        return final_return, portfolio_history

if __name__ == "__main__":
    # Load data
    data_array, df, scaler = load_and_preprocess_data('C:\\Andrey\\Kakua_Projets\\Trading\\Bot_RL_v1\\Datasets\\historical_01-01-2023_to_01-01-2025_ETHUSDT.csv')

    """ --- Data Splitting (80% train, 20% test) --- """

    train_size = int(0.8 * len(data_array))
    train_data = data_array[:train_size]
    test_data = data_array[train_size:]

    print(f"\nData Splitting:")
    print(f"Total: {len(data_array)}")
    print(f"Training: {len(train_data)}")
    print(f"Evaluation: {len(test_data)}")

    # --- Create Environments ---
    train_env = EnhancedTradingEnvironment(train_data)
    test_env = EnhancedTradingEnvironment(test_data)

    # Sizes for LSTM
    input_size = train_data.shape[1]
    action_size = train_env.action_space

    # --- Training Configuration ---
    agent = LSTMDQNAgent(input_size, Hyperparameters.HIDDEN_SIZE, 
                        Hyperparameters.NUM_LAYERS, action_size)

    episodes = Hyperparameters.TRAIN_EPISODES
    save_path = 'best_trading_model.pth'
    best_score = -np.inf
    no_improve = 0
    patience = Hyperparameters.PATIENCE

    # Dimension verification
    print(f"NNumber of features: {train_data.shape[1]}")
    print(f"Window size: {Hyperparameters.WINDOW_SIZE}")
    print(f"State size calculated: {Hyperparameters.WINDOW_SIZE * train_data.shape[1]}")

    # --- Training ---
    print("\nStarting training with optimized hyperparameters...")
    train_rewards = []
    val_returns = []

    # Each episode is a complete pass through the training data
    for e in range(episodes):
        state = train_env.reset() # Reset to the start of the data
        hidden = None
        total_reward = 0
        done = False
        
        while not done:
            action, hidden = agent.act(state, hidden)
            next_state, reward, done, _ = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > agent.batch_size:
                agent.replay()
        
        train_rewards.append(total_reward)
        
        # Validación periódica
        if e % 5 == 0:
            val_return, _, _, actions, _ = evaluate(agent, test_env, df, return_full_history=True)
            val_returns.append(val_return)
            elapsed = (time.time() - start_time) / 3600

            # Calculate action distribution
            actions_dist = pd.Series(actions).value_counts(normalize=True)

            print(f"Episode: {e+1}/{episodes}, "
                f"Training Reward: {total_reward:.2f}, "
                f"Validation Return: {val_return:.2f}%, "
                f"ε: {agent.epsilon:.4f}, "
                f"Steps: {agent.steps_done}, "
                f"Time: {elapsed:.2f}h")
            print(f"Actions: Buy={actions_dist.get(2, 0):.1%}, "
                  f"Sell={actions_dist.get(0, 0):.1%}, "
                  f"Hold={actions_dist.get(1, 0):.1%}")
            # Save the best model
            if val_return > best_score:
                best_score = val_return
                no_improve = 0
                agent.save_model(save_path,e,best_score,train_rewards) 
                
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\nEarly stopping at episode {e+1} - No improvement for {patience} evaluations")
                    break

        # Decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        if e % 2 == 0:
            print('e: ',agent.epsilon)

    """------------------------- Final evaluation and visualization -------------------------"""
    # Load the best model before final evaluation
    agent, best_score, train_rewards = agent.load_model(save_path, device)

    # --- Final Evaluation ---
    print("\nEvaluating on test data...")
    final_return, portfolio_history, price_history, actions_history, rewards_history = evaluate(
        agent, test_env, df, return_full_history=True
    )

    # --- Metric Calculation ---
    final_value = portfolio_history[-1]
    returns = np.diff(portfolio_history) / portfolio_history[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    max_drawdown = (np.maximum.accumulate(portfolio_history) - portfolio_history).max()
    max_drawdown_pct = (max_drawdown / np.maximum.accumulate(portfolio_history)).max()
    buy_hold_return = (price_history[-1] / price_history[0] - 1) * 100
    volatility = np.std(returns) * np.sqrt(252) * 100
    win_rate = len([r for r in returns if r > 0]) / len(returns) * 100

    print("\n--- Optimized Final Results ---")
    print(f"Initial Value: $10,000.00")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Strategy Return: {(final_value/10000-1)*100:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"\n--- Risk-Return Metrics ---")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Annualized Volatility: {volatility:.2f}%")
    print(f"Max Drawdown: ${max_drawdown:,.2f} ({max_drawdown_pct:.2%})")
    print(f"Win Rate: {win_rate:.2f}%")

    # Detailed action distribution
    actions_dist = pd.Series(actions_history).value_counts(normalize=True).sort_index()
    print("\nDetailed Action Distribution:")
    for action, percent in zip(['Sell', 'Hold', 'Buy'], actions_dist):
        print(f"{action}: {percent:.1%}")

    # Trade Analysis
    positive_trades = len([r for r in returns if r > 0])
    negative_trades = len([r for r in returns if r < 0])
    avg_win = np.mean([r for r in returns if r > 0]) * 100
    avg_loss = np.mean([r for r in returns if r < 0]) * 100
    profit_factor = -avg_win * positive_trades / (avg_loss * negative_trades) if negative_trades > 0 else np.inf

    print("\n--- Trade Analysis ---")
    print(f"Positive Trades: {positive_trades} ({positive_trades/len(returns):.1%})")
    print(f"Negative Trades: {negative_trades} ({negative_trades/len(returns):.1%})")
    print(f"Average Win: {avg_win:.2f}%")
    print(f"Average Loss: {avg_loss:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")

    # --- Visualization ---
    plt.figure(figsize=(18, 12))
    plt.suptitle('Optimized Trading Bot Results with LSTM', fontsize=16)

    # Subplot 1: Price vs Portfolio Value
    plt.subplot(2, 2, 1)
    plt.plot(price_history, label='ETH Price', color='blue', alpha=0.6)
    plt.ylabel('Price (USD)')
    plt.legend(loc='upper left')
    plt.grid(True)

    ax2 = plt.gca().twinx()
    ax2.plot(portfolio_history[1:], label='Portfolio Value', color='green')
    ax2.axhline(y=10000, color='red', linestyle='--', label='Initial Investment')
    ax2.set_ylabel('Value (USD)')
    ax2.legend(loc='upper right')
    plt.title('Price vs Portfolio Value')

    # Subplot 2: Action Distribution
    plt.subplot(2, 2, 2)
    action_names = ['Sell', 'Hold', 'Buy']
    action_counts = pd.Series(actions_history).value_counts()
    plt.bar(action_names, action_counts, color=['red', 'gray', 'green'])
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.title('Action Distribution')
    plt.grid(True)

    # Subplot 3: Cumulative Rewards
    plt.subplot(2, 2, 3)
    cumulative_rewards = np.cumsum(rewards_history)
    plt.plot(cumulative_rewards, label='Cumulative Rewards', color='purple')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.title('Cumulative Rewards')
    plt.legend()
    plt.grid(True)

    # Subplot 4: Training Progress
    plt.subplot(2, 2, 4)
    plt.plot(train_rewards, label='Training Reward', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    