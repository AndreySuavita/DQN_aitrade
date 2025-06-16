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
from datetime import datetime, timedelta, timezone
import pickle

from Double_DQN_hourly import EnhancedDQNAgent, metrics, plot_results

import sys
sys.path.append("../")
from binance_actions import binance_actions

def wait_until_next_hour_with_offset(offset_seconds=10):
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

class EnhancedTradingEnvironment:
    def __init__(self, scaler, window_size=15):
        
        self.scaler = scaler
        self.window_size = window_size # Visible history (15 steps)
        self.data, _, _ = self.load_and_preprocess_data()   # Normalized data
        self.action_space = 3  # Possible actions: 0=sell, 1=hold, 2=buy
        self.state_size = window_size * self.data.shape[1] # Flattened state size (window * features) (8*15)
        self.commission = 0.001  # Commission of 0.1% per trade

    def step(self):
        # Price Calculation
        # print(self.data.shape)
        #exit()
        current_price = self.data[self.window_size-1, 3]  # Current close price
        # print(current_price)
        # exit()
        return {"price": current_price}
    def reset(self):
        self.data, _, _ = self.load_and_preprocess_data()
        data_flattened = self.data.flatten()  # Flatten the data
        # print(self.data.shape)
        # print(data_flattened.shape)
        # exit()
        return data_flattened # Flattened data window

    # --- Load and preprocess data ---
    def load_and_preprocess_data(self):
        """Load and preprocess historical data"""
        binance = binance_actions()
        data=binance.get_klines(symbol="ETHUSDT", interval='1h', limit=49+self.window_size)
        df = data.set_index('close_time')
        # print(df.shape)
        # exit()

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

        # Normalize the data using MinMaxScaler
        df_normalized = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns)
        # print(df)
        # exit()
        return df_normalized.values, df, self.scaler
      
def evaluate(agent, env, scaler, initial_balance=10000):

    portfolio = initial_balance       # Example: $10,000 initial USD
    positions = 0                     # Amount of assets (ETH) held
    portfolio_history = [portfolio]   # Record portfolio value at each step
    price_history = []                # Store denormalized prices
    actions_history = []              # Record actions taken    

    # Prepare an empty array for scaling investment
    temp_array = np.zeros((1, len(scaler.feature_names_in_)))  # Use the same dimensionality as the scaler
    #while not done:
    for hour in list(range(10)):
        # Wait until 10 seconds after the next UTC hour
        #wait_until_next_hour_with_offset(10) 
        time.sleep(5)

        state = env.reset()  
        action = agent.act(state) # Choose action (0, 1, 2)
        info = env.step() # Execute action in the environment

        #print(scaler.feature_names_in_) check column order

        # Denormalization using the scaler - INDEX 3 FOR close_price
        temp_array.fill(0)  # Clear the temporary array
        temp_array[0, 3] = info['price']  # Place the normalized value in position 3 (close_price)
        denormalized = scaler.inverse_transform(temp_array)
        current_price = denormalized[0, 3]  # Get the denormalized close_price
        #print('current_price :',current_price)
        #exit()

        # Conservative trading logic
        if action == 2 and portfolio > 0: #buy
            buy_amount = portfolio * 0.1 # invest 10% of the portfolio
            positions += (buy_amount * (1 - env.commission)) / current_price  # Buy ETH
            portfolio -= buy_amount # Reduce cash
        elif action == 0 and positions > 0: # sell
            sell_amount = positions * 0.1 # Sell 10% of the ETH held
            portfolio += (sell_amount * current_price) * (1 - env.commission) # Convert to USD
            positions -= sell_amount  # Reduce ETH position

        current_value = portfolio + positions * current_price  # Total value (USD + ETH)
        portfolio_history.append(current_value)  # Record current value
        price_history.append(current_price)     # Record price history
        actions_history.append(action)          # Record action
        print('curret_value:',current_value)
        print('current_price',current_price)
        print('actions:',action)
    
        
    final_return = (portfolio_history[-1] / initial_balance - 1) * 100 # Percentage return
    
    return final_return, portfolio_history, price_history, actions_history

"""------------------------------ MAIN EXECUTION ------------------------------"""
if __name__ == "__main__":
    # Initial configuration
    start_time = time.time()
    logging.basicConfig(filename='trading_bot.log', level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    window_size = 15
    initial_balance = 10000 # USD

    # Load the scaler
    with open('eth_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    test_env = EnhancedTradingEnvironment(scaler, window_size)
    state_size = test_env.state_size
    action_size = test_env.action_space
    save_path = 'best_trading_model.pth'

    # Initialize agent 
    agent = EnhancedDQNAgent(state_size, action_size, device)

    # Load model 
    agent, best_score, train_rewards = agent.load_model(save_path)

    # --- Final Evaluation ---
    print("\nEvaluating with test data...")
    final_return, portfolio_history, price_history, actions_history = evaluate(
        agent, test_env, scaler, initial_balance)

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
    
    

