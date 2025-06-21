import logging
import sys
sys.path.append("../")
from utils import load_and_preprocess_data

# --- Trading Environment ---
class EnhancedTradingEnvironment:
    def __init__(self, data, window_size, time_cycle, scaler, binance_on=False):
        self.data = data # Normalized data
        self.window_size = window_size # Visible history 
        self.time_cycle = time_cycle # Time cycle for the data
        self.scaler = scaler # Scaler for the data
        self.binance_on = binance_on # If True, use Binance API for real-time data
        self.current_step = window_size # Current step, starts after having enough history
        self.max_steps = len(data) - 1 # Last possible step
        self.action_space = 3 # Possible actions: 0=sell, 1=hold, 2=buy
        self.state_size = window_size * self.data.shape[1] # Flattened state size (window * features) 
        self.position = 0 # 0=not invested, 1=invested (in ETH)
        self.commission = 0.001 # Commission of 0.1% per operation
    
    def reset(self):
        if self.binance_on:
            self.data, _, _ = load_and_preprocess_data(window_size=self.window_size, time_cycle=self.time_cycle, scaler=self.scaler, binance_on=self.binance_on)
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

            if self.time_cycle == '5m':
                # Reward system
                if valid_action == 0: # Sell
                    reward = -price_change * 1.5 # Punish selling before rises
                    self.position = 0
                elif valid_action == 2: # Buy
                    reward = price_change * 1.2 # Reward successful buys
                    self.position = 1
                else: # Hold
                    reward = 0.2 if abs(price_change) < 0.01 else -0.1 # Reward holding in sideways markets
            elif self.time_cycle == 'hourly':
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
        
