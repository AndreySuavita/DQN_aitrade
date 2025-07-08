import torch
from datetime import datetime
import sys
sys.path.append("../")
from binance_test import binance_test

"""------------------------------ MAIN EXECUTION ------------------------------"""

# Initial configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

print(f"\nProcess start at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

window_size = 180
initial_balance = 10000 # USD
time_cycle = '5m'  
binance_on = True # If True, Get data from binance kindle
with_binance_balance = True # If True, use Binance balance for the test
time_to_wait = 5 

# Run the Binance test
binance_test(device, window_size, time_cycle, initial_balance, binance_on, time_to_wait, with_binance_balance)