import pandas as pd
import torch
import logging
import time
import pickle
from Double_DQN_hourly_SSTCV_OP_V_T import EnhancedDQNAgent, EnhancedTradingEnvironment, evaluate, metrics, plot_results, load_and_preprocess_data

"""------------------------------ MAIN EXECUTION ------------------------------"""

# Initial configuration
start_time = time.time()
logging.basicConfig(filename='trading_bot.log', level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

window_size = 15
initial_balance = 10000 # USD
binance_on = True
hours_to_wait = 24 

# Load and preprocess data
data, _, _ = load_and_preprocess_data(window_size=window_size, load_scaler=True, binance_on=binance_on)
    
# Initialize the trading environment
test_env = EnhancedTradingEnvironment(data, window_size, binance_on)
state_size = test_env.state_size
action_size = test_env.action_space
save_path = 'best_trading_model.pth'

# Initialize agent 
agent = EnhancedDQNAgent(state_size, action_size, device)

# load the saved model
agent, best_score, episode, mean_fold_results, std_fold_results, fold_results, train_rewards = agent.load_model(save_path)

# Final Evaluation
print("\nEvaluating with test data...")

# Load the scaler
with open('eth_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

test_return, portfolio_history, price_history, actions_history = evaluate(
    agent, test_env, scaler, initial_balance, binance_on=hours_to_wait
)

#  Final results and metrics
metrics(portfolio_history, test_return, price_history, actions_history, initial_balance, mean_fold_results, std_fold_results)

print(f"\nTotal execution time: {(time.time() - start_time)/3600:.2f} hours")

#  Results visualization
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
    save_img=f'Binance_test_result_{hours_to_wait}.png'
)
    
    

