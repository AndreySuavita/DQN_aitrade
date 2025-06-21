import pandas as pd
import time
import pickle
from environments import EnhancedTradingEnvironment
from agents import EnhancedDQNAgent
from utils import evaluate, metrics, plot_results, load_and_preprocess_data
from binance_actions import binance_actions

def binance_test(device, window_size, time_cycle, initial_balance, binance_on, time_to_wait, with_binance_balance):
    """Function to test the model with Binance data"""
    start_time = time.time()
    
    # Load the scaler
    with open('eth_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print('Loaded existing scaler')

    # Load and preprocess data
    data, _, _ = load_and_preprocess_data(window_size=window_size, time_cycle=time_cycle, scaler=scaler, binance_on=binance_on)

    # Initialize the trading environment
    test_env = EnhancedTradingEnvironment(data, window_size, time_cycle, scaler, binance_on)
    state_size = test_env.state_size
    action_size = test_env.action_space
    save_path = 'best_trading_model.pth'

    # Initialize agent 
    agent = EnhancedDQNAgent(state_size, action_size, time_cycle, device)

    # load the saved model
    agent, best_score, episode, mean_fold_results, std_fold_results, fold_results, train_rewards = agent.load_model(save_path)

    # Binance Evaluation
    print("\nEvaluating with Binance data...")

    test_return, portfolio_history, price_history, actions_history = evaluate(
        agent, test_env, scaler, initial_balance=initial_balance, binance_on=time_to_wait, time_cycle=time_cycle,with_binance_balance=with_binance_balance
    )

    # Update initial balance with the value of the 
    initial_balance = portfolio_history[0] 

    #  Final results and metrics
    metrics(portfolio_history, test_return, price_history, actions_history, initial_balance, mean_fold_results, std_fold_results, time_cycle)

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
        time_cycle=time_cycle,
        save_img=f'Binance_test_result_{time_to_wait}.png'
    )
    