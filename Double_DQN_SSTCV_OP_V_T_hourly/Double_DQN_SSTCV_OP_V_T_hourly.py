import pandas as pd
import torch
import logging
import time
from datetime import datetime
import sys
sys.path.append("../")
from environments import EnhancedTradingEnvironment
from agents import EnhancedDQNAgent
from utils import evaluate, metrics, plot_results, load_and_preprocess_data
from training_models import training_model
from cross_validation import optimize_hyperparams, time_series_cv_train
"""
    Model created using Split Series Time Cross Validation (SSTCV) with Double DQN.
    This model is designed to trade ETH/USDT on Binance using hourly data.
    SSTCV is used to optimize the model's performance by training on different time splits independently using a separate agent for each fold.
    the final training is done on the entire dataset using the best parameters found during the optimization.
"""

if __name__ == "__main__":
    # --- Log and device configuration ---
    start_time = time.time()
    logging.basicConfig(filename='trading_bot.log', level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nProcess start at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Parameters ---
    window_size = 15
    initial_balance = 10000 # USD
    episodes = 6
    time_cycle = 'hourly'  
    save_path = 'best_trading_model.pth'
    patience = 50
    
    # 1. Load and prepare data
    data_array, df, scaler = load_and_preprocess_data(window_size=window_size, time_cycle=time_cycle, filepath='C:\\Andrey\\Kakua_Projets\\Trading\\Bot_RL_v1\\Datasets\\historical_01-01-2019_to_01-01-2025_ETHUSDT.csv')
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
    #     full_train_data, window_size, initial_balance, param_grid, scaler, time_cycle,
    #     n_splits=3, episodes=50, patience=3, device=device
    # )

    # print(f"\nBest parameters found: {best_params}")

    best_params = {
        'gamma': 0.95,
        'lr': 0.0001,
        'batch_size': 32
    }
    
    # 3. Final training with best parameters

    agent = EnhancedDQNAgent(
        state_size = window_size * full_train_data.shape[1],
        action_size=3,
        time_cycle=time_cycle,
        device=device,
        **best_params
    )

    ## Commented out the cross-validation training for now
    # print("\nSplit time series cross-validation with best parameters...")
    # mean_fold_results, std_fold_results, fold_results, trained_agent, train_rewards_cv = time_series_cv_train(
    #     agent, best_params, full_train_data, window_size, initial_balance, scaler, time_cycle,
    #     episodes=episodes, n_splits=5, patience=5
    # )

    # print('mean_fold_results',mean_fold_results)
    # print('std_fold_results',std_fold_results)
    # print('fold_results',fold_results)

    mean_fold_results = 89.84351506364882
    std_fold_results = 150.12635790298214
    fold_results = [384.4632589509047, 55.16628321984469, -34.54628107093532, 25.688069424624803, 18.4462447938053]
    
    # 4. Final training on entire dataset
    
    # --- Create environments ---
    train_env = EnhancedTradingEnvironment(full_train_data, window_size, time_cycle, scaler)
    test_env = EnhancedTradingEnvironment(test_data, window_size, time_cycle, scaler)

    # using best model from CV (optional)
    # agent = trained_agent

    print("\nStarting training...")
    training_model(episodes, time_cycle, start_time, patience, save_path, agent, train_env, test_env, scaler, initial_balance, mean_fold_results, std_fold_results, fold_results)

    # 5. Evaluation on test set

    # load the saved model
    agent, best_score ,episode, mean_fold_results, std_fold_results, fold_results, train_rewards = agent.load_model(save_path)
    
    print("\nEvaluating on test set...")

    test_return, portfolio_history, price_history, actions_history = evaluate(
        agent, test_env, scaler, initial_balance, time_cycle = time_cycle
    )

    # 6. Final results and metrics
    metrics(portfolio_history, test_return, price_history, actions_history, initial_balance, mean_fold_results, std_fold_results, time_cycle)

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
        time_cycle=time_cycle,
        save_img='Final_test_result.png'
    )
