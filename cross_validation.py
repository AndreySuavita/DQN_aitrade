import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import os
import sys
import torch
sys.path.append("../")
from utils import evaluate
from environments import EnhancedTradingEnvironment
from agents import EnhancedDQNAgent

def time_series_cv_train(agent, best_params, full_train_data, window_size, initial_balance, scaler, time_cycle, episodes=200, n_splits=5, patience=5 ):
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
            time_cycle=time_cycle,
            device=agent.device,
            **best_params  
        )
        
        train_fold = full_train_data[train_idx]
        val_fold = full_train_data[val_idx]
        
        train_env = EnhancedTradingEnvironment(train_fold, window_size, time_cycle, scaler)
        val_env = EnhancedTradingEnvironment(val_fold, window_size, time_cycle, scaler)
        
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
                val_return, _, _, _ = evaluate(fold_agent, val_env, scaler, initial_balance, time_cycle=time_cycle)
                
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
        final_return, _, _, _ = evaluate(fold_agent, val_env, scaler, initial_balance, time_cycle=time_cycle)
        fold_results.append(final_return)
        all_train_rewards.extend(fold_train_rewards)
        
        print(f"Fold {fold+1} completed. Return: {final_return:.2f}%")
    
    return np.mean(fold_results), np.std(fold_results), fold_results, agent, all_train_rewards

# --- Hyperparameter Optimization ---
def optimize_hyperparams(full_train_data, window_size, initial_balance, param_grid, scaler, time_cycle, n_splits=3, episodes=50, patience=3, device='cpu'):
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
                time_cycle=time_cycle,
                device=device,
                **params
            )
            
            train_fold = full_train_data[train_idx]
            val_fold = full_train_data[val_idx]
            
            train_env = EnhancedTradingEnvironment(train_fold, window_size, time_cycle, scaler)
            val_env = EnhancedTradingEnvironment(val_fold, window_size, time_cycle, scaler)
            
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
                    val_return, _, _, _ = evaluate(agent, val_env, scaler, initial_balance, time_cycle=time_cycle)
                    
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