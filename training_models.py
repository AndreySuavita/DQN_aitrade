import numpy as np
import pandas as pd
import time
from utils import evaluate, plot_evaluation_py

def training_model(episodes, time_cycle, start_time, patience, save_path, agent, train_env, test_env, scaler, initial_balance, mean_fold_results, std_fold_results, fold_results):
    train_rewards = []  # Store rewards for each training episode
    episode_counts = [] # Store episode counts for plotting
    eval_rewards = [] # Store evaluation rewards for plotting
    best_score = -np.inf  # Initialize the best score for early stopping
    no_improve = 0 # Counter for early stopping

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
            val_return, _, _ , actions= evaluate(agent, test_env, scaler, initial_balance, time_cycle = time_cycle)
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
        best_score=-np.inf,  # Not used in this case
        mean_fold_results=mean_fold_results,
        std_fold_results=std_fold_results,
        fold_results=fold_results,
        train_rewards=train_rewards
    )