from binance_actions import wait_until_next_time_cycle, binance_actions
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# --- Load and preprocess data ---
def load_and_preprocess_data(window_size, time_cycle, filepath='', scaler=False, binance_on=False):
    """Loads and preprocesses historical data
    
    Args:
        filepath (str): Path to the CSV file with historical data
        load_scaler (bool): If True, loads and uses the previously saved scaler.
            If False, creates and saves a new scaler.
        binance_on (bool): If True, uses the Binance API to fetch data.
    
    Returns:
        tuple: (normalized_data, original_data, scaler)
    """
    
    if time_cycle == '5m':
        window_1 = 12   # 1 hora (12 velas de 5m)
        window_2 = 72   # 6 horas
        window_3 = 288  # 24 horas (útil para soportes/resistencias diarias)
        max_window = max(window_1, window_2, window_3)
        if binance_on:
            binance = binance_actions()
            data=binance.get_klines(symbol="ETHUSDT", interval='5m', limit=max_window - 1 + window_size)
            df = data.set_index('close_time')
            # print(df.shape)
            # exit()
        else:
            df = pd.read_csv(filepath, index_col='close_time')
        # Data cleaning
        df = df.replace([np.inf, -np.inf], np.nan).ffill()
        df.index = pd.to_datetime(df.index, unit='ms')  # Timestamp en milisegundos
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')  # Si también es timestamp

        # Add basic technical indicators
        df[f'MA_{window_1}'] = df['close_price'].rolling(window=window_1).mean()
        df[f'MA_{window_2}'] = df['close_price'].rolling(window=window_2).mean()
        df[f'MA_{window_3}'] = df['close_price'].rolling(window=window_3).mean()
        df['Percentage_return'] = df['close_price'].pct_change()
        # Add time-based features

        df['minute_of_day'] = df.index.minute + df.index.hour * 60
        df['sin_time'] = np.sin(2 * np.pi * df['minute_of_day'] / (24 * 60))
        df['cos_time'] = np.cos(2 * np.pi * df['minute_of_day'] / (24 * 60))
        # Volatility features
        df['5m_volatility'] = df['close_price'].pct_change().rolling(12).std()
        # RSI de 3 periodos (sobre-reacción en marcos cortos)
        df['RSI_12'] = ta.rsi(close=df['close_price'], length=12)  

        # Volumen relativo (vs media móvil de 1h)
        df['volume_ma_12'] = df['close_volume'].rolling(12).mean()
        df['volume_ratio'] = df['close_volume'] / df['volume_ma_12']
        # Add session feature (New York session)
        df['is_ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 20)).astype(int)  # 8AM-3PM ET
        # Select features
        selected_features = ['open_price', 'high_price', 'low_price', 'close_price', 
                            'close_volume', f'MA_{window_1}', f'MA_{window_2}', f'MA_{window_3}', 'Percentage_return',
                            'sin_time', 'cos_time', '5m_volatility','RSI_12', 'volume_ratio','is_ny_session']
    elif time_cycle == 'hourly':
        window_1 = 10
        window_2 = 50
        max_window = max(window_1, window_2)
        if binance_on:
            binance = binance_actions()
            # print('window_size: ',window_size)
            # print(max_window - 1 + window_size)
            data=binance.get_klines(symbol="ETHUSDT", interval='1h', limit = max_window - 1 + window_size)
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
    
    # Normalize data, load or create scaler, or use existing scaler
    if scaler:
        try:
            # Ensure columns match the original scaler
            # print(df.columns)
            # print(scaler.feature_names_in_)
            if set(df.columns) != set(scaler.feature_names_in_):
                raise ValueError("Features in data don't match scaler features")
            # print('df')
            # print(df)
            df_normalized = pd.DataFrame(scaler.transform(df), columns=df.columns)
            
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

# --- Enhanced Evaluation Function ---
def evaluate(agent, env, scaler, initial_balance=10000, binance_on=False, time_cycle = 'hourly' , with_binance_balance=False):
    '''
    The evaluate function has two key objectives:
    Assess the agent's performance with a conservative trading strategy
    Simulate real trades with capital management (only invests 10% of the portfolio in each trade)

    Input parameters:
    - agent: The DQN agent we have trained (contains the neural network model)
    - env: The trading environment (can be training or testing)
    - scaler: The MinMaxScaler used to normalize the data
    - initial_balance: Initial capital for the simulation (default $10,000)
    - binance_on: If True, uses Binance API for real-time data and actions
    - time_cycle: Time cycle for the data (e.g., 'hourly', '5m')
    '''
    
    state = env.reset()
    portfolio = initial_balance       # Example: $10,000 initial USD
    positions = 0                     # Amount of assets (ETH) held
    portfolio_history = []            # Record portfolio value at each step
    price_history = []                # Store denormalized prices
    actions_history = []              # Record actions taken  
    done = False                      # Simulation end flag
    step_number = 0                   # Counter for Binance connection    
    only_portfolio = []
    
    
    # If connected full binance, get last data from API
    if with_binance_balance and binance_on:
        binance = binance_actions()
        def get_balance(binance):
            result = binance.get_balance("ETH") # Get current ETH balance
            portfolio = result['USDT']['free']# Get current ETH balance in binance account
            positions = result['ETH']['free'] # Get current USDT balance in binance account
            return portfolio, positions
        portfolio, positions = get_balance(binance)

        print('[+] Current portfolio: ',portfolio)
        print('[+] Current positions: ',positions)

    

    # Prepare an empty array for scaling investment
    temp_array = np.zeros((1, len(scaler.feature_names_in_))) # Use the same dimensionality as the scaler

    while not done:     
        # If Binance is connected, use its step method
        if binance_on:
            # Wait until 10 seconds after the next UTC hour
            # wait_until_next_time_cycle(time_cycle=time_cycle) # Wait for the next time cycle
            time.sleep(5)
            state = env.reset()
            action = agent.act(state) # Choose action (0, 1, 2)  
            info = env.step(action) # Execute action in the environment
        else:
            action = agent.act(state) # Choose action (0, 1, 2)
            next_state, _, done, info = env.step(action) # Execute action in the environment
        # print(scaler.feature_names_in_) check column order

        portfolio_amount_to_use = portfolio * 0.05 if time_cycle == 'hourly' else portfolio * 0.03
        positions_amount_to_use = positions * 0.05 if time_cycle == 'hourly' else positions * 0.03

        # Denormalization using the scaler - INDEX 3 FOR close_price
        temp_array.fill(0) # Clear the temporary array
        temp_array[0, 3] = info['price'] # Place the normalized value in position 3 (close_price)
        denormalized = scaler.inverse_transform(temp_array)
        current_price = denormalized[0, 3] # Get the denormalized close_price

        ''' Conservative trading logic '''
        # If Binance and balance is connected, buy/sell ETH using Binance API
        if with_binance_balance and binance_on:
            # logic to buy and sell
            if action == 2 and portfolio > 0: # buy
                binance.trade_eth_usdt(amount_usdt = portfolio_amount_to_use, side='buy')
            elif action == 0 and positions > 0: # sell
                positions_amount_to_use_USDT = (positions_amount_to_use * current_price) # get the position in USDT
                binance.trade_eth_usdt(amount_usdt = positions_amount_to_use_USDT, side='sell')
            # Get balance and position for account
            portfolio, positions = get_balance(binance)
            
        else: # if training or testing binance without buy and sell
            if action == 2 and portfolio > 0: # buy
                buy_amount = portfolio_amount_to_use # invest 10% of the portfolio
                positions += (buy_amount * (1 - env.commission)) / current_price # Buy ETH
                portfolio -= buy_amount # Reduce cash
            elif action == 0 and positions > 0: # sell
                sell_amount = positions_amount_to_use # Sell 10% of the ETH held
                portfolio += (sell_amount * current_price) * (1 - env.commission) # Convert to USD
                positions -= sell_amount # Reduce ETH position

        current_value = portfolio + positions * current_price # Total value (USD + ETH)
        portfolio_history.append(current_value)
        price_history.append(current_price)
        actions_history.append(action)
        
        # save lowest portfolio value
        if portfolio <=(initial_balance/2):
            only_portfolio.append(portfolio)
            
        # If Binance is connected
        if binance_on:
            print('[-] Portfolio: ',portfolio)
            print('[-] Positions: ',positions)
            print('[-] Current_value: ',current_value)
            print('[-] Current_price (EHT): ',current_price)
            if action == 2:
                print(f'[-] Action: {action} (Buy)')
            elif action == 0:
                print(f'[-] Action: {action} (Sell)')
            else:
                print(f'[-] Action: {action} (Hold)')
            step_number += 1
            print(f'[-] Step_number: {step_number}')
            timestamp = time.time()
            fecha_legible = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

            print(f'[-] Action_Time: {fecha_legible}')
            print('[+]------------------------------------------------')
            # Plot every step
            test_results = {
                'portfolio_history': portfolio_history,
                'price_history': price_history,
                'actions_dist': pd.Series(actions_history).value_counts(normalize=True).to_dict()
            }
            plot_results(test_results=test_results,
                         actions_history=actions_history,
                         initial_balance=initial_balance,
                        save_img=f'Binance_test_result_{binance_on}.png',
                        plot=False)
            # Save results step by step in csv
            df_new = pd.Series({'Portfolio':portfolio,'Positions':positions,'Current_value':current_value,'Current_price':current_price,'Action':action,
                                'Step_number':step_number,'Action_time':timestamp})
            df_new = df_new.to_frame().T.set_index('Action_time')
            file_csv = f'Binance_results_live.csv'
            try:
                df_old = pd.read_csv(file_csv, index_col='Action_time')
                Binance_results_live = pd.concat([df_old, df_new])
            except:
                Binance_results_live = df_new

            # Save new Dataframe with live Values
            Binance_results_live.to_csv(file_csv)

            # Wait number_steps steps to finish the evaluation
            if step_number >= binance_on:
                done = True
        else:
            state = next_state

    if only_portfolio:
        print('[+] Min Value reached portfolio',min(only_portfolio))

    final_return = (portfolio_history[-1] / portfolio_history[0] - 1) * 100 # Percentage return
    
    return final_return, portfolio_history, price_history, actions_history

# --- Metric Calculation ---
def metrics(portfolio_history, test_return, price_history, actions_history, initial_balance, mean_fold_results, std_fold_results, time_cycle):
    final_value = portfolio_history[-1]
    returns = np.diff(portfolio_history) / portfolio_history[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 if time_cycle == 'hourly' else 252*24*12)
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

# PLotting Function
def plot_results(fold_results=False, train_rewards=False, initial_balance=False, actions_history=False, test_results=False, time_cycle=False, save_img=False, eval_img_path=False, plot=True):
    plt.figure(figsize=(20, 12))  

    """
    Visualiza 6 métricas clave de estrategias de trading algorítmico:
    1. Evolución de precio vs valor de portafolio
    2. Rendimiento por fold (SSTCV)
    3. Recompensas durante el entrenamiento
    4. Historial de acciones (Buy/Hold/Sell)
    5. Distribución de acciones
    6. Retornos de evaluación (opcional)
    """
    # Título general
    #plt.subplots_adjust(top=0.9)  # Ajustar el espacio para el título
    plt.suptitle(f'Comprehensive Performance Analysis - Algorithmic Trading ({time_cycle})', 
                fontsize=16, y=1.02, fontweight='bold')

    # --- Gráfica 1: Price vs Portfolio Value ---
    if test_results:
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
        plt.title('Price vs Portfolio Value')

    # --- Gráfica 2: Performance by fold ---
    if fold_results:
        plt.subplot(2, 3, 2)  # Fila 2, Col 1
        plt.bar(range(1, len(fold_results)+1), fold_results, color='skyblue')
        plt.axhline(y=np.mean(fold_results), color='r', linestyle='--', label='Average')
        plt.title('Performance by Fold in SSTCV')
        plt.xlabel('Fold')
        plt.ylabel('Return (%)')
        plt.legend()
        plt.grid(True)

    # --- Gráfica 3: Training rewards ---
    if train_rewards:
        plt.subplot(2, 3, 3)  # Fila 1, Col 2
        plt.plot(train_rewards, label='Reward', color='purple')
        plt.title('Rewards During Training')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)

    # --- Gráfica 4: Action Timeline ---
    if actions_history:
        plt.subplot(2, 3, 4)  # Fila 1, Col 3
        plt.plot(actions_history, 'o', markersize=2, alpha=0.6)
        plt.yticks([0, 1, 2], ['Sell', 'Hold', 'Buy'])
        plt.xlabel('Time Step (hours)')
        plt.ylabel('Action')
        plt.title('Action Timeline')
        plt.grid(True)

    if test_results:
        # --- Gráfica 5: Actions Distribution ---
        plt.subplot(2, 3, 5)  # Fila 2, Col 2
        actions_dist = test_results['actions_dist']
        plt.bar(['Sell', 'Hold', 'Buy'], 
                [actions_dist.get(0, 0), actions_dist.get(1, 0), actions_dist.get(2, 0)])
        plt.title('Actions Distribution in Test')
        plt.ylabel('Proportion')

    # --- Gráfica 6: Evaluation Returns ---
    if eval_img_path:
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
    if plot:
        print('printing img')
        print(plot)
        plt.show()
    plt.close()

# --- Plotting Evaluation Returns ---
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