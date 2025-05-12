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

# Configuración inicial
start_time = time.time()
logging.basicConfig(filename='trading_bot_optimized.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

# --- Hiperparámetros Optimizados ---
class Hyperparameters:
    WINDOW_SIZE = 20                  # Aumentado de 15 para capturar más contexto temporal
    HIDDEN_SIZE = 128                 # Aumentado de 64 para mayor capacidad de modelado
    NUM_LAYERS = 2                    # Mantenido pero con mayor hidden_size
    BATCH_SIZE = 128                  # Aumentado de 64 para mejor estabilidad
    GAMMA = 0.95                      # Reducido de 0.99 para menos enfoque en recompensas lejanas
    EPS_START = 1.0
    EPS_END = 0.01                    # Reducido de 0.05 para menos exploración al final
    EPS_DECAY = 0.99999995            # Ajustado de 0.998 para decaimiento más lento
    TAU = 0.01                        # Aumentado de 0.005 para actualizaciones más rápidas del target network
    LR = 0.001                        # Aumentado de 0.0005 para aprendizaje más rápido
    TRAIN_EPISODES = 250              # Aumentado de 200
    PATIENCE = 20                     # Aumentado de 15 para permitir más iteraciones sin mejora

""" --- Cargar y preprocesar datos --- """
def load_and_preprocess_data(filepath):
    """Carga y preprocesa los datos históricos con más indicadores técnicos"""
    df = pd.read_csv(filepath, index_col='close_time', parse_dates=True)
    
    # Limpieza de datos
    df = df.replace([np.inf, -np.inf], np.nan).ffill()
    
    # Añadir más indicadores técnicos
    df['MA_10'] = df['close_price'].rolling(window=10).mean()
    df['MA_50'] = df['close_price'].rolling(window=50).mean()
    df['MA_200'] = df['close_price'].rolling(window=200).mean()
    df['hourly_return'] = df['close_price'].pct_change()
    df['RSI'] = compute_rsi(df['close_price'], 14)
    df['MACD'] = compute_macd(df['close_price'])
    df['Bollinger_Upper'], df['Bollinger_Lower'] = compute_bollinger_bands(df['close_price'])
    
    # Seleccionar características
    selected_features = ['open_price', 'high_price', 'low_price', 'close_price', 
                        'close_volume', 'MA_10', 'MA_50', 'MA_200', 'hourly_return',
                        'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
    df = df[selected_features].dropna()
    
    # Normalizar datos
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return df_normalized.values, df, scaler

# Funciones para indicadores técnicos
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



# --- Entorno de Trading Mejorado ---
class EnhancedTradingEnvironment:
    def __init__(self, data, window_size=Hyperparameters.WINDOW_SIZE):
        self.data = data
        self.window_size = window_size
        self.current_step = window_size
        self.max_steps = len(data) - 1
        self.action_space = 3  # 0=vender, 1=mantener, 2=comprar
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
        current_price = self.data[self.current_step, 3]
        next_price = self.data[self.current_step + 1, 3] if self.current_step < self.max_steps else current_price
        
        # Manejo seguro del cálculo de price_change, divisiones por 0
        try:
            price_change = (next_price - current_price) / current_price if current_price != 0 else 0
        except Exception as e:
            logging.warning(f"Error calculando price_change: {e}")
            price_change = 0
        
        # Validar acción según posición actual
        valid_action = action
        if action == 2 and self.position == 1:
            valid_action = 1
        elif action == 0 and self.position != 1:
            valid_action = 1
        
        # Sistema de recompensas mejorado
        reward = 0
        if valid_action == 0:  # Vender
            reward = price_change * -2.5  # Castiga vender antes de subidas
            self.position = 0
            # Lógica de venta real
            sell_amount = self.positions * 0.1  # Vender 10% de la posición
            self.current_balance += (sell_amount * current_price) * (1 - self.commission)
            self.positions -= sell_amount
        elif valid_action == 2:  # Comprar
            reward = price_change * 2.0  # Premia compras acertadas
            self.position = 1
            # Lógica de compra real
            buy_amount = self.current_balance * 0.1  # Comprar con 10% del balance
            self.positions += (buy_amount * (1 - self.commission)) / current_price
            self.current_balance -= buy_amount
        else:  # Mantener
            reward = 0.5 if abs(price_change) < 0.005 else -0.2  # Premia mantener en mercados laterales
        
        # Aplicar comisión
        if valid_action != 1:
            reward -= self.commission * 2
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Calcular valor actual del portafolio
        current_value = self.current_balance + self.positions * current_price
        
        next_state = self._get_state()
        
        return next_state, reward, done, {
            "price": current_price,
            "valid_action": valid_action,
            "portfolio_value": current_value
        }

# --- Red Neuronal LSTM ---
class EnhancedLSTMDQN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, action_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Capa LSTM mejorada con dropout recurrente
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Capas fully connected con inicialización mejorada
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
        # Dropout y activaciones
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()
        
        # Inicialización de pesos
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
        # Reorganizar los datos para LSTM
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
        
        # Tomar solo el último paso de tiempo
        out = out[:, -1, :]
        
        # Capas fully connected
        out = self.activation(self.bn1(self.fc1(out)))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, hidden

# --- Agente DQN Mejorado con LSTM ---
class LSTMDQNAgent:
    def __init__(self, input_size, hidden_size, num_layers, action_size):
        self.memory = deque(maxlen=50000)  # Buffer más grande
        self.gamma = Hyperparameters.GAMMA
        self.epsilon = Hyperparameters.EPS_START
        self.epsilon_min = Hyperparameters.EPS_END
        self.epsilon_decay = Hyperparameters.EPS_DECAY
        self.input_size = input_size
        self.action_size = action_size
        
        # Redes LSTM
        self.model = EnhancedLSTMDQN(input_size, hidden_size, num_layers, action_size).to(device)
        self.target_model = EnhancedLSTMDQN(input_size, hidden_size, num_layers, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Optimizador mejorado
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                   lr=Hyperparameters.LR, 
                                   weight_decay=1e-4)
        
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
        
        # Muestreo de experiencias
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(device)
        
        # Double DQN con LSTM
        current_q, _ = self.model(states)
        current_q = current_q.gather(1, actions.unsqueeze(1))
        
        next_actions, _ = self.model(next_states)
        next_actions = next_actions.max(1)[1]
        
        next_q, _ = self.target_model(next_states)
        next_q = next_q.gather(1, next_actions.unsqueeze(1))
        
        target = rewards + (1 - dones) * self.gamma * next_q.squeeze()
        
        # Cálculo de pérdida
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target.detach())
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Soft update del target network
        self.soft_update_target_network()
        
        # Decaimiento de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.steps_done += 1
    
    def soft_update_target_network(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def save_model(self, save_path, e, best_score, train_rewards):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),  # Importante para Double DQN
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': e,  # Episodio actual
            'best_score': best_score,
            'train_rewards': train_rewards  # Historial de recompensas
        }, save_path)
        print(f"💾 Modelo guardado en {save_path} (Episodio {e}, ε={self.epsilon:.4f})")

    def load_model(self, saved_path,device):
        try:
            # 1. Cargar el checkpoint con manejo de seguridad
            checkpoint = torch.load(saved_path, 
                                map_location=device,
                                weights_only=False)  # Necesario para tu versión de PyTorch
            
            # 2. Cargar pesos del modelo
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            
            # 3. Mover modelos al dispositivo correcto
            self.model.to(device)
            self.target_model.to(device)
            
            # 4. Cargar estado del optimizador
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Mover tensores del optimizador al dispositivo
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            
            # 5. Restaurar parámetros de entrenamiento
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            best_score = checkpoint.get('best_score', -np.inf)
            train_rewards = checkpoint.get('train_rewards', [])
            
            print(f"✅ Modelo cargado correctamente en {device}")
            print(f"| ε: {self.epsilon:.4f} | Mejor Score: {best_score:.2f}% |")
            
            return self, best_score, train_rewards
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {str(e)}")
            print("Asegúrate que:")
            print("1. La arquitectura del modelo no ha cambiado")
            print("2. El archivo no está corrupto")
            print("3. Las versiones de PyTorch son compatibles")
            raise

# --- Función de Evaluación Mejorada ---
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
        
        # Desnormalizar precio
        price_range = df['close_price'].max() - df['close_price'].min()
        current_price = info['price'] * price_range + df['close_price'].min()
        
        # Registrar información
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
    # Cargar datos
    data_array, df, scaler = load_and_preprocess_data('C:\\Andrey\\Kakua_Projets\\Trading\\Bot_RL_v1\\Datasets\\historical_01-01-2019_to_01-01-2025_ETHUSDT.csv')

    """ --- División de datos (80% train, 20% test) --- """

    train_size = int(0.8 * len(data_array))
    train_data = data_array[:train_size]
    test_data = data_array[train_size:]

    print(f"\nDivisión de datos:")
    print(f"Total: {len(data_array)}")
    print(f"Entrenamiento: {len(train_data)}")
    print(f"Evaluación: {len(test_data)}")

    # --- Creación de entornos ---
    train_env = EnhancedTradingEnvironment(train_data)
    test_env = EnhancedTradingEnvironment(test_data)

    # Tamaños para la LSTM
    input_size = train_data.shape[1]
    action_size = train_env.action_space

    # --- Configuración del entrenamiento ---
    agent = LSTMDQNAgent(input_size, Hyperparameters.HIDDEN_SIZE, 
                        Hyperparameters.NUM_LAYERS, action_size)

    episodes = Hyperparameters.TRAIN_EPISODES
    save_path = 'best_trading_model.pth'
    best_score = -np.inf
    no_improve = 0
    patience = Hyperparameters.PATIENCE

    # --- Entrenamiento ---
    print("\nComenzando entrenamiento con hiperparámetros optimizados...")
    train_rewards = []
    val_returns = []
    best_model_info = {
        'episode': 0,
        'val_return': -np.inf,
        'train_reward': -np.inf
    }

    # Cada episodio es una pasada completa por los datos de entrenamiento
    for e in range(episodes):
        state = train_env.reset() # Reinicia al inicio de los datos
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
            val_return, _, _, actions = evaluate(agent, test_env, df, return_full_history=True)
            val_returns.append(val_return)
            elapsed = (time.time() - start_time) / 3600

            # Calcular distribución de acciones
            actions_dist = pd.Series(actions).value_counts(normalize=True)

            print(f"Episodio: {e+1}/{episodes}, "
                f"Recompensa Entrenamiento: {total_reward:.2f}, "
                f"Retorno Validación: {val_return:.2f}%, "
                f"ε: {agent.epsilon:.4f}, "
                f"Pasos: {agent.steps_done}, "
                f"Tiempo: {elapsed:.2f}h")
            print(f"Acciones: Comprar={actions_dist.get(2, 0):.1%}, "
                  f"Vender={actions_dist.get(0, 0):.1%}, "
                  f"Mantener={actions_dist.get(1, 0):.1%}")
            # Guardar el mejor modelo
            if val_return > best_score:
                best_score = val_return
                no_improve = 0
                agent.save_model(save_path)
                
                best_model_info = {
                    'episode': e+1,
                    'val_return': val_return,
                    'train_reward': total_reward
                }
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\nEarly stopping en episodio {e+1} - Sin mejora por {patience} evaluaciones")
                    break
    
    """------------------------- Evaluación final y visualización -------------------------"""
    # Cargar el mejor modelo antes de la evaluación final
    agent, best_score, train_rewards = agent.load_model(save_path, device)
    print(f"\nMejor modelo cargado (Episodio {best_model_info['episode']}):")
    print(f"Recompensa Entrenamiento: {best_model_info['train_reward']:.2f}")
    print(f"Retorno Validación: {best_model_info['val_return']:.2f}%")

    # --- Evaluación Final ---
    print("\nEvaluando con datos de test...")
    final_return, portfolio_history, price_history, actions_history, rewards_history = evaluate(
        agent, test_env, df, return_full_history=True
    )

    # --- Cálculo de métricas ---
    final_value = portfolio_history[-1]
    returns = np.diff(portfolio_history) / portfolio_history[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    max_drawdown = (np.maximum.accumulate(portfolio_history) - portfolio_history).max()
    max_drawdown_pct = (max_drawdown / np.maximum.accumulate(portfolio_history)).max()
    buy_hold_return = (price_history[-1] / price_history[0] - 1) * 100
    volatility = np.std(returns) * np.sqrt(252) * 100
    win_rate = len([r for r in returns if r > 0]) / len(returns) * 100

    print("\n--- Resultados Finales Optimizados ---")
    print(f"Valor inicial: $10,000.00")
    print(f"Valor final: ${final_value:,.2f}")
    print(f"Retorno estrategia: {(final_value/10000-1)*100:.2f}%")
    print(f"Retorno Buy & Hold: {buy_hold_return:.2f}%")
    print(f"\n--- Métricas de Riesgo-Retorno ---")
    print(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
    print(f"Volatilidad Anualizada: {volatility:.2f}%")
    print(f"Max Drawdown: ${max_drawdown:,.2f} ({max_drawdown_pct:.2%})")
    print(f"Tasa de Aciertos: {win_rate:.2f}%")

    # Distribución detallada de acciones
    actions_dist = pd.Series(actions_history).value_counts(normalize=True).sort_index()
    print("\nDistribución Detallada de Acciones:")
    for action, percent in zip(['Vender', 'Mantener', 'Comprar'], actions_dist):
        print(f"{action}: {percent:.1%}")

    # Análisis de operaciones
    positive_trades = len([r for r in returns if r > 0])
    negative_trades = len([r for r in returns if r < 0])
    avg_win = np.mean([r for r in returns if r > 0]) * 100
    avg_loss = np.mean([r for r in returns if r < 0]) * 100
    profit_factor = -avg_win * positive_trades / (avg_loss * negative_trades) if negative_trades > 0 else np.inf

    print("\n--- Análisis de Operaciones ---")
    print(f"Operaciones positivas: {positive_trades} ({positive_trades/len(returns):.1%})")
    print(f"Operaciones negativas: {negative_trades} ({negative_trades/len(returns):.1%})")
    print(f"Ganancia promedio: {avg_win:.2f}%")
    print(f"Pérdida promedio: {avg_loss:.2f}%")
    print(f"Factor de beneficio: {profit_factor:.2f}") 

    # --- Visualización ---
    plt.figure(figsize=(18, 12))
    plt.suptitle('Resultados del Trading Bot con LSTM Optimizado', fontsize=16)

    # Gráfico 1: Precio vs Valor Portafolio
    plt.subplot(2, 2, 1)
    plt.plot(price_history, label='Precio ETH', color='blue', alpha=0.6)
    plt.ylabel('Precio (USD)')
    plt.legend(loc='upper left')
    plt.grid(True)

    ax2 = plt.gca().twinx()
    ax2.plot(portfolio_history[1:], label='Valor Portafolio', color='green')
    ax2.axhline(y=10000, color='red', linestyle='--', label='Inversión Inicial')
    ax2.set_ylabel('Valor (USD)')
    ax2.legend(loc='upper right')
    plt.title('Precio vs Valor del Portafolio')

    # Gráfico 2: Distribución de Acciones
    plt.subplot(2, 2, 2)
    action_names = ['Vender', 'Mantener', 'Comprar']
    action_counts = pd.Series(actions_history).value_counts()
    plt.bar(action_names, action_counts, color=['red', 'gray', 'green'])
    plt.xlabel('Acción')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Acciones')
    plt.grid(True)

    # Gráfico 3: Recompensas durante Evaluación
    plt.subplot(2, 2, 3)
    cumulative_rewards = np.cumsum(rewards_history)
    plt.plot(cumulative_rewards, label='Recompensas Acumuladas', color='purple')
    plt.xlabel('Paso de Tiempo')
    plt.ylabel('Recompensa')
    plt.title('Recompensas durante Evaluación')
    plt.legend()
    plt.grid(True)

    # Gráfico 4: Progreso del Entrenamiento
    plt.subplot(2, 2, 4)
    plt.plot(train_rewards, label='Recompensa Entrenamiento', color='orange')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title('Progreso del Entrenamiento')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    