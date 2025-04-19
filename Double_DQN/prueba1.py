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

# Configuración de logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO)

# --- Configuración de GPU/CPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

# --- Cargar y preprocesar datos ---
def load_and_preprocess_data(filepath):
    """Carga y preprocesa los datos históricos"""
    df = pd.read_csv(filepath, index_col='close_time')
    
    # Limpieza de datos
    df = df.replace([np.inf, -np.inf], np.nan).ffill()  
    # Añadir indicadores técnicos básicos
    df['MA_10'] = df['close_price'].rolling(window=10).mean()
    df['MA_50'] = df['close_price'].rolling(window=50).mean()
    df['daily_return'] = df['close_price'].pct_change()
    
    # Seleccionar características
    selected_features = ['open_price', 'high_price', 'low_price', 'close_price', 
                        'close_volume', 'MA_10', 'MA_50', 'daily_return']
    df = df[selected_features].dropna()
    
    # Normalizar datos
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return df_normalized.values, df, scaler

# Cargar datos
data_array, df, scaler = load_and_preprocess_data('C:\\Andrey\\Kakua_Projets\\Trading\\Bot_RL_v1\\Datasets\\historical_01-01-2019_to_01-01-2025_ETHUSDT.csv')


# --- División de datos (80% train, 20% test) ---
train_size = int(0.8 * len(data_array))
train_data = data_array[:train_size]
test_data = data_array[train_size:]

print(f"\nDivisión de datos:")
print(f"Total: {len(data_array)}")
print(f"Entrenamiento: {len(train_data)}")
print(f"Evaluación: {len(test_data)}")

# --- Entorno de Trading Mejorado ---
class EnhancedTradingEnvironment:
    def __init__(self, data, window_size=15):
        self.data = data   # Datos normalizados [pasos x features]
        self.window_size = window_size # Historial visible (15 pasos)
        self.current_step = window_size # Comienza después de tener suficiente historial
        self.max_steps = len(data) - 1 # Último paso posible
        self.action_space = 3  # Acciones posibles: 0=vender, 1=mantener, 2=comprar
        self.state_size = window_size * data.shape[1] # Tamaño del estado aplanado
        self.position = 0  # 0=no invertido, 1=invertido (en ETH)
        self.commission = 0.001  # Comisión del 0.1% por operación
        # print('--state_size--')
        # print(self.state_size)
        # exit()
    def reset(self):
        self.current_step = self.window_size
        self.position = 0
        return self._get_state()
    
    def _get_state(self):
        return self.data[self.current_step - self.window_size : self.current_step].flatten() # ventana de datos aplanada
    
    def step(self, action):
        # Cálculo de Precios
        current_price = self.data[self.current_step, 3]  # Precio de cierre actual
        next_price = self.data[self.current_step + 1, 3] if self.current_step < self.max_steps else current_price # Cambio porcentual
        # print('--data--')
        # print(type(self.data))
        # print('--current_price--')
        # print(current_price)
        # print('--next_price--')
        # print(next_price)
        # exit()
        # Manejo seguro del cálculo de price_change, divisiones por 0
        try:
            price_change = (next_price - current_price) / current_price if current_price != 0 else 0
        except Exception as e:
            logging.warning(f"Error calculando price_change: {e}")
            price_change = 0
        
        # Validar acción según posición actual
        valid_action = action
        if action == 2 and self.position == 1:  # Quiere comprar pero ya está invertido
            valid_action = 1  # Forzar mantener
        elif action == 0 and self.position != 1: # Quiere vender sin tener posición
            valid_action = 1  # Forzar mantener
        
        # Sistema de recompensas mejorado
        if valid_action == 0:  # Vender
            reward = -price_change * 2.5 # Castiga vender antes de subidas
            self.position = 0
        elif valid_action == 2:  # Comprar
            reward = price_change * 2.0  # Premia compras acertadas
            self.position = 1
        else:  # Mantener
            reward = 0.2 if abs(price_change) < 0.01 else -0.1 # Premia mantener en mercados laterales
        
        # Aplicar comisión
        if valid_action != 1: # Si no es mantener
            reward -= self.commission * 2 # Penaliza comisión (ida y vuelta)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_state = self._get_state()
        
        return next_state, reward, done, {"price": current_price, "valid_action": valid_action}


# --- Red Neuronal Mejorada ---
class EnhancedDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128) # capa de entrada, 120 entradas, 128 salidas
        self.bn1 = nn.BatchNorm1d(128) # normalización por lotes
        self.fc2 = nn.Linear(128, 64) # capa oculta, 128 entradas, 64 salidas
        self.bn2 = nn.BatchNorm1d(64) # normalización por lotes
        self.fc3 = nn.Linear(64, action_size) # capa de salida, 64 entradas, 3 salidas (acciones)
        self.dropout = nn.Dropout(0.25) # Apaga el 25% de las neuronas para evitar sobreajuste
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0) # Añade dimensión batch si es necesario
        x = torch.relu(self.bn1(self.fc1(x))) # Primera transformación
        x = self.dropout(x) # Apaga ~32 neuronas (25% de 128) para evitar overfitting
        x = torch.relu(self.bn2(self.fc2(x))) # Capa oculta
        return self.fc3(x) # [1,64] → [1,3] (Q-values para vender/mantener/comprar)


class EnhancedDQNAgent:
    """
        Double DQN: Separa la selección y evaluación de acciones

        Experience Replay: Memoria de 20,000 transiciones

        Target Network: Red separada para cálculos estables

        Soft Updates: Actualización progresiva de la red objetivo
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size # 120
        self.action_size = action_size # 3
        self.memory = deque(maxlen=20000)  # Buffer de experiencias
        self.gamma = 0.99  # Factor de descuento de recompensas futuras
        self.epsilon = 1.0  # Probabilidad inicial de exploración (100%)
        self.epsilon_min = 0.05  # Mínima exploración permitida (5%)
        self.epsilon_decay = 0.998  # Tasa de decaimiento de epsilon
        self.model = EnhancedDQN(state_size, action_size).to(device)  # Red principal
        self.target_model = EnhancedDQN(state_size, action_size).to(device)  # Red objetivo
        self.target_model.load_state_dict(self.model.state_dict())  # Inicialización idéntica
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=1e-5)
        self.batch_size = 64  # Tamaño del mini-batch
        self.tau = 0.005  # Para soft update del target network
        self.update_every = 4  # Frecuencia de actualización
    
    def remember(self, state, action, reward, next_state, done):
        """
            Función: Almacena experiencias (state, action, reward, next_state, done)
            Capacidad: 20,000 muestras (elimina las más antiguas al superar este límite)
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state): # Toma de decisiones
        """
            ε-greedy: Balance entre exploración (acciones aleatorias) y explotación (usar el modelo)
            Procesamiento:
                - Convierte el estado a tensor
                - Añade dimensión de batch (unsqueeze)
                - Mueve a GPU si está disponible
        """
        if np.random.rand() <= self.epsilon:  # Exploración
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.model.eval()  # Modo evaluación
        with torch.no_grad():
            q_values = self.model(state)
        self.model.train()  # Vuelve a modo entrenamiento
        return torch.argmax(q_values).item()  # Explotación (mejor acción)
    
    def replay(self): # Entrenamiento (replay) - Cuando memoria ≥ batch_size
        if len(self.memory) < self.batch_size:
            return
        
        # Muestreo de Experiencias
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(device)
        
        # Double DQN
        next_actions = self.model(next_states).max(1)[1]  # Selección con red principal
        next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))  # Evaluación con red objetivo
        target = rewards + (1 - dones) * self.gamma * next_q.squeeze()  # Cálculo del target
        
        # Actualización de pesos # Backpropagation
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        loss = nn.MSELoss()(current_q.squeeze(), target.detach())
        self.optimizer.zero_grad()
        loss.backward()  # Calcula gradientes
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Evita exploding gradients
        self.optimizer.step()  # Actualiza pesos
        
        # Soft update del target network
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- Función de Evaluación Mejorada ---
def evaluate(agent, env, initial_balance=10000, return_full_history=False):
    '''
    La función evaluate tiene dos objetivos clave:
    Evaluar el rendimiento del agente con una estrategia de trading conservadora
    Simular operaciones reales con gestión de capital (solo invierte el 10% del portafolio en cada operación)

    Parametros de entrada:
    - agent: El agente DQN que hemos entrenado (contiene el modelo de red neuronal)
    - env: El entorno de trading (puede ser de entrenamiento o de test)
    - initial_balance: Capital inicial para la simulación (por defecto $10,000)
    '''
    state = env.reset()
    portfolio = initial_balance
    positions = 0
    portfolio_history = [portfolio]
    price_history = []
    actions_history = []
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        # Desnormalizar precio
        price_range = df['close_price'].max() - df['close_price'].min()
        current_price = info['price'] * price_range + df['close_price'].min()
        
        # Lógica de trading conservadora
        if action == 2 and portfolio > 0:
            buy_amount = portfolio * 0.1
            positions += (buy_amount * (1 - env.commission)) / current_price
            portfolio -= buy_amount
        elif action == 0 and positions > 0:
            sell_amount = positions * 0.1
            portfolio += (sell_amount * current_price) * (1 - env.commission)
            positions -= sell_amount
        
        current_value = portfolio + positions * current_price
        portfolio_history.append(current_value)
        price_history.append(current_price)
        actions_history.append(action)
        state = next_state
    
    final_return = (portfolio_history[-1] / initial_balance - 1) * 100
    
    if return_full_history:
        return final_return, portfolio_history, price_history, actions_history
    else:
        return final_return, portfolio_history

# --- Creación de entornos ---
window_size = 15
train_env = EnhancedTradingEnvironment(train_data, window_size)
test_env = EnhancedTradingEnvironment(test_data, window_size)
state_size = train_env.state_size
action_size = train_env.action_space

# print('--state_size--')
# print(state_size)
# print('--action_size--')
# print(action_size)
# exit()

# --- Configuración del entrenamiento ---
agent = EnhancedDQNAgent(state_size, action_size)
episodes = 200
save_path = 'best_trading_model.pth'
best_score = -np.inf
no_improve = 0
patience = 15

# --- Entrenamiento ---
print("\nComenzando entrenamiento...")
train_rewards = []

for e in range(episodes):
    state = train_env.reset() # Reinicia al inicio de los datos
    total_reward = 0
    done = False
    
    while not done: # Hasta llegar al final de los datos de entrenamiento
        action = agent.act(state)  # Decide comprar/vender/mantener  (ε-greedy)
        next_state, reward, done, _ = train_env.step(action) # Aplica acción
        agent.remember(state, action, reward, next_state, done) # Almacena experiencias para aprender después
        state = next_state # Avanza al siguiente estado
        total_reward += reward
        
        if len(agent.memory) > agent.batch_size:
            agent.replay() #backpropagation, Entrena con mini-batches
    
    train_rewards.append(total_reward)
    
    # Validación periódica
    if e % 5 == 0:
        val_return, _ = evaluate(agent, test_env)
        print(f"Episodio: {e+1}/{episodes}, Recompensa: {total_reward:.2f}, Retorno Val: {val_return:.2f}%, ε: {agent.epsilon:.3f}")
        
        # Early stopping
        if val_return > best_score:
            best_score = val_return
            no_improve = 0
            torch.save(agent.model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping en episodio {e}")
                break

# --- Evaluación Final usando la función evaluate mejorada ---
print("\nEvaluando con datos de test...")
final_return, portfolio_history, price_history, actions_history = evaluate(
    agent, test_env, return_full_history=True
)

# --- Visualización ---
plt.figure(figsize=(16, 10))

# Gráfico 1: Precio
plt.subplot(2, 2, 1)
plt.plot(price_history, label='Precio ETH', color='blue', alpha=0.6)
plt.xlabel('Paso de Tiempo')
plt.ylabel('Precio (USD)')
plt.title('Precio durante Evaluación')
plt.grid(True)

# Gráfico 2: Portafolio
plt.subplot(2, 2, 2)
plt.plot(portfolio_history, label='Valor Portafolio', color='green')
plt.axhline(y=10000, color='red', linestyle='--', label='Inversión Inicial')
plt.xlabel('Paso de Tiempo')
plt.ylabel('Valor (USD)')
plt.title('Rendimiento del Portafolio')
plt.legend()
plt.grid(True)

# Gráfico 3: Acciones
plt.subplot(2, 2, 3)
plt.plot(actions_history, 'o', markersize=2, alpha=0.6)
plt.yticks([0, 1, 2], ['Vender', 'Mantener', 'Comprar'])
plt.xlabel('Paso de Tiempo')
plt.ylabel('Acción')
plt.title('Distribución de Acciones')
plt.grid(True)

# Gráfico 4: Recompensas
plt.subplot(2, 2, 4)
plt.plot(train_rewards, label='Recompensa', color='purple')
plt.xlabel('Episodio')
plt.ylabel('Recompensa Acumulada')
plt.title('Progreso del Entrenamiento')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Métricas Finales ---
final_value = portfolio_history[-1]
returns = np.diff(portfolio_history) / portfolio_history[:-1]
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
max_drawdown = (np.maximum.accumulate(portfolio_history) - portfolio_history).max()
buy_hold_return = (price_history[-1] / price_history[0] - 1) * 100

print("\n--- Resultados Finales ---")
print(f"Valor inicial: $10,000.00")
print(f"Valor final: ${final_value:,.2f}")
print(f"Retorno estrategia: {(final_value/10000-1)*100:.2f}%")
print(f"Retorno Buy & Hold: {buy_hold_return:.2f}%")
print(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
print(f"Max Drawdown: ${max_drawdown:,.2f} ({max_drawdown/10000:.2%})")

# Distribución de acciones
actions_dist = pd.Series(actions_history).value_counts(normalize=True)
print("\nDistribución de acciones:")
print(f"Comprar: {actions_dist.get(2, 0):.1%}")
print(f"Mantener: {actions_dist.get(1, 0):.1%}")
print(f"Vender: {actions_dist.get(0, 0):.1%}")