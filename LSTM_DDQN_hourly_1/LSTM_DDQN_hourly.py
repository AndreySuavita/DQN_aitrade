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
import pandas_ta as ta
import time


# Configuración inicial
start_time = time.time()
logging.basicConfig(filename='trading_bot.log', level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

# --- Cargar y preprocesar datos mejorado ---
def load_and_preprocess_data(filepath):
    """Carga y preprocesa los datos históricos con normalización mejorada"""
    df = pd.read_csv(filepath, index_col='close_time')
    df.index = pd.to_datetime(df.index)
    df = df.replace([np.inf, -np.inf], np.nan).ffill()
    
    # --- Indicadores técnicos mejorados ---
    df['OBV'] = ta.obv(df['close_price'], df['close_volume'])
    df['VWAP'] = ta.vwap(df['high_price'], df['low_price'], df['close_price'], df['close_volume'])
    df['MA_24h'] = df['close_price'].rolling(window=24).mean()
    df['MA_168h'] = df['close_price'].rolling(window=168).mean()
    df['hourly_return'] = df['close_price'].pct_change()
    df['RSI_14h'] = ta.rsi(df['close_price'], length=14)
    df['EMA_12'] = df['close_price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close_price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['ATR_24h'] = ta.atr(df['high_price'], df['low_price'], df['close_price'], length=24)
    df['Momentum_24h'] = df['close_price'].pct_change(24)
    
    # Normalización especial para indicadores acotados
    df['RSI_14h'] = df['RSI_14h'] / 100  # Normalizar RSI entre 0-1
    df['MACD'] = np.tanh(df['MACD'].values * 0.1)  # Versión más segura
    
    # Normalización mejorada
    df['OBV'] = np.tanh(df['OBV'].values * 1e-7)
    df['VWAP'] = (df['VWAP'] - df['close_price'].mean()) / df['close_price'].std()

    # Normalización estándar para otras features
    scaler = MinMaxScaler()
    features_to_scale = ['open_price', 'high_price', 'low_price', 'close_price', 
                         'close_volume', 'MA_24h', 'MA_168h', 'hourly_return',
                         'ATR_24h', 'Momentum_24h']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    return df.values, df, scaler

# --- Entorno de Trading Mejorado (AJUSTES PARA HORARIO) ---
class EnhancedTradingEnvironment:
    def __init__(self, data, window_size=24):  # Ventana de 24 horas (1 día)
        self.data = data   # Datos normalizados (numpy array)
        self.window_size = window_size # Historial de 24 horas (1 día)
        self.current_step = window_size # Paso actual (empieza después de tener ventana completa)
        self.max_steps = len(data) - 1 # Máximo de pasos posibles
        self.action_space = 3  # 0=vender, 1=mantener, 2=comprar
        self.state_size = window_size * data.shape[1] # Tamaño del estado (ventana * features) (192=8*24)
        self.position = 0  # 0=no invertido, 1=invertido
        self.commission = 0.001  # comisión por operación, cambiar a 0.001
        self.max_position_size = 0.1  # 10% del portafolio máximo por operación

    def reset(self): # Se llama al inicio de cada episodio de entrenamiento para comenzar desde un estado limpio.
        self.current_step = self.window_size  # Reinicia al inicio de los datos (después de la ventana)
        self.position = 0                     # Cierra cualquier posición abierta
        return self._get_state()    
    
    def _get_state(self):
        """
        Toma los datos de las últimas window_size horas (ej: 24 filas).
        flatten(): Convierte la matriz 2D (24h x 8 features) en un vector 1D (para la red neuronal).
        """
        return self.data[self.current_step - self.window_size : self.current_step].flatten()
    
    def step(self, action):
        current_price = self.data[self.current_step, 3]  # Precio de cierre actual (columna 3)
        next_price = self.data[self.current_step + 1, 3] if self.current_step < self.max_steps else current_price # Precio de la siguiente hora
        price_change = (next_price - current_price) / current_price if current_price != 0 else 0
        
        # Validar acción (mantenido igual) -- Evita acciones imposibles (ej: vender sin tener activos).
        valid_action = action
        if action == 2 and self.position == 1:   # Quiere comprar pero ya está invertido
            valid_action = 1                     # Forzar a mantener
        elif action == 0 and self.position != 1: # Quiere vender sin tener posición
            valid_action = 1                    # Forzar a mantener
        
        # Sistema de recompensas mejorado
        
        
        # Nuevo sistema de recompensas con momentum
        momentum = self.data[self.current_step, 10]  # Índice de Momentum
        obv = self.data[self.current_step, 5]       # OBV normalizado
        rsi = self.data[self.current_step, 7]  # RSI normalizado (0-1)

        if valid_action == 2:  # Comprar
            reward = price_change * (6.0 + 2.0 * momentum)  # Refuerzo positivo con momentum
            if rsi > 0.7:
                reward *= 0.2  # Penalización más fuerte
            elif obv < -0.5:   # Si el flujo de dinero es negativo
                reward *= 0.3
            self.position = 1
            
        elif valid_action == 0:  # Vender
            reward = -price_change * (2.0 - 1.0 * momentum)  # Menor recompensa
            if rsi < 0.3:
                reward *= 0.3
            elif obv > 0.5:     # Si el flujo de dinero es positivo
                reward *= 0.5
            self.position = 0
            
        else:  # Mantener
            reward = 0.1 * (1 + obv)  # Recompensa basada en flujo de dinero


        # Penalización por sobre-operar
        if valid_action != 1 and abs(price_change) < 0.005:  # Movimientos pequeños
            reward -= 0.1

        # Limitar recompensas
        reward = np.clip(reward, -2.0, 2.0)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps # ¿Llegamos al final de los datos?
        next_state = self._get_state()
        
        """
        next_state: Nuevo estado (ventana deslizada 1 hora).

        reward: Recompensa/penalización.

        done: True si el episodio terminó.

        info: Metadata útil (precio actual, acción válida).
        """
        return next_state, reward, done, {"price": current_price, "valid_action": valid_action}

# --- Red Neuronal  ---
class EnhancedDQN(nn.Module):
    def __init__(self, state_size, action_size, window_size=24):
        self.window_size = window_size
        self.state_size = state_size
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_size//self.window_size, 
                           hidden_size=64, 
                           num_layers=2,
                           batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(64, 512),
            nn.SiLU(),  # Nueva función de activación
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # Normalización por capas
            nn.SiLU(),
            nn.Linear(256, action_size)
        )
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(-1, self.window_size, self.state_size//self.window_size)  # Reformar para LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Tomar sólo la última salida
        return self.net(x)

class EnhancedDQNAgent:
    """
        Double DQN: Separa la selección y evaluación de acciones

        Experience Replay: Memoria de 50,000 transiciones

        Target Network: Red separada para cálculos estables

        Soft Updates: Actualización progresiva de la red objetivo
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size # 192
        self.action_size = action_size # 3
        self.memory = deque(maxlen=50000)  # Buffer de experiencias
        self.gamma = 0.98  # Factor de descuento de recompensas futuras
        self.epsilon = 1.0  # Probabilidad inicial de exploración (100%)
        self.epsilon_min = 0.05  # Mínima exploración permitida (10%)
        self.epsilon_decay = 0.9999995 # Tasa de decaimiento de epsilon muy lenta para 120 episodios llega a 0.1
        self.model = EnhancedDQN(state_size, action_size).to(device) #red objetivo
        self.target_model = EnhancedDQN(state_size, action_size).to(device) #red principal
        self.target_model.load_state_dict(self.model.state_dict())  # Inicialización idéntica
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=1e-5, amsgrad=True)
        self.batch_size = 512  # Tamaño del mini-batch #128
        self.tau = 0.005  # Para soft update del target network
        self.update_every = 5  # Frecuencia de actualización 
    
    def remember(self, state, action, reward, next_state, done):
        """
            Función: Almacena experiencias (state, action, reward, next_state, done)
            Capacidad: 50,000 muestras (elimina las más antiguas al superar este límite)
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
        if np.random.rand() <= self.epsilon:  # Exploración (acción aleatoria)
            return random.randrange(self.action_size)
        
        # Explotación (usar el modelo)
        # Asegurar que el estado tiene la forma correcta
        state = torch.FloatTensor(state).to(device)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Añadir dimensión batch si es necesario

        self.model.eval()  # Modo evaluación
        with torch.no_grad(): #anula los gradientes para la evaluación
            q_values = self.model(state) # Shape: [1, 3] (Q-values para cada acción)
        self.model.train()  # Vuelve a modo entrenamiento
        return torch.argmax(q_values).item()  # Explotación (mejor acción), # Devuelve la acción con mayor Q-value
    
    def replay(self): # Entrenamiento (replay) - Cuando memoria ≥ batch_size
        """
            Muestrea 64 experiencias aleatorias del buffer.

            Calcula los Q-targets (usando Double DQN).

            Realiza backpropagation y actualiza los pesos de la red.

            Suaviza la actualización de la red objetivo (target_model).
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Muestreo de Experiencias
        minibatch = random.sample(self.memory, self.batch_size) # selecciona 64 experiencias aleatorias de experiencias pasadas
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(device) 
        #print("Dimensiones de states:", states.shape)  # Debería ser [batch_size, state_size]
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(device)
        
        # Double DQN # Usa model para elegir la acción, pero target_model para evaluar su Q-value.
        next_actions = self.model(next_states).max(1)[1]  # Selección con red principal
        next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))  # Evaluación con red objetivo
        target = rewards + (1 - dones) * self.gamma * next_q.squeeze()  # Cálculo del target ,Fórmula de Bellman ajustada
        
        # Actualización de pesos # Backpropagation
        current_q = self.model(states).gather(1, actions.unsqueeze(1)) # Predicciones actuales
        loss = nn.MSELoss()(current_q.squeeze(), target.detach()) #  Cálculo de pérdida (MSE)

        # Optimización
        self.optimizer.zero_grad()
        loss.backward()  # Calcula gradientes
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Evita exploding gradients
        self.optimizer.step()  # Actualiza pesos
        
        # Soft update del target network
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # desminuir epsilon (exploración) para aumentar la explotación
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

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

# --- Función de Evaluación (AJUSTES PARA HORARIO) ---
def evaluate(agent, env, df, initial_balance=10000, return_full_history=False):
    """
    Propósito Principal
    Evaluar cómo se comportaría tu estrategia de trading en el mundo real, usando:

    Datos históricos (precios por hora).

    Política aprendida por el agente (compra/venta/mantener).

    Gestión de capital conservadora (inversión del 1% por operación).
    
    """
    state = env.reset()               # Reinicia el entorno (primer estado)
    portfolio = initial_balance       # Ej: $10,000 USD iniciales
    positions = 0                     # Cantidad de activos (ETH) en posesión
    portfolio_history = [portfolio]   # Registra valor del portafolio en cada paso
    price_history = []                # Guarda precios desnormalizados
    actions_history = []              # Registra acciones tomadas
    done = False

    # Desnormalizador de precios
    price_min = df['close_price'].min()
    price_range = df['close_price'].max() - price_min
    
    while not done:
        action = agent.act(state)
        next_state, _, done, info = env.step(action)
        
        # Desnormalizar precio
        current_price = info['price'] * price_range + price_min
        
        # Ejecutar operaciones
        if action == 2 and portfolio > 0:  # Comprar
            buy_amount = portfolio * 0.1  # 10% del portafolio
            positions += (buy_amount * (1 - env.commission)) / current_price
            portfolio -= buy_amount
        elif action == 0 and positions > 0:  # Vender
            sell_amount = positions * 0.1  # Vender 10% de la posición
            portfolio += (sell_amount * current_price) * (1 - env.commission)
            positions -= sell_amount
        
        # Registrar valores
        current_value = portfolio + positions * current_price
        portfolio_history.append(current_value)
        price_history.append(current_price)
        actions_history.append(action)
        state = next_state
    
    # Calcular retorno porcentual
    final_return = (portfolio_history[-1] / initial_balance - 1) * 100
    
    if return_full_history:
        return final_return, portfolio_history, price_history, actions_history
    return final_return, portfolio_history

""" ----------------------------------------------------- Implementación Principal ------------------------------------------------------ """
if __name__ == "__main__":
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

    # --- Creación de entornos ---
    window_size = 24  # Cambiado a 24 horas 
    train_env = EnhancedTradingEnvironment(train_data, window_size)
    test_env = EnhancedTradingEnvironment(test_data, window_size)
    state_size = window_size * train_data.shape[1]  # features * ventana temporal
    print(f"State size calculado: {state_size}")
    action_size = train_env.action_space

    # --- Configuración del entrenamiento ---
    agent = EnhancedDQNAgent(state_size, action_size)
    episodes = 150 
    save_path = 'best_trading_model.pth'
    best_score = -np.inf
    no_improve = 0
    patience = max(15, int(episodes * 0.01))

    # Verificación de dimensiones
    print(f"Número de features: {train_data.shape[1]}")
    print(f"Window size: {window_size}")
    print(f"State size calculado: {window_size * train_data.shape[1]}")

    # --- Entrenamiento por fases
    print("\nComenzando entrenamiento...")
    train_rewards = []

    # Cada episodio es una pasada completa por los datos de entrenamiento
    for e in range(episodes):
        state = train_env.reset() # Reinicia al inicio de los datos
        total_reward = 0
        done = False
        
        # Ajustar parámetros por fase
        if e < int(episodes*0.4):
            agent.epsilon = max(0.6, agent.epsilon)
            train_env.commission = 0.0003
        elif e < int(episodes*0.8):
            agent.epsilon = max(0.25, agent.epsilon)
            train_env.commission = 0.0005
        else:
            agent.epsilon = max(0.1, agent.epsilon)
            train_env.commission = 0.0008

        while not done: # Hasta llegar al final de los datos de entrenamiento
            action = agent.act(state) # Decide comprar/vender/mantener  (ε-greedy)
            next_state, reward, done, _ = train_env.step(action) # Aplica acción
            agent.remember(state, action, reward, next_state, done) # Almacena experiencias para aprender después
            state = next_state # Avanza al siguiente estado
            total_reward += reward
            
            if len(agent.memory) > agent.batch_size: # Si hay suficientes experiencias almacenadas (batch_size=64)
                agent.replay() # backpropagation, Entrena con mini-batches la red neuronal
            #print(f'epsilon: {agent.epsilon}')
        
        train_rewards.append(total_reward)
        
        # Evaluación y guardado
        if e % 10 == 0:
            val_return, _, _, actions = evaluate(agent, test_env, df, return_full_history=True)
            elapsed = (time.time() - start_time) / 3600
            
            # Calcular distribución de acciones
            actions_dist = pd.Series(actions).value_counts(normalize=True)
            
            print(f"Episodio: {e+1}/{episodes}, Recompensa: {total_reward:.2f}, "
                  f"Retorno Val: {val_return:.2f}%, ε: {agent.epsilon:.3f}, "
                  f"Tiempo: {elapsed:.2f}h")
            print(f"Acciones: Comprar={actions_dist.get(2, 0):.1%}, "
                  f"Vender={actions_dist.get(0, 0):.1%}, "
                  f"Mantener={actions_dist.get(1, 0):.1%}")
            
            # Early stopping
            if val_return > best_score:
                best_score = val_return
                no_improve = 0
                # Guardar TODO el estado del agente (no solo el modelo)
                agent.save_model(save_path,e,best_score,train_rewards) 
                # torch.save({
                #     'model_state_dict': agent.model.state_dict(),
                #     'target_model_state_dict': agent.target_model.state_dict(),  # Importante para Double DQN
                #     'optimizer_state_dict': agent.optimizer.state_dict(),
                #     'epsilon': agent.epsilon,
                #     'episode': e,  # Episodio actual
                #     'best_score': best_score,
                #     'train_rewards': train_rewards  # Historial de recompensas
                # }, save_path)
                # print(f"💾 Modelo guardado en {save_path} (Episodio {e}, ε={agent.epsilon:.4f})")
            else:
                no_improve += 1 
                if no_improve >= patience: # Si no mejora en "patience" evaluaciones
                    print(f"Early stopping en episodio {e}")
                    break
        
    """------------------------- Evaluación final y visualización -------------------------"""
    # cargar el modelo guardado
    agent, best_score, train_rewards = agent.load_model(save_path, device)

    # --- Evaluación Final ---
    print("\nEvaluando con datos de test...")
    final_return, portfolio_history, price_history, actions_history = evaluate(
        agent, test_env, df, return_full_history=True)
    
    # Cálculo de métricas
    final_value = portfolio_history[-1]
    returns = np.diff(portfolio_history) / portfolio_history[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365)
    max_drawdown = (np.maximum.accumulate(portfolio_history) - portfolio_history).max()
    buy_hold_return = (price_history[-1] / price_history[0] - 1) * 100
    actions_dist = pd.Series(actions_history).value_counts(normalize=True)
    print("\n--- Resultados Finales ---")
    print(f"Valor inicial: $10,000.00")
    print(f"Valor final: ${final_value:,.2f}")
    print(f"Retorno estrategia: {(final_value/10000-1)*100:.2f}%")
    print(f"Retorno Buy & Hold: {buy_hold_return:.2f}%")
    print(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: ${max_drawdown:,.2f} ({max_drawdown/10000:.2%})")
    print(f"Acciones: Comprar={actions_dist.get(2, 0):.1%}, "
          f"Vender={actions_dist.get(0, 0):.1%}, "
          f"Mantener={actions_dist.get(1, 0):.1%}")
    print(f"Tiempo total: {(time.time() - start_time)/3600:.2f} horas")

    # --- Visualización ---
    plt.figure(figsize=(16, 10))

    # Gráfico 1: Precio
    plt.subplot(2, 2, 1)
    plt.plot(price_history, label='Precio ETH', color='blue', alpha=0.6)
    plt.xlabel('Paso de Tiempo (horas)')
    plt.ylabel('Precio (USD)')
    plt.title('Precio durante Evaluación')
    plt.grid(True)

    # Gráfico 2: Portafolio
    plt.subplot(2, 2, 2)
    plt.plot(portfolio_history, label='Valor Portafolio', color='green')
    plt.axhline(y=10000, color='red', linestyle='--', label='Inversión Inicial')
    plt.xlabel('Paso de Tiempo (horas)')
    plt.ylabel('Valor (USD)')
    plt.title('Rendimiento del Portafolio')
    plt.legend()
    plt.grid(True)

    # Gráfico 3: Acciones
    plt.subplot(2, 2, 3)
    plt.plot(actions_history, 'o', markersize=2, alpha=0.6)
    plt.yticks([0, 1, 2], ['Vender', 'Mantener', 'Comprar'])
    plt.xlabel('Paso de Tiempo (horas)')
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

    # Marca el final
    end_time = time.time()
    # Calcula el tiempo transcurrido en horas
    elapsed_time = end_time - start_time
    print(f"Tiempo de ejecución: {elapsed_time/3600:.4f} Horas")