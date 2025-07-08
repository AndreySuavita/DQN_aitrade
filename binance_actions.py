from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pandas as pd
from binance.enums import *
import math
from decouple import config
from datetime import datetime, timedelta, timezone
import time
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('binance_trace.log'),  # Guarda en archivo
        logging.StreamHandler()          # Muestra en consola
    ]
)
logger = logging.getLogger(__name__)

API_KEY = config('API_KEY')
API_SECRET = config('API_SECRET')

class binance_actions:
    def __init__(self, API_KEY=API_KEY, API_SECRET=API_SECRET, testnet=True):
        
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.testnet = testnet
        self.client = None
        self._connect_with_retry()

        #self.client = Client(API_KEY, API_SECRET, testnet = testnet)

    def _connect_with_retry(self, max_retries: int = 5, base_delay: float = 1.0):
        """
        Try to connect with binance with exponential lapses of time.
        
        Params:
            max_retries (int): Number of attemps
            base_delay (float): Base time to wait in seconds
        """
        for attempt in range(max_retries):
            try:
                self.client = Client(self.api_key, self.api_secret, testnet=self.testnet)
                # Test connection
                self.client.ping()
                logger.info("Connection successful with API")
                return
            except (BinanceAPIException, BinanceRequestException, Exception) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Conection fail after {max_retries} attemps: {str(e)}")
                    raise
                
                wait_time = base_delay * (2 ** attempt)  # Backoff exponential
                logger.warning(f"Attemp {attempt + 1} fail. Trying again in {wait_time} seconds...")
                time.sleep(wait_time)

    def _execute_with_retry(self, func, *args, max_retries: int = 3, **kwargs):
        """
        Execute a function from the Api Binance with automatic attemps.
        
        Params:
            func: Function to execute
            max_retries (int): Number of attemps
            *args, **kwargs: Arg for the function
        """
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (BinanceAPIException, BinanceRequestException) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Operation fail after {max_retries} attemps: {str(e)}")
                    raise
                
                if e.status_code == 429:  # Rate limit
                    wait_time = 60  # wait 1 minute
                else:
                    wait_time = 5 * (attempt + 1)  # progressive wait
                
                logger.warning(f"Intento {attempt + 1} fallido. Reintentando en {wait_time} segundos...")
                time.sleep(wait_time)
                
                # Reconnect if necessary
                if "Invalid API-key" in str(e) or "API signature" in str(e):
                    self._connect_with_retry()

    def get_klines(self, symbol='ETHUSDT', interval='1h', limit=1000):
        """
        Function to get the last klines of a symbol for a given interval.
        
        :param symbol: The symbol to get the klines for. Default is 'ETHUSDT'.
        :param interval: The interval of klines. Examples: '1m', '5m', '1h', '1d'.
        :param limit: The number of klines to get. Default is 1000.
        :return: DataFrame with kline data.
        """

        # Map string intervals to Binance API intervals
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '3m': Client.KLINE_INTERVAL_3MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '2h': Client.KLINE_INTERVAL_2HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '6h': Client.KLINE_INTERVAL_6HOUR,
            '8h': Client.KLINE_INTERVAL_8HOUR,
            '12h': Client.KLINE_INTERVAL_12HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
            '3d': Client.KLINE_INTERVAL_3DAY,
            '1w': Client.KLINE_INTERVAL_1WEEK,
            '1M': Client.KLINE_INTERVAL_1MONTH
        }

        if interval not in interval_map:
            raise ValueError(f"Unsupported interval '{interval}'. Must be one of: {list(interval_map.keys())}")

            
        try:
            klines = self._execute_with_retry(
                self.client.get_klines,
                symbol=symbol,
                interval=interval_map[interval],
                limit=limit + 1  # +1 para evitar traer la vela actual no cerrada
            )

            # selected_interval = interval_map[interval]
            # klines = self.client.get_klines(symbol=symbol, interval=selected_interval, limit=limit+1) # Fetch one more to avoid bringing the current kline which has not yet closed
            
            columns = [
                "open_time", "open_price", "high_price", "low_price", "close_price",
                "close_volume", "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
            ]
            df = pd.DataFrame(klines[:-1], columns=columns) # avoids bringing the last kline, which has not yet closed

            # Convert numeric columns to float
            numeric_cols = [
                "open_price", "high_price", "low_price", "close_price",
                "close_volume", "quote_asset_volume",
                "taker_buy_base_volume", "taker_buy_quote_volume"
            ]
            df[numeric_cols] = df[numeric_cols].astype(float)

            # Optionally convert timestamps to datetime:
            # df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
            # df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')

            return df

        except Exception as e:
            logger.error(f"Error getting klines: {str(e)}")
            raise

    def get_balance(self, symbol='ETH'):
        """
        Obtiene el balance de la cuenta con reintentos automáticos.
        
        Parámetros:
            symbol (str): Símbolo del activo a consultar
            
        Retorna:
            dict: Diccionario con los balances {'asset': {'free': float, 'locked': float}}
        """
        try:
            account_info = self._execute_with_retry(self.client.get_account)
            response = {}
            for asset in account_info['balances']:
                if asset['asset'] in [symbol, 'USDT']:
                    response[asset['asset']] = {
                        'free': float(asset['free']),
                        'locked': float(asset['locked'])
                    }
            return response
        except Exception as e:
            logger.error(f"Error al obtener balance: {str(e)}")
            raise

    def trade_eth_usdt(self,amount_usdt, side):
        """
        Ejecuta una orden de compra o venta de ETH contra USDT usando testnet.
        
        :param amount_usdt: Monto en USDT a invertir (para compra o venta)
        :param side: 'buy' para comprar ETH, 'sell' para vender ETH
        """
        symbol = 'ETHUSDT'

        try:
            # Obtener precio con reintentos
            ticker = self._execute_with_retry(self.client.get_symbol_ticker, symbol=symbol)
            price = float(ticker['price'])
            
            # Obtener información del símbolo
            info = self._execute_with_retry(self.client.get_symbol_info, symbol=symbol)
            
            # Procesamiento de filtros (igual que antes)
            lot_filter = next(f for f in info['filters'] if f['filterType'] == 'LOT_SIZE')
            step_size = float(lot_filter['stepSize'])
            min_qty = float(lot_filter['minQty'])
            precision = len(str(step_size).split('.')[1])

            notional_filter = next((f for f in info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
            min_notional = float(notional_filter['minNotional']) if notional_filter else 10.0

            # Cálculos de cantidad (igual que antes)
            eth_quantity = round(amount_usdt / price, precision)
            eth_quantity = math.floor(eth_quantity / step_size) * step_size
            eth_quantity = round(eth_quantity, precision)

            # Verificar balances si es venta
            if side == 'sell':
                balances = self._execute_with_retry(self.client.get_account)['balances']
                eth_balance = float(next(b['free'] for b in balances if b['asset'] == 'ETH'))
                if eth_quantity > eth_balance:
                    logger.error(f"No hay suficiente ETH. Disponible: {eth_balance}, Necesario: {eth_quantity}")
                    print(f"❌ No tienes suficiente ETH. Disponible: {eth_balance}, requerido: {eth_quantity}")
                    return

            # Verificaciones generales
            if eth_quantity < min_qty:
                print(f"❌ Cantidad mínima no alcanzada: {eth_quantity} ETH (mínimo permitido: {min_qty})")
                return

            notional = eth_quantity * price
            if notional < min_notional:
                print(f"❌ Valor total ({notional:.2f} USDT) es menor al mínimo requerido ({min_notional} USDT)")
                return

            # Ejecutar la orden
            if side == 'buy':
                '''
                Qué se cobra: La comisión se deduce del activo de cotización (en ETH/USDT, se cobra en USDT).
                Ejemplo:
                Compras 0.1 ETH a $1000 (total: $100).
                Comisión (0.1%): $0.10 en USDT.
                Recibes: 0.1 ETH (sin deducción).
                Balance final:
                USDT: -100 - 0.10 = -100.10.
                ETH: +0.1.
                '''
                #order = self.client.order_market_buy(symbol=symbol, quantity=eth_quantity)
                order = self._execute_with_retry(
                    self.client.order_market_buy,
                    symbol=symbol,
                    quantity=eth_quantity
                )
            elif side == 'sell':
                '''
                Qué se cobra: La comisión se deduce del activo base (en ETH/USDT, se cobra en ETH).
                Ejemplo:
                Vendes 0.1 ETH a $1000 (total: $100).
                Comisión (0.1%): 0.0001 ETH (equivalente a $0.10).
                Recibes: $100 en USDT.
                Balance final:
                ETH: -0.1 - 0.0001 = -0.1001.
                USDT: +100.
                '''
                #order = self.client.order_market_sell(symbol=symbol, quantity=eth_quantity)
                order = self._execute_with_retry(
                    self.client.order_market_sell,
                    symbol=symbol,
                    quantity=eth_quantity
                )
            else:
                logger.error("Acción no reconocida. Usar 'buy' o 'sell'")
                print("❌ Acción no reconocida. Usa 'buy' o 'sell'.")
                return
            commission = order['fills'][0]['commission'] # Real Commission in USDT or BNB
            commission_asset = order['fills'][0]['commissionAsset'] # Activo de la commission (USDT o BNB) , USDT if buy, ETH if sell
            logger.info(f'Orden ejecutada: {side.upper()} {eth_quantity} ETH por {amount_usdt} USDT')
            print(f'✅ ORDER OF {side.upper()} EXECUTED: {eth_quantity} ETH FOR {amount_usdt} USDT, COMMISSION: {commission} USDT , COMMISSION ASSET: {commission_asset}')
            return {'ID_order':order['orderId'], 'symbol':order['symbol'], 'status':order['status'], 'ETH_quantity':order['executedQty'], 'USDT_quantity':amount_usdt, 'commission':commission, 'commission_asset':commission_asset} 

        except Exception as e:
            logger.error(f"Error en trade_eth_usdt: {str(e)}")
            print("❌ Error al ejecutar la orden:", e)

def wait_until_next_time_cycle(time_cycle, offset_seconds=10):
    """Wait until the next time cycle with an offset"""
    now = datetime.now(timezone.utc)  # Modern and recommended way
    if time_cycle not in ['5m', 'hourly']:
        raise ValueError("time_cycle must be 'hourly' or '5m'")
    if time_cycle == 'hourly':
        # Calculate the next hour start
        next_time = (now.replace(minute=0, second=0, microsecond=0) 
                    + timedelta(hours=1)) 
    elif time_cycle == '5m':
        next_time = (now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0) 
                    + timedelta(minutes=5))
    # Add the offset
    target_time = next_time + timedelta(seconds=offset_seconds)
    # Calculate how long to wait
    wait_seconds = (target_time - now).total_seconds()
    if wait_seconds > 0:
        time.sleep(wait_seconds)
        
if __name__ == "__main__":
    binance = binance_actions()
    
    print(binance.get_klines(symbol='ETHUSDT', interval='1h', limit=100))
    # print(binance.get_balance("ETH"))
    # print(binance.trade_eth_usdt(amount_usdt=11, side='buy')) #side='buy' or side='sell'
    # print(binance.trade_eth_usdt(amount_usdt=11, side='sell')) #side='buy' or side='sell'

