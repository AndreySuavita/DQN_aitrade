from binance.client import Client
import pandas as pd
from binance.enums import *
import math
from decouple import config

API_KEY = config('API_KEY')
API_SECRET = config('API_SECRET')

class binance_actions:
    def __init__(self, API_KEY=API_KEY, API_SECRET=API_SECRET, testnet=True):
        self.client = Client(API_KEY, API_SECRET, testnet = testnet)
    # def get_last_klines_hours(self, symbol= 'ETHUSDT', limit=1000):
    #     """
    #     function to get the last klines of a symbol in hours.
    #     param symbol: The symbol to get the klines for. Default is 'ETHUSDT'.
    #     param limit: The number of klines to get. Default is 1000.  
    #     """
    #     interval = Client.KLINE_INTERVAL_1HOUR  
    #     klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit+1) # trae uno más para evitar traer el actual

    #     columns = [
    #         "open_time", "open_price", "high_price", "low_price", "close_price",
    #         "close_volume", "close_time", "quote_asset_volume", "number_of_trades",
    #         "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    #     ]
    #     df = pd.DataFrame(klines[:-1], columns=columns) # evita traer el último kline, el cual aun no ha cerrado

    #     # Convertir columnas numéricas
    #     numeric_cols = [
    #         "open_price", "high_price", "low_price", "close_price",
    #         "close_volume", "quote_asset_volume",
    #         "taker_buy_base_volume", "taker_buy_quote_volume"
    #     ]
    #     df[numeric_cols] = df[numeric_cols].astype(float)

    #     # Convertir timestamps (borrar en producción)
    #     # df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    #     # df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')

    #     return df
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

        selected_interval = interval_map[interval]

        klines = self.client.get_klines(symbol=symbol, interval=selected_interval, limit=limit+1) # Fetch one more to avoid bringing the current kline which has not yet closed

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

    def get_balance(self, symbol='ETH'):
        """
        returns the balance of the given symbol in the account and 'USDT'.
        param symbol: The symbol to get the balance for. Default is 'ETH'.
        """
        account_info = self.client.get_account()
        response = {}
        for asset in account_info['balances']:
            if asset['asset'] in [symbol, 'USDT']:
                response[asset['asset']] = {'free': float(asset['free']),'locked': float(asset['locked'])}
        
        return response
    def trade_eth_usdt(self,amount_usdt, side='buy'):
        """
        Ejecuta una orden de compra o venta de ETH contra USDT usando testnet.
        
        :param amount_usdt: Monto en USDT a invertir (para compra o venta)
        :param side: 'buy' para comprar ETH, 'sell' para vender ETH
        """
        symbol = 'ETHUSDT'

        # Obtener precio actual
        price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])

        # Obtener filtros de mercado
        info = self.client.get_symbol_info(symbol)
        lot_filter = next(f for f in info['filters'] if f['filterType'] == 'LOT_SIZE')
        step_size = float(lot_filter['stepSize'])
        min_qty = float(lot_filter['minQty'])
        precision = len(str(step_size).split('.')[1])

        notional_filter = next((f for f in info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
        min_notional = float(notional_filter['minNotional']) if notional_filter else 10.0

        # Calcular cantidad de ETH
        eth_quantity = round(amount_usdt / price, precision)
        eth_quantity = math.floor(eth_quantity / step_size) * step_size
        eth_quantity = round(eth_quantity, precision)

        # Verificar balances si es venta
        if side == 'sell':
            balances = self.client.get_account()['balances']
            eth_balance = float(next(b['free'] for b in balances if b['asset'] == 'ETH'))
            if eth_quantity > eth_balance:
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
        try:
            if side == 'buy':
                order = self.client.order_market_buy(symbol=symbol, quantity=eth_quantity)
                print(f"✅ ORDEN DE COMPRA REALIZADA: {eth_quantity} ETH por aproximadamente {amount_usdt} USDT")      
            elif side == 'sell':
                order = self.client.order_market_sell(symbol=symbol, quantity=eth_quantity)
                print(f"✅ ORDEN DE VENTA REALIZADA: {eth_quantity} ETH por aproximadamente {amount_usdt} USDT")
            else:
                print("❌ Acción no reconocida. Usa 'buy' o 'sell'.")
                return
            
            return {'ID_order':order['orderId'], 'symbol':order['symbol'], 'status':order['status'], 'ETH_quantity':order['executedQty'], 'USDT_quantity':amount_usdt}

        except Exception as e:
            print("❌ Error al ejecutar la orden:", e)

        
if __name__ == "__main__":

    binance = binance_actions()
    
    # print(binance.get_last_klines_hours("ETHUSDT",15))
    print(binance.get_klines(symbol='ETHUSDT', interval='5m', limit=15))
    # print(binance.get_balance("ETH"))
    # print(binance.trade_eth_usdt(amount_usdt=11, side='buy')) #side='buy' or side='sell'
    # print(binance.trade_eth_usdt(amount_usdt=11, side='sell')) #side='buy' or side='sell'


    