# Trading Bot with Reinforcement Learning

This project implements a trading bot using Reinforcement Learning (RL) techniques, specifically focusing on Double Deep Q-Network (DDQN) and LSTM-based approaches for cryptocurrency trading on the Binance platform.

## Project Overview

The project combines traditional trading strategies with advanced machine learning techniques to create an automated trading system. It includes implementations of:
- Double DQN (DDQN) for hourly trading
- LSTM-based DDQN for enhanced time series analysis
- Integration with Binance API for real-time trading

## Features

- Real-time cryptocurrency data fetching from Binance
- Automated trading execution with market orders
- Balance tracking and management
- Multiple RL model implementations
- Support for both testnet and mainnet trading
- Advanced risk management and position sizing
- Real-time performance monitoring and analytics
- Customizable trading strategies and parameters

## Prerequisites

- Python 3.12
- Binance account (for API access)
- CUDA-compatible GPU (recommended for training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AndreySuavita/DQN_aitrade.git
cd Bot_RL_v1
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/`: Source code directory
  - `models/`: RL model implementations
  - `utils/`: Utility functions and helpers
  - `config/`: Configuration files
  - `data/`: Data processing and management
- `tests/`: Test files and test data
- `notebooks/`: Jupyter notebooks for analysis and development
- `docs/`: Documentation and guides
- `data/`: Directory for storing training and testing data

## Usage

1. Set up your Binance API credentials in the configuration file
2. Configure your trading parameters in the respective model files
3. Run the training process:
```bash
python src/models/train.py
```

## Features

- **Market Data Integration**: Real-time data fetching from Binance
- **Automated Trading**: Support for market orders with proper quantity calculations
- **Balance Management**: Real-time balance tracking and position management
- **Multiple Model Architectures**: Different RL approaches for various trading strategies
- **Risk Management**: Advanced position sizing and risk control
- **Performance Analytics**: Real-time monitoring and performance metrics
- **Customizable Strategies**: Flexible configuration for different trading approaches

## Security Notes

- Never commit your API keys to the repository
- Always use testnet for development and testing
- Implement proper risk management in your trading strategies
- Use environment variables for sensitive information
- Regular security audits and updates

## Contributing

Feel free to submit issues and enhancement requests! Please follow the contribution guidelines when submitting pull requests.

## Disclaimer

This project is for educational purposes only. Cryptocurrency trading involves significant risk. Use at your own discretion. The authors are not responsible for any financial losses incurred through the use of this software. 