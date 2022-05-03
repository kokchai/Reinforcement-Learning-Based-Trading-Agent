import gym

import numpy as np
import pandas as pd

from config import config

INITIAL_USDT_BALANCE = config.INITIAL_USDT_BALANCE
INITIAL_CRYPTO_BALANCE = config.INITIAL_CRYPTO_BALANCE

TRANSACTION_FEE_PERCENT = config.TRANSACTION_FEE_PERCENT

class validation_env(gym.Env):
    
    def __init__(self, df, ticker, ts=0, model_name='', iteration=''):
        self.df = df # dataframe
        self.ts = ts # timestep
        
        # initialize ticker, adjust trade size depending on asset being traded
        self.ticker = ticker
        if self.ticker == 'BTC':
            self.trade_size = 0.1
        elif self.ticker == 'ETH':
            self.trade_size = 1
        elif self.ticker == 'BNB':
            self.trade_size = 10
        elif self.ticker == 'SOL':
            self.trade_size = 100
        elif self.ticker == 'ADA':
            self.trade_size = 10000

        # initialize action and observation space
        self.action_space = gym.spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32) 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,))
        
        # initialize data
        self.data = self.df.loc[self.ts, :]
        self.terminal = False   
        
        # initalize state
        self.state = [INITIAL_USDT_BALANCE,
                      INITIAL_CRYPTO_BALANCE,
                      self.data['Close'],
                      self.data['Open'],
                      self.data['High'],
                      self.data['Low'],
                      self.data['MA(7)'],     # indicator 1: 7-day moving average
                      self.data['MA(25)'],    # indicator 2: 25-day moving average
                      self.data['MA(99)'],    # indicator 3: 99-day moving average
                      self.data['Change'],    # indicator 4: 
                      self.data['Amplitude'], # indicator 5: 
                      self.data['RSI'],       # indicator 6: relative strength index
                      self.data['CCI']]       # indicator 7: commodity channel index
        
        # initialize reward, cost, and number of trades
        self.reward = 0
        self.cost = 0
        self.trades = 0
        
        # initialize memory of assets and rewards
        self.asset_memory = [INITIAL_USDT_BALANCE]
        self.holding_memory = []
        self.return_memory = []
        self.reward_memory = []
        self._seed()
        
        self.iteration = iteration
        self.model_name = model_name
    
    
    def _sell(self, action):
        """
        Method to execute a selling action, which includes an update on the cash balance and account holding.
        Checking is present to prevent the agent from selling more assets than possible.
        """
        crypto_trade_amount = min(abs(action[0]), self.state[1])
        
        if self.state[1] > 0:
            #update balance
            self.state[0] += self.state[2] * crypto_trade_amount * (1-TRANSACTION_FEE_PERCENT)
            self.state[1] -= crypto_trade_amount
            
            self.cost += self.state[2] * crypto_trade_amount * TRANSACTION_FEE_PERCENT
            self.trades+=1
        else:
            pass
    
    
    def _buy(self, action):
        """
        Method to execute a buying action, which includes an update on the cash balance and account holding.
        Checking is present to prevent the agent from buying more assets than possible.
        """
        available_amount = self.state[0] / self.state[2]
        crypto_trade_amount = min(available_amount, action[0])
        
        if crypto_trade_amount > 0:
            #update balance
            self.state[0] -= self.state[2] * crypto_trade_amount * (1+TRANSACTION_FEE_PERCENT)
            self.state[1] += crypto_trade_amount
            
            self.cost += self.state[2] * crypto_trade_amount * TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            pass
    
    
    def step(self, action):
        """
        Method to advance the environment by a single timestep, execute the trading action and update the state.
        If the simulation has reached its terminal state, the finalized array containing history of account value and account holdings are written to a .csv file.
        """
        self.terminal = self.ts >= len(self.df.index.unique())-1

        if self.terminal:
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_validation_{}_{}.csv'.format(self.model_name, self.iteration))
            end_total_asset = self.state[0] + (self.state[2] * self.state[1])

            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252**0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            
            df_total_holding = pd.DataFrame(self.holding_memory)
            df_total_holding.to_csv('results/account_holding_validation_{}_{}.csv'.format(self.model_name, self.iteration))
            return self.state, self.reward, self.terminal,{}

        else:
            begin_total_asset = self.state[0] + (self.state[2] * self.state[1])

            action = action * self.trade_size
            if action < 0:
                self._sell(action)
            else:
                self._buy(action)

            self.ts += 1
            self.data = self.df.loc[self.ts,:]
            
            self.state = [self.state[0],
                          self.state[1],
                          self.data['Close'],
                          self.data['Open'],
                          self.data['High'],
                          self.data['Low'],
                          self.data['MA(7)'],     # indicator 1: 7-day moving average
                          self.data['MA(25)'],    # indicator 2: 25-day moving average
                          self.data['MA(99)'],    # indicator 3: 99-day moving average
                          self.data['Change'],    # indicator 4: 
                          self.data['Amplitude'], # indicator 5: 
                          self.data['RSI'],       # indicator 6: relative strength index
                          self.data['CCI']]       # indicator 7: commodity channel index
            
            end_total_asset = self.state[0] + (self.state[2] * self.state[1])
            self.asset_memory.append(end_total_asset)
            self.holding_memory.append(self.state[1])
            
            profit_rate = (begin_total_asset-end_total_asset) / begin_total_asset
            if np.abs(profit_rate) > 0:
                self.return_memory.append(profit_rate)
            else:
                self.return_memory.append(-0.02)
            
            self.reward = (252**0.5) * np.mean(self.return_memory[-10:]) / np.std(self.return_memory[-10:])
            self.reward_memory.append(self.reward)
        return self.state, self.reward, self.terminal, {}
    
    
    def reset(self):
        """
        Method to reset the environment to its initial state.
        """
        self.ts = 0

        self.data = self.df.loc[self.ts,:]
        self.terminal = False 

        self.asset_memory = [INITIAL_USDT_BALANCE]
        self.return_memory = [0]
        self.reward_memory = []

        # re-initiate state
        self.state = [INITIAL_USDT_BALANCE,
                      INITIAL_CRYPTO_BALANCE,
                      self.data['Close'],
                      self.data['Open'],
                      self.data['High'],
                      self.data['Low'],
                      self.data['MA(7)'],     # indicator 1: 7-day moving average
                      self.data['MA(25)'],    # indicator 2: 25-day moving average
                      self.data['MA(99)'],    # indicator 3: 99-day moving average
                      self.data['Change'],    # indicator 4: 
                      self.data['Amplitude'], # indicator 5: 
                      self.data['RSI'],       # indicator 6: relative strength index
                      self.data['CCI']]       # indicator 7: commodity channel index
        
        self.cost = 0
        self.trades = 0
        return self.state
    
    
    def render(self, mode='human', close=False):
        """
        Method to render the environment.
        """
        return self.state
    

    def _seed(self, seed=None):
        """
        Method to instantiate a random seed.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

