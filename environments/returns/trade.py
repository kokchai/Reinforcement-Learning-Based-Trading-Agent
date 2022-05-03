import gym

import numpy as np
import pandas as pd

from config import config

INITIAL_USDT_BALANCE = config.INITIAL_USDT_BALANCE
INITIAL_CRYPTO_BALANCE = config.INITIAL_CRYPTO_BALANCE

TRANSACTION_FEE_PERCENT = config.TRANSACTION_FEE_PERCENT

class trade_env(gym.Env):
    
    def __init__(self, df, ticker, ts=0, initial=True, previous_state=[], model_name='', iteration=''):
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
        
        #initialize data
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
                      self.data['Change'],    # indicator 4: Inter-day change
                      self.data['Amplitude'], # indicator 5: Intra-day amplitude
                      self.data['RSI'],       # indicator 6: relative strength index
                      self.data['CCI']]       # indicator 7: commodity channel index
        
        # initialize reward
        self.reward = 0
        self.cost = 0
        self.trades = 0
        
        # initialize memory of assets and rewards
        self.asset_memory = [INITIAL_USDT_BALANCE]
        self.holding_memory = []
        self.reward_memory = []
        self._seed()
        
        self.model_name = model_name
        self.iteration = iteration
                
        self.initial = initial
        self.previous_state = previous_state
    
    
    def _sell(self, action):
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
        available_amount = self.state[0] // self.state[2]
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
        self.terminal = self.ts >= len(self.df.index.unique())-1

        if self.terminal:
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            end_total_asset = self.state[0] + (self.state[2] * self.state[1])

            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252**0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            
            df_total_holding = pd.DataFrame(self.holding_memory)
            df_total_holding.to_csv('results/account_holding_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            return self.state, self.reward, self.terminal,{}

        else:
            action = action * self.trade_size
            begin_total_asset = self.state[0] + (self.state[2] * self.state[1])

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
                          self.data['Change'],    # indicator 4: Intra-day change
                          self.data['Amplitude'], # indicator 5: Intra-day amplitude
                          self.data['RSI'],       # indicator 6: relative strength index
                          self.data['CCI']]       # indicator 7: commodity channel index
            
            end_total_asset = self.state[0] + (self.state[2] * self.state[1])
            self.holding_memory.append(self.state[1])
            self.asset_memory.append(end_total_asset)
            
            self.reward = end_total_asset - begin_total_asset            
            self.reward_memory.append(self.reward)
        return self.state, self.reward, self.terminal, {}
    
    
    def reset(self):
        if self.initial:
            self.ts = 0

            self.data = self.df.loc[self.ts,:]
            self.terminal = False

            self.asset_memory = [INITIAL_USDT_BALANCE]
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
                          self.data['Change'],    # indicator 4: Intra-day change
                          self.data['Amplitude'], # indicator 5: Intra-day amplitude
                          self.data['RSI'],       # indicator 6: relative strength index
                          self.data['CCI']]       # indicator 7: commodity channel index

            self.cost = 0
            self.trades = 0
            
        else:
            self.ts = 0
            
            self.data = self.df.loc[self.ts,:]
            self.terminal = False
            
            previous_total_asset = self.previous_state[0] + (self.previous_state[2] * self.previous_state[1])
            self.asset_memory = [previous_total_asset]
            self.rewards_memory = []
            
            self.state = [self.previous_state[0],
                          self.previous_state[1],
                          self.data['Close'],
                          self.data['Open'],
                          self.data['High'],
                          self.data['Low'],
                          self.data['MA(7)'],     # indicator 1: 7-day moving average
                          self.data['MA(25)'],    # indicator 2: 25-day moving average
                          self.data['MA(99)'],    # indicator 3: 99-day moving average
                          self.data['Change'],    # indicator 4: Intra-day change
                          self.data['Amplitude'], # indicator 5: Intra-day amplitude
                          self.data['RSI'],       # indicator 6: relative strength index
                          self.data['CCI']]       # indicator 7: commodity channel index
            self.cost = 0
            self.trades = 0
        return self.state
    
    
    def render(self, mode='human', close=False):
        return self.state
    

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
