import gym
import time

import numpy as np
import pandas as pd

from config import config
from datetime import datetime

# RL models from stable-baselines
from stable_baselines import A2C, ACKTR, PPO2
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from config import config

# customized env
if config.OPERATING_MODE == 'returns':
    from environments.returns.train import train_env
    from environments.returns.trade import trade_env
    from environments.returns.validation import validation_env
else:
    from environments.sharpe.train import train_env
    from environments.sharpe.trade import trade_env
    from environments.sharpe.validation import validation_env


def data_split(df, start, end):
    data = df[(df["Open time"]>=start) & (df["Open time"]<end)].reset_index(drop=True)
    return data


def train_A2C(env_train, model_name, timesteps=25000):
    start = time.perf_counter()
    
    model = A2C('MlpLstmPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    
    end = time.perf_counter()
    print(f"A2C Training Time: {end - start:0.4f} seconds")
    return model


def train_ACKTR(env_train, model_name, timesteps=25000):
    start = time.perf_counter()
    
    model = ACKTR('MlpLstmPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    
    end = time.perf_counter()
    print(f"ACKTR Training Time: {end - start:0.4f} seconds")
    return model


def train_PPO(env_train, model_name, timesteps=25000):
    start = time.perf_counter()
    model = PPO2('MlpLstmPolicy', env_train, ent_coef=0.005, nminibatches=1)
    model.learn(total_timesteps=timesteps)
    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    
    end = time.perf_counter()
    print(f"PPO Training Time: {end - start:0.4f} seconds")
    return model


def DRL_prediction(df, model, ticker, model_name, last_state, iter_num, unique_trade_date, rebalance_window, initial):
    trade_data = data_split(df, start=unique_trade_date[iter_num-rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: trade_env(trade_data, ticker, initial=initial, previous_state=last_state, model_name=model_name, iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(model_name, i), index=False)
    return last_state


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration, model_name):
    df_total_value = pd.read_csv('results/account_value_validation_{}_{}.csv'.format(model_name, iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
    return sharpe


def run_RLTrade(df, ticker, model, unique_trade_date, rebalance_window, validation_window) -> None:
    
    last_state_A2C = []
    last_state_ACKTR = []
    last_state_PPO = []
    
    if model == 'A2C':
        for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
            ## initial state is empty
            if i - rebalance_window - validation_window == 0:
                initial = True
            else:
                initial = False

            ## training env
            train = data_split(df, start=datetime.strptime("01-01-2017", "%d-%m-%Y"), end=unique_trade_date[i-rebalance_window-validation_window])
            env_train = DummyVecEnv([lambda: train_env(train, ticker, model_name="A2C")])

            ## validation env
            validation = data_split(df, start=unique_trade_date[i-rebalance_window-validation_window], end=unique_trade_date[i-rebalance_window])
            env_val = DummyVecEnv([lambda: validation_env(validation, ticker, model_name="A2C", iteration=i)])
            obs_val = env_val.reset()

            ## model training
            print("======A2C Training========")
            model_a2c = train_A2C(env_train, model_name="A2C_{}_{}".format(ticker, i))
            print("======A2C Validation from: ", unique_trade_date[i-rebalance_window-validation_window], "to ", unique_trade_date[i-rebalance_window])
            DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
            sharpe_a2c = get_validation_sharpe(i, model_name="A2C")
            print("A2C Sharpe Ratio: ", sharpe_a2c)

            ## model evaluation
            print("======A2C Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
            last_state_A2C = DRL_prediction(df=df, model=model_a2c, ticker=ticker, model_name="A2C", last_state=last_state_A2C, iter_num=i, unique_trade_date=unique_trade_date, rebalance_window=rebalance_window, initial=initial)
            
    elif model == 'ACKTR':
        for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
            ## initial state is empty
            if i - rebalance_window - validation_window == 0:
                initial = True
            else:
                initial = False

            ## training env
            train = data_split(df, start=datetime.strptime("01-01-2017", "%d-%m-%Y"), end=unique_trade_date[i-rebalance_window-validation_window])
            env_train = DummyVecEnv([lambda: train_env(train, ticker, model_name="ACKTR")])

            ## validation env
            validation = data_split(df, start=unique_trade_date[i-rebalance_window-validation_window], end=unique_trade_date[i-rebalance_window])
            env_val = DummyVecEnv([lambda: validation_env(validation, ticker, model_name="ACKTR", iteration=i)])
            obs_val = env_val.reset()
        
            ## model training
            print("======ACKTR Training========")
            model_acktr = train_ACKTR(env_train, model_name="ACKTR_{}_{}".format(ticker, i))
            print("======ACKTR Validation from: ", unique_trade_date[i-rebalance_window-validation_window], "to ", unique_trade_date[i-rebalance_window])
            DRL_validation(model=model_acktr, test_data=validation, test_env=env_val, test_obs=obs_val)
            sharpe_acktr = get_validation_sharpe(i, model_name="ACKTR")
            print("ACKTR Sharpe Ratio: ", sharpe_acktr)
            
            ## model evaluation
            print("======ACKTR Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
            last_state_ACKTR = DRL_prediction(df=df, model=model_acktr, ticker=ticker, model_name="ACKTR", last_state=last_state_ACKTR, iter_num=i, unique_trade_date=unique_trade_date, rebalance_window=rebalance_window, initial=initial)
            
    elif model == 'PPO':
        for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
            ## initial state is empty
            if i - rebalance_window - validation_window == 0:
                initial = True
            else:
                initial = False
                
            ## training env
            train = data_split(df, start=datetime.strptime("01-01-2017", "%d-%m-%Y"), end=unique_trade_date[i-rebalance_window-validation_window])
            env_train = DummyVecEnv([lambda: train_env(train, ticker, model_name="PPO")])

            ## validation env
            validation = data_split(df, start=unique_trade_date[i-rebalance_window-validation_window], end=unique_trade_date[i-rebalance_window])
            env_val = DummyVecEnv([lambda: validation_env(validation, ticker, model_name="PPO", iteration=i)])
            obs_val = env_val.reset()

            ## model training
            print("======PPO Training========")
            model_ppo = train_PPO(env_train, model_name="PPO_{}_{}".format(ticker, i))
            print("======PPO Validation from: ", unique_trade_date[i-rebalance_window-validation_window], "to ", unique_trade_date[i-rebalance_window])
            DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
            sharpe_ppo = get_validation_sharpe(i, model_name="PPO")
            print("PPO Sharpe Ratio: ", sharpe_ppo)
            
            ## model evaluation
            print("======PPO Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
            last_state_PPO = DRL_prediction(df=df, model=model_ppo, ticker=ticker, model_name="PPO", last_state=last_state_PPO, iter_num=i, unique_trade_date=unique_trade_date, rebalance_window=rebalance_window, initial=initial)
            
    else:
        print("Model name not recognized. Please try again.")