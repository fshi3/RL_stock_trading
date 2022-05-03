# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:16:26 2022

@author: Noah
"""
import gym
import gym_anytrading

from gym_anytrading.envs import StocksEnv, Positions, Actions
from gym_anytrading import datasets
from gym.envs.registration import register
from copy import deepcopy


class AdvancedEnv(StocksEnv):
    def _calculate_reward(self, action):
        
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
    
        percent_diff = (current_price-last_trade_price)/last_trade_price
        
        if self._position == Positions.Short:
            return -percent_diff
        else:
            return percent_diff
            
        
        
register(
    id='advanced-stocks-v0',
    entry_point=AdvancedEnv,
    kwargs={
        'df': deepcopy(datasets.STOCKS_GOOGL),
        'window_size': 30,
        'frame_bound': (30, len(datasets.STOCKS_GOOGL))
    }
)