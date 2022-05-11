class AdvancedEnv(StocksEnv):

  def __init__(self, df, window_size, frame_bound):
      super().__init__(df, window_size, frame_bound)
      self.trade_fee_bid_percent = 0  # unit
      self.trade_fee_ask_percent = 0  # unit

  def reset(self):
    obs = super().reset()
    self._total_reward = 1
    return obs

  def _update_mdd():
    pass

  def update_sharpe():
    pass

  def _get_observation(self):
      obs = self.signal_features[(self._current_tick-self.window_size):self._current_tick]
      obs = np.array(obs)
      prices = obs[1:, 1]
      changes = obs[:-1, 0]
      opt = prices/changes
      return opt

  def step(self, action):
    self._done = False
    self._current_tick += 1

    if self._current_tick == self._end_tick:
            self._done = True

    step_reward = self._calculate_reward(action)
    self._total_reward *= (1+step_reward)

    self._update_profit(action)

    trade = False
    if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True
    if trade:
        self._position = self._position.opposite()
        self._last_trade_tick = self._current_tick

    self._position_history.append(self._position)
    observation = self._get_observation()
    info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
    )
    self._update_history(info)

    return observation, step_reward, self._done, info
    
  def _calculate_reward(self, action):
        
        current_price = self.prices[self._current_tick]
        last_price = self.prices[self._current_tick-1]
    
        percent_diff = (current_price-last_price)/last_price
        
        if self._position == Positions.Short:
            return float(-percent_diff)
        else:
            return float(percent_diff)
            
        