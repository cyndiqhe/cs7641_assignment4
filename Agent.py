import numpy as np



class Agent:
  
  def __init__(self, n_states, n_actions, decay_rate=0.0001, learning_rate=0.7, gamma=0.618):
    self.n_actions = n_actions
    self.q_table = np.zeros((n_states, n_actions))
    self.epsilon = 1.0
    self.max_epsilon = 1.0
    self.min_epsilon = 0.01
    self.decay_rate = decay_rate
    self.learning_rate = learning_rate
    self.gamma = gamma # discount rate
    self.epsilons_ = []
    
  def choose_action(self, state, explore=True):
    exploration_tradeoff = np.random.uniform(0, 1)
    
    if explore and exploration_tradeoff < self.epsilon:
      # exploration
      return np.random.randint(self.n_actions)    
    else:
      # exploitation (taking the biggest Q value for this state)
      return np.argmax(self.q_table[state, :])
  
  def learn(self, state, action, reward, next_state, done, episode):
    # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
    self.q_table[state, action] = self.q_table[state, action] + \
      self.learning_rate * (reward + self.gamma * \
        np.max(self.q_table[next_state, :]) - self.q_table[state, action])
    
    if done:
      # Reduce epsilon to decrease the exploration over time
      self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
        np.exp(-self.decay_rate * episode)
      self.epsilons_.append(self.epsilon)





 
