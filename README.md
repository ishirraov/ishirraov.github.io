<html lang="en">
  <head>
    <title>Project Code</title>
  </head>
  <body>
    <pre>
    <code>from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt
import random
import math
import tensorflow as tf
from tensorflow import keras
from collections import deque
from tqdm import tqdm


MEMORY_SIZE = 7000000
memory = deque(maxlen=MEMORY_SIZE) #STORAGE
reward_memory_total = deque(maxlen=MEMORY_SIZE)
terminal_memory_total = deque(maxlen=MEMORY_SIZE)


class Simulation:

  def __init__(self, beta, sigma, gamma, mu, susceptible_population, exposed_population, infected_population, recovered_population, dead_population, current_economy):
    self.beta = beta #transmission rate
    self.sigma = sigma #incubation rate
    self.gamma = gamma #recovery rate
    self.mu = mu #death rate
    self.susceptible_population = susceptible_population
    self.exposed_population = exposed_population
    self.infected_population = infected_population
    self.recovered_population = recovered_population
    self.dead_population = dead_population
    self.current_economy = current_economy
    self.economic_impact_from_actions = 1
    self.day = [0,1]
    self.population_size = self.susceptible_population + self.infected_population
  
  def seir_f(self, t, y, beta, sigma, gamma, mu):
        s, e, i, r, d = y
        N = self.population_size
        return np.array([(-beta * i * s)/N,
                         -sigma * e + (beta * i * s)/N, 
                         -(gamma * i) -(mu * i)+ sigma * e, 
                         (gamma) * i,
                         (mu * i)])

  def step(self):
    y = [self.susceptible_population, self.exposed_population, self.infected_population, self.recovered_population, self.dead_population] #current state

    sol = solve_ivp(self.seir_f, self.day, y, args=(self.beta, self.sigma, self.gamma, self.mu),t_eval = self.day)

    self.day[0] += 1
    self.day[1] += 1
    
    #new state
    self.susceptible_population, self.exposed_population, self.infected_population, self.recovered_population, self.dead_population = sol.y.T[1][0], sol.y.T[1][1], sol.y.T[1][2],sol.y.T[1][3],sol.y.T[1][4]
    self.current_economy = (self.susceptible_population + self.exposed_population + self.recovered_population)*self.economic_impact_from_actions
    
  def run(self, run_type = 'natural'):
    if run_type == 'natural':
      i = 0
      while self.infected_population >= 1:
        i+=1
        self.step()
      print(i)
    else:
      for i in range(run_type):
        self.step()
        
class Agent:
  def __init__(self, name):
    self.name = name
    self.model, self.target = None, None
    self.discount_factor = 0.95
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    self.loss_fn = tf.keras.losses.mean_squared_error
    self.time_steps = 30
    self.input_shape = 7 #SEIRDE + Action
    self.n_outputs = 4

  def set_model(self):
    model = keras.Sequential([
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True), input_shape=(self.time_steps,self.input_shape)),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Dense(128, activation = 'relu'),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(32, activation = 'relu'),
        keras.layers.Dense(self.n_outputs, activation = 'softmax')
    ])
    self.model = model
    self.model.compile(loss=self.loss_fn, optimizer = self.optimizer)
    self.target=model
  
  def epsilon_greedy_policy(self,state, epsilon):
    #small probability of exploring - epsilon is probability of exploration, np.random.rand() gives # between 0,1
    if np.random.rand() < epsilon:
      return random.randint(0,self.n_outputs-1)
    #predict next action using model
    else:
      state_array = np.array(state)
      state_array = state_array[...,np.newaxis]
      Q_values = self.model.predict(state_array)
      prediction = np.argmax(Q_values[0][0])
      return prediction
  
  def calculate_reward(self,current_economy,dead_population,infected_population, total):
    r = 12 #weight of economy vs cases
    s = 5 #weight of deaths
    #economy
    Et = current_economy/total
    #deaths
    Dt = s* (dead_population/total)
    #active cases
    At = -r * (infected_population/total) * 100
    #REWARD FUNCTION
    reward =  Et * math.e**(At)+Dt
    return reward

  def update_model(self,current_states, next_states, model_history):
    rewards = []
    terminals = []

    for i in range(1,31):
      rewards.append(reward_memory_total[-i])#last 30 days
      terminals.append(terminal_memory_total[-i])
    
    next_Q_values = (self.model.predict(next_states))[0] #predict next
    all_Q_values = (self.model.predict(current_states))[0] #predict i-1
    prediction = (self.target.predict(next_states))[0] #use target to find next
    #ALL ARE (1,30,4)

    ACTION_HISTORY = []

    for i in range(30):
      ACTION_HISTORY.append(current_states[0][i][6]) #retrieve actions

    best_next_actions = []
    for i in range(len(next_Q_values)):
      best_next_actions.append(np.argmax(next_Q_values[i]))
    
    next_mask = tf.one_hot(best_next_actions, self.n_outputs).numpy()

    next_best_Q_values = (prediction * next_mask).sum(axis=1)

    target_Q_values = []

    for i in range(30):
      target_Q_values.append(rewards[i] + (1-terminals[i]) * self.discount_factor * next_best_Q_values[i])
      action = ACTION_HISTORY[i] #action taken at every step  
      all_Q_values[i][int(action)] = target_Q_values[i] #this is like "informal" mask

    all_Q_values = [all_Q_values.tolist()]
    all_Q_values = np.asarray(all_Q_values)

    history = self.model.fit(current_states, all_Q_values, verbose = 0) #calculate loss, update weights
    model_history.append(history.history['loss'])


def get_current_states():
  current_states = []
  for i in range(1,31):
    current_states.append(memory[-i])
  current_states = np.array(current_states[::-1])
  current_states = current_states[np.newaxis,...]
  return current_states


def get_previous_states():
  previous_states = []
  for i in range(2,32):
    previous_states.append(memory[-i])
  previous_states = np.array(previous_states[::-1])
  previous_states = previous_states[np.newaxis,...]
  return previous_states

def execute_action(sim, action):
  if action == 0: #none
    sim.beta = beta
    sim.sigma = sigma
    sim.gamma = gamma
    sim.mu = mu
    sim.economic_impact_from_actions = 1
  
  if action == 1: #social distance
    factor = 0.5
    sim.beta = beta*factor
    sim.sigma = sigma*factor
    sim.gamma = gamma*factor
    sim.mu = mu *factor
    sim.economic_impact_from_actions = factor
  
  if action == 2: #lockdown
    factor = 0.25
    sim.beta = beta*factor
    sim.sigma = sigma*factor
    sim.gamma = gamma*factor
    sim.mu = mu*factor
    sim.economic_impact_from_actions = factor
  
  if action == 3: #curfew + lockdown
    factor = 0.15
    sim.beta = beta*factor
    sim.sigma = sigma*factor
    sim.gamma = gamma*factor
    sim.mu = mu*factor
    sim.economic_impact_from_actions = factor

agent = Agent('agent_1')
agent.set_model()
num_episodes = 200

cum_reward_graph = []

model_history = []

#da super loop

for episode in tqdm(range(num_episodes)):
  #values for simulation when t=0, set according to real-world data
  # beta = random.randint(25,30)*0.01 #infection rate
  # sigma = random.randint(30,50)*0.01 #incubation rate
  # gamma = random.randint(10,30)*0.01 # recovery rate
  # mu=random.randint(15,20)*0.001 #mortality rate
  # population = random.randint(200,1000) #population
  # E0, I0, R0, D0 = 0,random.randint(10,60)*population*0.01,0, 0 #infected can be 10-60% of the population
  beta = 0.12 #infection
  sigma = 1 #incubation
  gamma = (1/27) #recovery
  mu = 0.009  #mortality
  population = 1000
  E0, I0, R0, D0 = 0, 70, 0, 0
  S0 = population - E0 - I0 - R0 - D0
  econ_0 = S0
  #initialize simulation
  sim = Simulation(beta,sigma,gamma,mu,S0,E0,I0,R0,D0,econ_0) 

  #cumulative reward
  cumulative_reward = 0

  #run for a 30 day buffer first time through
  if episode == 0:
    for i in range(31):
      sim.run(run_type = 1)
      memory.append([sim.susceptible_population, sim.exposed_population, sim.infected_population, sim.recovered_population, sim.dead_population, sim.current_economy, 0])
      reward = agent.calculate_reward(sim.current_economy, sim.dead_population, sim.infected_population,sim.population_size)
      cumulative_reward += reward
      reward_memory_total.append(reward)
      terminal_memory_total.append(0)
  #calculate epsilon for policy
  epsilon = max(1- (episode/200),0.01)

  while sim.infected_population >= 1:

    sim.run(run_type = 1)

    current_states = get_current_states() #update 30 day list of SEIRDE
    if episode == 0 and sim.day[0] == 32:
      pass
    else:
      previous_states = get_previous_states()
      agent.update_model(previous_states, current_states, model_history)
    
    action = agent.epsilon_greedy_policy(current_states, epsilon) #ADD ACTION EFFECT ON SIM
    memory.append([sim.susceptible_population, sim.exposed_population, sim.infected_population, sim.recovered_population, sim.dead_population, sim.current_economy,action])

    reward = agent.calculate_reward(sim.current_economy, sim.dead_population, sim.infected_population, sim.population_size)
    cumulative_reward += reward
    reward_memory_total.append(reward)

    execute_action(sim, action)

    if sim.infected_population >= 1:
      terminal_memory_total.append(0)

  terminal_memory_total.append(1)
  cum_reward_graph.append(cumulative_reward*(1/sim.day[0]))

#graph all information


plt.plot(model_history) #loss
print(model_history)
plt.show()

plt.plot(cum_reward_graph) #reward
print(cum_reward_graph)
plt.show()
print('we have finished (woohoo!)')

#save model
version = 1
fn = ("C:/Users/ishir/OneDrive/Documents/Model_Weights/" + str(version))
agent.model.save_weights(fn)

#TEST MODEL
#values for simulation when t=0, set according to real-world data
beta = 0.12
sigma = 1
gamma = (1/27)
mu = 0.009
population = 1000
E0, I0, R0, D0 = 0, 70, 0, 0
S0 = population - E0 - I0 - R0 - D0
econ_0 = S0
#initialize simulation
sim = Simulation(beta,sigma,gamma,mu,S0,E0,I0,R0,D0,econ_0) 

action_hist = []
susceptible_hist = []
exposed_hist = []
infected_hist = []
recovered_hist = []
death_hist = []
economy_hist = []
reward_hist = []

epsilon = 0 #always use model

#doesn't update model

while sim.infected_population >= 1:
  sim.run(run_type = 1)

  current_states = get_current_states()
  
  print(sim.day[0])
  
  action = agent.epsilon_greedy_policy(current_states, epsilon)
  action_hist.append(action)

  susceptible_hist.append(sim.susceptible_population)
  exposed_hist.append(sim.exposed_population)
  infected_hist.append(sim.infected_population)
  recovered_hist.append(sim.recovered_population)
  death_hist.append(sim.dead_population)
  economy_hist.append(sim.current_economy)
  memory.append([sim.susceptible_population, sim.exposed_population, sim.infected_population, sim.recovered_population, sim.dead_population, sim.current_economy, action])

  reward = agent.calculate_reward(sim.current_economy, sim.dead_population, sim.infected_population, sim.population_size)
  reward_hist.append(reward)

  execute_action(sim, action)


fig, ax = plt.subplots()
ax.grid()
ax.margins(0)

ax.plot(susceptible_hist, label = 'susceptible')
ax.plot(exposed_hist, label = 'exposed')
ax.plot(infected_hist, label = 'infected')
ax.plot(recovered_hist, label = 'recovered')
ax.plot(death_hist, label = 'deaths')
ax.plot(economy_hist, label = 'economy')

x_new_start = 0

for i in range(len(action_hist)-1):
  current = action_hist[i]
  next =  action_hist[i+1]
  print(current)

  if current == 0:
    color = 'blue'
  elif current == 1:
    color = 'green'
  elif current == 2:
    color = 'yellow'
  elif current == 3:
    color = 'red'
  ax.axvspan(i, i+1, facecolor = color, alpha = 0.4)

ax.legend()
plt.show()

fig, ax = plt.subplots()

ax.plot(reward_hist)
plt.show()

def run_action(action_type, color):
  #TEST MODEL
  #values for simulation when t=0, set according to real-world data
  beta = 0.12
  sigma = 1
  gamma = (1/27)
  mu = 0.009
  population = 1000
  E0, I0, R0, D0 = 0, 70, 0, 0
  S0 = population - E0 - I0 - R0 - D0
  econ_0 = S0
  #initialize simulation
  sim = Simulation(beta,sigma,gamma,mu,S0,E0,I0,R0,D0,econ_0) 

  action_hist = []
  susceptible_hist = []
  exposed_hist = []
  infected_hist = []
  recovered_hist = []
  death_hist = []
  economy_hist = []
  reward_hist = []

  while sim.infected_population >= 1:
    sim.run(run_type = 1)

    action_hist.append(action_type)

    susceptible_hist.append(sim.susceptible_population)
    exposed_hist.append(sim.exposed_population)
    infected_hist.append(sim.infected_population)
    recovered_hist.append(sim.recovered_population)
    death_hist.append(sim.dead_population)
    economy_hist.append(sim.current_economy)

    reward = agent.calculate_reward(sim.current_economy, sim.dead_population, sim.infected_population, sim.population_size)
    reward_hist.append(reward)

    execute_action(sim, action_type)

  fig, ax = plt.subplots()
  ax.grid()
  ax.margins(0)

  ax.plot(susceptible_hist, label = 'susceptible')
  ax.plot(exposed_hist, label = 'exposed')
  ax.plot(infected_hist, label = 'infected')
  ax.plot(recovered_hist, label = 'recovered')
  ax.plot(death_hist, label = 'deaths')
  ax.plot(economy_hist, label = 'economy')

  ax.axvspan(0, sim.day[1], facecolor = color, alpha = 0.4)

  ax.legend()
  plt.show()

  fig, ax = plt.subplots()

  ax.plot(reward_hist)
  plt.show()

  return infected_hist

none = run_action(0, 'blue')
one = run_action(1,'green')
two = run_action(2,'yellow')
three = run_action(3, 'red')

plt.plot(none)
plt.plot(one)
plt.plot(two)
plt.plot(three)
      </code>
      </pre>
      </body>
    </html>
