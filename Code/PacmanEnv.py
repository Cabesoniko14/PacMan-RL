import gym
class PacmanEnv(gym.Env):
    def __init__(self, render_mode="human", seed=None):
        if render_mode == "human":
            self.env = gym.make("ALE/MsPacman-v5", render_mode="human")
        else:
            self.env = gym.make("ALE/MsPacman-v5")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.seed = seed
        self.lives = self.env.ale.lives()
        self.lives_2 = 3
        
        # Dejamos lo demás igual, nos ayuda mucho que tengamos las posibles acciones
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # custom reward
        if reward > 0: # recibió algún tipo de reward
            # Si el número de vidas de pacman es positivo
            if reward == 10:  # En el caso por default, 10 de reward es por comer un puntito
                reward = 50 # Si come un puntito, que el reward sea 25
            elif reward == 50: # Si come un power pellet que activa el poder comer fantasmas
                reward = 75
            elif reward == 200:  # En el caso por default, hay reward de 200 por comer un fantasma
                reward = 100 # Si come un fantasma, cuánto queremos de reward
                
                
        else: # Si pacman no recibió reward 
            reward = -20  # Cuando camine pero no consiga nada, que busque la ruta óptima. Importante !                
                
                
        if self.lives_2 > info['lives']:
                # Agent lost a life:
                reward -= 1000
                self.lives_2 -= 1
                
        if info['lives'] == 0:
            reward -= 5000
        
        if (info['lives'] > 0) & (terminated or truncated):
            reward = 10000
                
        return observation, reward, terminated, truncated, info

    def reset(self):
        return self.env.reset()

    def render(self, mode=None, render=True):
        if self.render_mode == 'human' and render:
            self.env.render(mode=mode)

    def close(self):
        self.env.close()