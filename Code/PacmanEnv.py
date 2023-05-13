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
        
        # Dejamos lo demás igual, nos ayuda mucho que tengamos las posibles acciones
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # custom reward
        if reward > 0: # recibió algún tipo de reward
            
            if info['lives'] > 0:  # Si el número de vidas de pacman es positivo
                if reward == 10:  # En el caso por default, 10 de reward es por comer un puntito
                    reward = 25 # Si come un puntito, que el reward sea __
                elif reward == 50: # Si come un power pellet que activa el poder comer fantasmas
                    reward = 60
                elif reward == 200:  # En el caso por default, hay reward de 200 por comer un fantasma
                    reward = 100 # Si come un fantasma, cuánto queremos de reward
            else: # Si a pacman se le acabaron las vidas
                done = True  # Pone en true la bandera de que acabó el juego
                reward = -500  # Si perdemos, cuánto queremos que pierda de reward. Que le duela al pacman
                
        else:
            reward = -1  # Cuando camine pero no consiga nada, que busque la ruta óptima. Importante !
            
            # Castigo por perder vidas
            lives = info['lives']
            if self.env.ale.lives() < lives:
                # Agent lost a life:
                reward -= 1000

        return observation, reward, terminated, truncated, info

    def reset(self):
        return self.env.reset()

    def render(self, mode=None, render=True):
        if self.render_mode == 'human' and render:
            self.env.render(mode=mode)

    def close(self):
        self.env.close()