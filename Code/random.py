# -------------------------------- RANDOM RUN: PACMAN ------------------------------------------

from PacmanEnv import PacmanEnv

env_new = PacmanEnv(seed = 45, render_mode = 'human')
observacion, info = env_new.reset() # Resetear el environment a uno aleatorio, pero siempre el mismo
frames = [] # almacenar los frames
for _ in range(1000):  # Por cada paso, que son 100, ejecuta lo siguiente
    observacion, reward, terminated, truncated, info = env_new.step(env_new.action_space.sample())
    frames.append(observacion)

    if terminated or truncated:
        observacion, info = env_new.reset()

env_new.close()

observation, reward, terminated, truncated, info = env_new.step(env_new.action_space.sample())
print(observation)
