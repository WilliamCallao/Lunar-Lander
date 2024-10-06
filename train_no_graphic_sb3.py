import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import sys
import os

MODEL_PATH = "ppo_lunarlander"

env = gym.make("LunarLander-v2")

class CustomTrainingCallback(BaseCallback):
    def __init__(self, verbose=0, print_freq=100):
        super(CustomTrainingCallback, self).__init__(verbose)
        self.total_steps = 0
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        self.total_steps += 1
        reward = self.locals.get('rewards')
        action = self.locals.get('actions')
        done = self.locals.get('dones')
        episode_reward = sum(reward) if reward is not None else 0

        if self.total_steps % self.print_freq == 0:
            message = (f"Paso {self.total_steps} | Acción: {action} | Recompensa: {reward} | "
                       f"Recompensa acumulada: {episode_reward} | "
                       f"¿Terminado?: {'Sí' if done else 'No'}")
            sys.stdout.write(f"\r{message.ljust(100)}")
            sys.stdout.flush()

        return True

training_callback = CustomTrainingCallback(print_freq=100)  # Actualizar cada 100 pasos

if os.path.exists(f"{MODEL_PATH}.zip"):
    model = PPO.load(MODEL_PATH, env=env)
    print("Modelo cargado desde la ruta existente.")
else:
    model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=1000000, callback=training_callback)

    model.save(MODEL_PATH)
    print(f"Modelo entrenado y guardado en '{MODEL_PATH}.zip'.")

env_eval = gym.make("LunarLander-v2", render_mode="human")

episodes = 5
for ep in range(episodes):
    obs, info = env_eval.reset()
    done = False
    total_reward = 0
    step_count = 0
    print(f"\nEpisodio {ep+1} iniciado:")
    sys.stdout.flush()
    while not done:
        step_count += 1
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_eval.step(action)
        total_reward += reward

        env_eval.render()
        done = terminated or truncated

    print(f"\nEpisodio {ep+1} terminado con recompensa total: {total_reward}\n")
    sys.stdout.flush()
    
env.close()
env_eval.close()
