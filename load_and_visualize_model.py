import gymnasium as gym
from stable_baselines3 import PPO
import os
import sys

def main():
    MODEL_PATH = "trained_model_1M"

    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"Error: El modelo '{MODEL_PATH}.zip' no existe. Asegúrate de haber entrenado y guardado el modelo antes de ejecutar este script.")
        sys.exit(1)

    env = gym.make("LunarLander-v2", render_mode="human")

    model = PPO.load(MODEL_PATH, env=env)
    print("Modelo cargado exitosamente.")

    episodes = 5

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0 
        print(f"\nEpisodio {ep+1} iniciado:")
        sys.stdout.flush()
        while not done:
            step_count += 1
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            env.render()
            done = terminated or truncated

        print(f"\nEpisodio {ep+1} terminado con recompensa total: {total_reward}\n")
        sys.stdout.flush()

    env.close()

if __name__ == "__main__":
    main()
