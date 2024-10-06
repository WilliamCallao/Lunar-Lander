import gymnasium as gym
from stable_baselines3 import PPO
import os
import sys

def main():
    # Definir la ruta del modelo guardado
    MODEL_PATH = "ppo_lunarlander"

    # Verificar si el modelo existe
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"Error: El modelo '{MODEL_PATH}.zip' no existe. Asegúrate de haber entrenado y guardado el modelo antes de ejecutar este script.")
        sys.exit(1)

    # Crear el entorno de evaluación con renderizado
    env = gym.make("LunarLander-v2", render_mode="human")

    # Cargar el modelo PPO entrenado
    model = PPO.load(MODEL_PATH, env=env)
    print("Modelo cargado exitosamente.")

    # Número de episodios para evaluar
    episodes = 5

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0  # Contador de pasos por episodio
        print(f"\nEpisodio {ep+1} iniciado:")
        sys.stdout.flush()  # Asegurar que el mensaje se imprima de inmediato
        while not done:
            step_count += 1
            # Predecir la acción a tomar
            action, _states = model.predict(obs, deterministic=True)
            # Ejecutar la acción en el entorno
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            env.render()  # Renderizar la navecita en la ventana gráfica
            done = terminated or truncated

        print(f"\nEpisodio {ep+1} terminado con recompensa total: {total_reward}\n")
        sys.stdout.flush()  # Asegurar que el mensaje se imprima de inmediato

    # Cerrar el entorno
    env.close()

if __name__ == "__main__":
    main()
