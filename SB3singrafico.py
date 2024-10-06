import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import sys
import os

# Definir la ruta para guardar el modelo
MODEL_PATH = "ppo_lunarlander"

# Crear el entorno de entrenamiento sin renderizado
env = gym.make("LunarLander-v2")

# Definir una clase de callback para monitorear el entrenamiento
class CustomTrainingCallback(BaseCallback):
    def __init__(self, verbose=0, print_freq=100):
        super(CustomTrainingCallback, self).__init__(verbose)
        self.total_steps = 0
        self.print_freq = print_freq  # Frecuencia para mostrar los mensajes

    def _on_step(self) -> bool:
        self.total_steps += 1
        # Obtener detalles sobre el paso actual del entorno
        reward = self.locals.get('rewards')
        action = self.locals.get('actions')
        done = self.locals.get('dones')
        episode_reward = sum(reward) if reward is not None else 0

        # Mostrar detalles cada 'print_freq' pasos
        if self.total_steps % self.print_freq == 0:
            message = (f"Paso {self.total_steps} | Acción: {action} | Recompensa: {reward} | "
                       f"Recompensa acumulada: {episode_reward} | "
                       f"¿Terminado?: {'Sí' if done else 'No'}")
            sys.stdout.write(f"\r{message.ljust(100)}")
            sys.stdout.flush()  # Asegurar que los mensajes se impriman de inmediato

        return True

# Inicializar el callback personalizado
training_callback = CustomTrainingCallback(print_freq=100)  # Actualizar cada 100 pasos

# Verificar si ya existe un modelo guardado
if os.path.exists(f"{MODEL_PATH}.zip"):
    # Cargar el modelo existente
    model = PPO.load(MODEL_PATH, env=env)
    print("Modelo cargado desde la ruta existente.")
else:
    # Inicializar el modelo PPO
    model = PPO("MlpPolicy", env, verbose=1)

    # Entrenar el modelo con el callback personalizado para monitorear el entrenamiento
    model.learn(total_timesteps=1000000, callback=training_callback)

    # Guardar el modelo entrenado
    model.save(MODEL_PATH)
    print(f"Modelo entrenado y guardado en '{MODEL_PATH}.zip'.")

# Crear el entorno de evaluación con renderizado
env_eval = gym.make("LunarLander-v2", render_mode="human")

# Evaluar el modelo después del entrenamiento (opcional)
episodes = 5
for ep in range(episodes):
    obs, info = env_eval.reset()
    done = False
    total_reward = 0
    step_count = 0  # Contador de pasos por episodio
    print(f"\nEpisodio {ep+1} iniciado:")
    sys.stdout.flush()  # Asegurar que el mensaje se imprima de inmediato
    while not done:
        step_count += 1
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_eval.step(action)
        total_reward += reward

        env_eval.render()  # Renderizar la navecita en la ventana gráfica
        done = terminated or truncated

    print(f"\nEpisodio {ep+1} terminado con recompensa total: {total_reward}\n")
    sys.stdout.flush()  # Asegurar que el mensaje se imprima de inmediato

# Cerrar los entornos
env.close()
env_eval.close()
