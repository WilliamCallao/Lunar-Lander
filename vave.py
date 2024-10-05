import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import sys

# Crear el entorno
env = gym.make("LunarLander-v2", render_mode="human")

# Definir una clase de callback para monitorear el entrenamiento
class CustomTrainingCallback(BaseCallback):
    def __init__(self, verbose=0, print_freq=100):
        super(CustomTrainingCallback, self).__init__(verbose)
        self.total_steps = 0
        self.print_freq = print_freq  # Frecuencia para mostrar los mensajes

    def _on_step(self) -> bool:
        self.total_steps += 1
        # Obtener detalles sobre el paso actual del entorno
        reward = self.locals['rewards']
        action = self.locals['actions']
        done = self.locals['dones']
        episode_reward = sum(reward)

        # Mostrar detalles cada 'print_freq' pasos
        if self.total_steps % self.print_freq == 0:
            message = f"Paso {self.total_steps} | Acción: {action} | Recompensa: {reward} | Recompensa acumulada: {episode_reward} | ¿Terminado?: {'Sí' if done else 'No'}"
            sys.stdout.write(f"\r{message.ljust(100)}")
            sys.stdout.flush()  # Asegurar que los mensajes se impriman de inmediato

        return True

# Inicializar el callback personalizado
training_callback = CustomTrainingCallback(print_freq=100)  # Actualizar cada 100 pasos

# Inicializar el modelo PPO
model = PPO("MlpPolicy", env, verbose=1)

# Entrenar el modelo con el callback personalizado para monitorear el entrenamiento
model.learn(total_timesteps=100000, callback=training_callback)

# Evaluar el modelo después del entrenamiento (opcional)
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
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        env.render()  # Renderizar la navecita en la ventana gráfica
        done = terminated or truncated

    print(f"\nEpisodio {ep+1} terminado con recompensa total: {total_reward}\n")
    sys.stdout.flush()  # Asegurar que el mensaje se imprima de inmediato
