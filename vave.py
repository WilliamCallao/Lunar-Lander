import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np

# Crear el entorno
env = gym.make("LunarLander-v2", render_mode="human")

# Definir una clase de callback para registrar recompensas y agregar gráficos llamativos
class RewardPlotCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardPlotCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.mean_rewards = []
        self.fig, self.ax = plt.subplots()
        plt.ion()
        plt.show()

    def _on_step(self) -> bool:
        # Verificar si el episodio ha terminado
        if 'episode' in self.locals:
            episode_info = self.locals['episode']
            reward = episode_info['r']
            self.episode_rewards.append(reward)
            # Calcular la recompensa media de las últimas 100 episodios
            if len(self.episode_rewards) >= 100:
                mean_reward = np.mean(self.episode_rewards[-100:])
                self.mean_rewards.append(mean_reward)
                # Actualizar el gráfico con colores más llamativos
                self.ax.clear()
                self.ax.set_facecolor('#1c1c1e')  # Fondo oscuro (negro profundo)
                self.ax.plot(self.mean_rewards, color='#FFDD44', lw=2)  # Línea amarilla vibrante
                self.ax.set_xlabel('Episodios (media sobre 100)', color='#FFEEAA')  # Texto amarillento
                self.ax.set_ylabel('Recompensa Media', color='#FFEEAA')
                self.ax.set_title('Progreso del Entrenamiento', color='#FFEEAA')
                # Cambiar color de los bordes y etiquetas a un dorado vibrante
                self.ax.spines['top'].set_color('#FFAA00')
                self.ax.spines['bottom'].set_color('#FFAA00')
                self.ax.spines['left'].set_color('#FFAA00')
                self.ax.spines['right'].set_color('#FFAA00')
                self.ax.tick_params(axis='x', colors='#FFEEAA')
                self.ax.tick_params(axis='y', colors='#FFEEAA')
                plt.pause(0.001)
        return True

# Función para mostrar una explosión cuando la nave se estrella
def show_explosion():
    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    x, y = np.random.uniform(-1, 1, 100), np.random.uniform(-1, 1, 100)
    scatter = ax.scatter(x, y, s=200, c=np.random.choice(['#FF5500', '#FFAA00', '#FF0000']), alpha=0.8)
    plt.title('¡Explosión!', color='#FFEEAA', fontsize=16)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    plt.pause(0.5)  # Mostrar la explosión por 0.5 segundos
    plt.close(fig)

# Inicializar el callback
reward_callback = RewardPlotCallback()

# Inicializar el modelo PPO
model = PPO("MlpPolicy", env, verbose=1)

# Entrenar el modelo con el callback
model.learn(total_timesteps=100000, callback=reward_callback)

# Evaluar el modelo después del entrenamiento con detección de colisiones
episodes = 5
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        done = terminated or truncated
        if terminated and reward < -100:  # Si el aterrizaje fue fallido
            show_explosion()
