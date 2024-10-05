import gymnasium as gym
from stable_baselines3 import PPO
import sys

# Crear el entorno Humanoid-v4 con parámetros personalizados
env = gym.make(
    'Humanoid-v4',
    ctrl_cost_weight=0.1,
    reset_noise_scale=1e-2,
    exclude_current_positions_from_observation=False,
    render_mode='human'  # Habilitar renderizado gráfico
)

# Inicializar el modelo PPO con la política MlpPolicy
model = PPO('MlpPolicy', env, verbose=1)

# Entrenar el modelo
model.learn(total_timesteps=100000)

# Evaluar el modelo después del entrenamiento
episodes = 5
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    print(f"\nEpisodio {ep+1} iniciado:")

    while not done:
        step_count += 1
        # Predecir la acción utilizando el modelo entrenado
        action, _states = model.predict(obs, deterministic=True)
        # Tomar una acción en el entorno
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Mostrar mensajes en cada paso
        message = f"Paso {step_count} | Acción: {action} | Recompensa del paso: {reward} | Recompensa acumulada: {total_reward}"
        print(message)
        sys.stdout.flush()

        # Renderizar el entorno (mostrar el humanoide)
        env.render()

        # Verificar si el episodio ha terminado
        done = terminated or truncated

    print(f"Episodio {ep+1} terminado con recompensa total: {total_reward}\n")

# Cerrar el entorno
env.close()
