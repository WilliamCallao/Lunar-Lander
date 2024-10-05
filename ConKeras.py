import gymnasium as gym
import time

# Cargar el entorno de LunarLander-v2
env = gym.make("LunarLander-v2", render_mode="human")

# Reiniciar el entorno para empezar
state, info = env.reset()

done = False
while not done:
    # Seleccionar una acción aleatoria
    action = env.action_space.sample()

    # Ejecutar la acción en el entorno
    next_state, reward, terminated, truncated, info = env.step(action)

    # Verificar si el episodio ha terminado
    done = terminated or truncated

    # Renderizar la simulación
    env.render()

    # Pequeña pausa para que se pueda visualizar bien la simulación
    time.sleep(0.02)

# Cerrar el entorno una vez finalizado
env.close()
