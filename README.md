# Proyecto Lunar Lander

Este proyecto demuestra la evolución del entrenamiento de un modelo de aprendizaje por refuerzo en el entorno Lunar Lander. El objetivo es mostrar el progreso desde un estado no entrenado hasta un agente entrenado capaz de aterrizar con éxito.

## Estructura del Proyecto

- **assets/**
  - Contiene recursos visuales como GIFs.
- **env2/**
  - Configuración del entorno virtual para la gestión de dependencias.
- **Scripts**: Archivos Python para entrenar y visualizar el modelo de Lunar Lander utilizando diferentes enfoques (TensorFlow, Stable-Baselines3, etc.).

## GIFs del Lunar Lander
<p align="center">
  <img src="assets/a.gif" alt="No Entrenado" width="45%">
  <img src="assets/b.gif" alt="Entrenado" width="45%">
</p>

Los GIFs ilustran el rendimiento del agente antes y después del entrenamiento, destacando la mejora en las habilidades de aterrizaje.

## Configuración del Entorno Virtual

Para configurar el proyecto, sigue estos pasos:

1. **Crear un Entorno Virtual**:
    ```bash
    python -m venv env
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\env2\Scripts\Activate
    ```

2. **Instalar Dependencias**:
    ```bash
    pip install swig
    pip install gymnasium[box2d]
    pip install stable_baselines3==2.3.2
    pip install tensorflow==2.17.0
    ```

## Ejecución de los Scripts

Los scripts proporcionados permiten entrenar y evaluar diferentes modelos de aprendizaje por refuerzo. Puedes visualizar el progreso durante y después del entrenamiento.

1. **Entrenar un Modelo (Stable Baselines3 - PPO)**:
    - Usa `train_visual_sb3.py` para entrenar el modelo utilizando la librería Stable-Baselines3.

2. **Entrenar un Modelo (TensorFlow - DQN)**:
    - Usa `train_visual_tf.py` para entrenar un modelo de Red Neuronal Profunda (DQN) utilizando TensorFlow.

3. **Visualizar un Modelo Pre-entrenado**:
    - Usa `load_and_visualize_model.py` para cargar un modelo pre-entrenado y observar su desempeño en el entorno Lunar Lander.

## Dependencias

Este proyecto requiere las siguientes bibliotecas de Python:
- `gymnasium`
- `stable_baselines3`
- `tensorflow`
