# TD3_inference.py 	6 Running Inference with a Trained TD3 Agent

## Overview

This script loads a trained actor model and runs inference in the same environment. It uses the actor network to select actions based on the current state, applies smoothing and oscillation similar to training, and steps through the environment.

## Key Components

- `load_agent()`: Loads the saved actor model weights into an Actor network instance.

- `run_inference()`: Runs a single episode of interaction with the environment using the loaded agent.  
  Uses the same action smoothing and oscillation logic as training.  
  Logs rewards to TensorBoard.

- Main function initializes environment, loads model, and runs inference.

## Massage graphs before TD3 Training/Tuning

![image](https://github.com/user-attachments/assets/0fc6ca64-387c-4fbd-b8e1-8ec961245e04)


## Massage graphs after TD3 Training/Tuning

![Picture1](https://github.com/user-attachments/assets/99f336d9-2a3a-45d8-b075-da85c1c169fb)
