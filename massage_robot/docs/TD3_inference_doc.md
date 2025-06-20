# TD3_inference.py 	6 Running Inference with a Trained TD3 Agent

## Overview

This script loads a trained actor model and runs inference in the same environment. It uses the actor network to select actions based on the current state, applies smoothing and oscillation similar to training, and steps through the environment.

## Key Components

- `load_agent()`: Loads the saved actor model weights into an Actor network instance.

- `run_inference()`: Runs a single episode of interaction with the environment using the loaded agent.  
  Uses the same action smoothing and oscillation logic as training.  
  Logs rewards to TensorBoard.

- Main function initializes environment, loads model, and runs inference.
