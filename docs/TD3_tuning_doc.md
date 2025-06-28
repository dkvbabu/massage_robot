# TD3_tuning.py 	6 Hyperparameter Tuning for TD3 Agent

## Overview

This script performs grid search hyperparameter tuning for the TD3 agent. It tests combinations of learning rates, batch sizes, discount factors, tau values, policy noise, noise clipping, and policy update frequencies. It runs short training episodes for each combination, collects performance metrics, and selects the best hyperparameters based on a weighted score of average reward and losses.

## Key Components

- Defines ranges for hyperparameters to test.

- For each combination: 
  - Initializes environment and agent with those parameters.
  - Runs a limited number of episodes.
  - Tracks rewards, actor loss, and critic loss.
  - Implements early stopping based on critic loss spikes or low rewards.

- After all tests, normalizes metrics and computes a combined score.

- Prints and plots the best hyperparameters and performance metrics.
