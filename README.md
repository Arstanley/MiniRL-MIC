# reinforcement_learning
## Overview
This project uses WebGME to build a design studio that automatically trains openAI gym reinforcement learning agents. Users can design experiments by selecting an environment, choosing an RL algorithm/policy, and specifying training settings such as hyperparameters, network architecture, and runtime options. The design studio uses this configuration and automatically trains an agent with the specified parameters. After the agent is trained, the studio evaluates the agent on a rollout of the environment and displays a gif of the agent's policy and the downloadable trained model artifact.

## Installation
First, install the reinforcement_learning following:
- [NodeJS](https://nodejs.org/en/) (LTS recommended)
- [MongoDB](https://www.mongodb.com/)
- [gymnasium](https://gymnasium.farama.org/)
- [pytorch](https://pytorch.org/)
- [stable_baselines3](https://stable-baselines3.readthedocs.io/en/master/)

Second, start mongodb locally by running the `mongod` executable in your mongodb installation (you may need to create a `data` directory or set `--dbpath`).

Then, run `webgme start` from the project root to start . Finally, navigate to `http://localhost:8888` to start using reinforcement_learning!
