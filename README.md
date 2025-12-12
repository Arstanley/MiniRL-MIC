# reinforcement_learning
## Overview
This project uses WebGME to build a design studio that automatically trains openAI gym reinforcement learning agents. Users can configure experiments by customizing the agent parameters, policy, and neural network architectures as well as switch between gym environments. The results of the trained agent are displayed in the evaluation of the agent and allows for downloading the trained model artifact.

## Installation
First, install the reinforcement_learning following:
- [NodeJS](https://nodejs.org/en/) (LTS recommended)
- [MongoDB](https://www.mongodb.com/)
- gymnasium
- pytorch
- stable_baselines3

Second, start mongodb locally by running the `mongod` executable in your mongodb installation (you may need to create a `data` directory or set `--dbpath`).

Then, run `webgme start` from the project root to start . Finally, navigate to `http://localhost:8888` to start using reinforcement_learning!
