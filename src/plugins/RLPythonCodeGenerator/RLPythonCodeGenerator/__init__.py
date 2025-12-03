import sys
import os
import tempfile
import logging
import io 
import json
import imageio.v2 as imageio
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO, DQN, A2C, DDPG, SAC, TD3
from webgme_bindings import PluginBase

logger = logging.getLogger('RLPythonCodeGenerator')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

class RLPythonCodeGenerator(PluginBase):
    def main(self):
        core = self.core
        root_node = self.root_node
        active_node = self.active_node
        children = core.load_children(active_node)


        for child in children:

            meta_type = core.get_meta_type(child)
            meta_name = core.get_attribute(meta_type, 'name') if meta_type else ''
            
            if meta_name != 'Training_Run':
                continue

            run_node = child
            run_name = core.get_attribute(child, 'name')
            logger.info(f'Processing Training Run: {run_name}')
            
            run_params = {
                'batch_size': core.get_attribute(child, 'batch_size'),
                'timesteps': core.get_attribute(child, 'timesteps')
            }
            agent_node = self.get_pointer_node(child, 'Agent')
            env_node = self.get_pointer_node(child, 'Environment')
            arch_node = self.get_pointer_node(child, 'Architecture')
            
            agent_params = {
                'algorithm': core.get_attribute(agent_node, 'algorithm'),
                'policy_type': core.get_attribute(agent_node, 'policy_type'),
                'learning_rate': core.get_attribute(agent_node, 'learning_rate'),
                'discount_factor': core.get_attribute(agent_node, 'discount_factor'),
            }

            arch_params = {
                'activation': core.get_attribute(arch_node, 'activation'),
                'layer_size': core.get_attribute(arch_node, 'layer_size'),
                'num_layers': core.get_attribute(arch_node, 'num_layers'),
                'arch_type': core.get_attribute(arch_node, 'type')
            }

            env_params = {
                'env_id': core.get_attribute(env_node, 'env_id'),
                'seed': core.get_attribute(env_node, 'seed'),
            }

            self.train(run_node, run_name, env_params, run_params, agent_params, arch_params)

        self.util.save(
            root_node, self.commit_hash, 'master', 'Ran RL training with SB3'
        )

    def get_pointer_node(self, node, pointer_name):
        path = self.core.get_pointer_path(node, pointer_name)
        if path:
            return self.core.load_by_path(self.root_node, path)
        return None
    
    def train(self, run_node, run_name, env_params, run_params, agent_params, arch_params):
        core = self.core
        logger.info(f"Training run: {run_name}")

        algo_map = {"PPO": PPO, "DQN": DQN, "A2C": A2C, "DDPG": DDPG, "SAC": SAC, "TD3": TD3}
        algorithm = algo_map.get(agent_params['algorithm'])
        act_map = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}
        activation_fn = act_map.get(arch_params['activation'])
        net_arch = [arch_params['layer_size']] * max(1, int(arch_params['num_layers']))

        policy_kwargs = dict(
            activation_fn=activation_fn, 
            net_arch=net_arch)
        
        train_env = gym.make(env_params['env_id'])
        train_env.reset(seed=env_params['seed'])

        algo_kwargs = dict(
            learning_rate=agent_params['learning_rate'],
            gamma=agent_params['discount_factor'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="cpu",
        )
        if agent_params['algorithm'] in ("PPO", "DQN"):
            algo_kwargs["batch_size"] = run_params["batch_size"]

        logger.info(
                    f"Training with timesteps={run_params['timesteps']}, "
                    f"lr={agent_params['learning_rate']}, gamma={agent_params['discount_factor']}"
                )
        model = algorithm(agent_params["policy_type"], train_env, **algo_kwargs)
        model.learn(total_timesteps=run_params["timesteps"])
        train_env.close()
        logger.info(f"Finished training for run '{run_name}'")

        tmp_model_path = os.path.join(
            tempfile.gettempdir(), f"{run_name}_model.zip"
        )
        model.save(tmp_model_path)
        with open(tmp_model_path, "rb") as f:
            model_bytes = f.read()

        eval_env = gym.make(env_params["env_id"], render_mode="rgb_array")
        eval_env.reset(seed=env_params["seed"])    

        frames = []
        obs, _ = eval_env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = eval_env.step(action)   
            frame = eval_env.render()
            frames.append(frame)
            if done or truncated:
                obs, _ = eval_env.reset()
        eval_env.close()

        config = {
            "run_name": run_name,
            "algorithm": agent_params['algorithm'],
            "policy_type": agent_params['policy_type'],
            "env": env_params,
            "run": run_params,
            "agent": {
                "learning_rate": agent_params['learning_rate'],
                "discount_factor": agent_params['discount_factor'],
            },
            "architecture": arch_params,
            "policy_kwargs": {
                "activation_fn": arch_params['activation'],
                "net_arch": net_arch,
            },
            "algo_kwargs": {
                k: v for k, v in algo_kwargs.items() if k != "policy_kwargs"
            },
        }
        config_str = json.dumps(config, indent=2)
 
        buffer = io.BytesIO()
        imageio.mimsave(buffer, frames, format="GIF", fps=30)
        gif_bytes = buffer.getvalue()

        output_dir = "./rl_outputs"
        os.makedirs(output_dir, exist_ok=True)
        gif_path = os.path.join(output_dir, f"{run_name}_rollout.gif")
        with open(gif_path, "wb") as f:
            f.write(gif_bytes)

        model_path = os.path.join(output_dir, f"{run_name}_model.zip")
        model.save(model_path)
        with open(model_path, "rb") as f:
            model_bytes = f.read()

        config_path = os.path.join(output_dir, f"{run_name}_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_str)

        config_bytes = config_str.encode("utf-8")

        gif_hash = self.add_file(f"{run_name}_rollout.gif", gif_bytes)
        core.set_attribute(run_node, "gif_env", {
            "filename": f"{run_name}_rollout.gif",
            "hash": gif_hash,
        })

        model_hash = self.add_file(f"{run_name}_model.zip", model_bytes)
        core.set_attribute(run_node, "trained_model", {
            "filename": f"{run_name}_model.zip",
            "hash": model_hash,
        })

        config_hash = self.add_file(f"{run_name}_config.json", config_bytes)
        core.set_attribute(run_node, "config", {
            "filename": f"{run_name}_config.json",
            "hash": config_hash,
        })

