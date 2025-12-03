import sys
import os
import tempfile
import logging
import io
import imageio.v2 as imageio   
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO, DQN, A2C
from webgme_bindings import PluginBase

logger = logging.getLogger('RLPythonCodeGenerator')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

            run_node = child  # keep reference so we can set attributes later
            run_name = core.get_attribute(child, 'name')
            logger.info(f'Processing Training Run: {run_name}')

            run_params = {
                'batch_size': core.get_attribute(child, 'batch_size'),
                'timesteps': core.get_attribute(child, 'timesteps'),
            }

            agent_node = self.get_pointer_node(child, 'Agent')
            env_node = self.get_pointer_node(child, 'Environment')

            if not agent_node or not env_node:
                logger.error(
                    f"Training Run {run_name} is missing a pointer to Agent or Environment!"
                )
                continue

            agent_params = {
                'algorithm': core.get_attribute(agent_node, 'algorithm'),
                'policy_type': core.get_attribute(agent_node, 'policy_type'),
                'learning_rate': core.get_attribute(agent_node, 'learning_rate'),
                'discount_factor': core.get_attribute(agent_node, 'discount_factor'),
            }

            agent_children = core.load_children(agent_node)
            arch_params = {
                'activation': 'ReLU',
                'layer_size': 64,
                'num_layers': 2,
                'arch_type': 'mlp',
            }

            for grand_child in agent_children:
                gc_meta = core.get_meta_type(grand_child)
                if gc_meta and core.get_attribute(gc_meta, 'name') == 'Architecture':
                    arch_params['activation'] = core.get_attribute(
                        grand_child, 'activation'
                    )
                    arch_params['layer_size'] = core.get_attribute(
                        grand_child, 'layer_size'
                    )
                    arch_params['num_layers'] = core.get_attribute(
                        grand_child, 'num_layers'
                    )
                    arch_params['arch_type'] = core.get_attribute(
                        grand_child, 'type'
                    )
                    break

            env_params = {
                'env_id': core.get_attribute(env_node, 'env_id'),
                'seed': core.get_attribute(env_node, 'seed'),
            }

            self.train_with_sb3(run_node, run_name, env_params, run_params, agent_params, arch_params)


        self.util.save(
            root_node, self.commit_hash, 'master', 'Ran RL training with SB3'
        )
        logger.info('Plugin completed.')

    def get_pointer_node(self, node, pointer_name):
        path = self.core.get_pointer_path(node, pointer_name)
        if path:
            return self.core.load_by_path(self.root_node, path)
        return None

    def train_with_sb3(self, run_node, run_name, env_p, run_p, agent_p, arch_p):
        core = self.core
        logger.info(f"Starting SB3 training for run '{run_name}'")

        algo_map = {"PPO": PPO, "DQN": DQN, "A2C": A2C}
        algo_cls = algo_map.get(agent_p['algorithm'])
        if algo_cls is None:
            logger.error(f"Unknown algorithm '{agent_p['algorithm']}'")
            return

        act_map = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}
        activation_fn = act_map.get(arch_p['activation'], nn.ReLU)
        net_arch = [arch_p['layer_size']] * max(1, int(arch_p['num_layers']))
        policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch)

        # 3) Training env (no rendering)
        train_env = gym.make(env_p['env_id'])
        if env_p['seed'] is not None:
            train_env.reset(seed=env_p['seed'])

        algo_kwargs = dict(
            learning_rate=agent_p['learning_rate'],
            gamma=agent_p['discount_factor'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="cpu",
        )
        if agent_p['algorithm'] in ("PPO", "DQN"):
            algo_kwargs["batch_size"] = run_p["batch_size"]

        model = algo_cls(agent_p["policy_type"], train_env, **algo_kwargs)

        logger.info(
            f"Training with timesteps={run_p['timesteps']}, "
            f"lr={agent_p['learning_rate']}, gamma={agent_p['discount_factor']}"
        )
        model.learn(total_timesteps=run_p["timesteps"])
        train_env.close()
        logger.info(f"Finished SB3 training for run '{run_name}'")

        tmp_model_path = os.path.join(
            tempfile.gettempdir(), f"{run_name}_model.zip"
        )
        model.save(tmp_model_path)
        with open(tmp_model_path, "rb") as f:
            model_bytes = f.read()
        model_hash = self.add_file(f"{run_name}_model.zip", model_bytes)
        core.set_attribute(run_node, "trained_model", model_hash)

        vis_env = gym.make(env_p["env_id"], render_mode="rgb_array")
        if env_p["seed"] is not None:
            vis_env.reset(seed=env_p["seed"])

        frames = []
        obs, _ = vis_env.reset()
        for _ in range(200):    
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = vis_env.step(action)
            frame = vis_env.render()        
            frames.append(frame)
            if done or truncated:
                obs, _ = vis_env.reset()

        vis_env.close()

        buf = io.BytesIO()
        imageio.mimsave(buf, frames, format="GIF", fps=30)
        gif_bytes = buf.getvalue()

        gif_hash = self.add_file(f"{run_name}_rollout.gif", gif_bytes)
        core.set_attribute(run_node, "gif_env", gif_hash)

        logger.info(
            f"Saved trained_model asset={model_hash} and gif_env asset={gif_hash} for '{run_name}'"
        )