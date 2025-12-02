import sys
import logging
from webgme_bindings import PluginBase

# Setup a logger
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
        
        # 1. Load all children of the active node (The Folder)
        children = core.load_children(active_node)
        
        # 2. Iterate through children to find 'Training_Run' nodes
        for child in children:
            meta_type = core.get_meta_type(child)
            meta_name = core.get_attribute(meta_type, 'name') if meta_type else ''
            
            if meta_name == 'Training_Run':
                run_name = core.get_attribute(child, 'name')
                logger.info(f'Processing Training Run: {run_name}')
                
                # --- A. Get Training Parameters ---
                run_params = {
                    'batch_size': core.get_attribute(child, 'batch_size'),
                    'timesteps': core.get_attribute(child, 'timesteps')
                }

                # --- B. Resolve Pointers (Agent & Environment) ---
                # We assume the pointers are named 'Agent' and 'Environment' based on the diagram connections
                agent_node = self.get_pointer_node(child, 'Agent')
                env_node = self.get_pointer_node(child, 'Environment')
                
                if not agent_node or not env_node:
                    logger.error(f"Training Run {run_name} is missing a pointer to Agent or Environment!")
                    continue
                
                # --- C. Get Agent Parameters ---
                agent_name = core.get_attribute(agent_node, 'name')
                agent_params = {
                    'learning_rate': core.get_attribute(agent_node, 'learning_rate'),
                    'discount_factor': core.get_attribute(agent_node, 'discount_factor'),
                    'epsilon_decay': core.get_attribute(agent_node, 'epsilon_decay'),
                    'initial_epsilon': core.get_attribute(agent_node, 'initial_epsilon'),
                    'final_epsilon': core.get_attribute(agent_node, 'final_epsilon'),
                    'policy_type': core.get_attribute(agent_node, 'policy')
                }
                
                # --- D. Get Architecture (Child of Agent) ---
                # We need to load children of the Agent to find the Architecture node
                agent_children = core.load_children(agent_node)
                arch_params = {'layer_size': 64, 'activation': 'ReLU'} # Defaults
                
                for grand_child in agent_children:
                    gc_meta = core.get_meta_type(grand_child)
                    if gc_meta and core.get_attribute(gc_meta, 'name') == 'Architecture':
                        # Note: accessing 'activation_funtion' exactly as spelled in diagram
                        arch_params['activation'] = core.get_attribute(grand_child, 'activation_funtion')
                        arch_params['layer_size'] = core.get_attribute(grand_child, 'layer_size')
                        break

                # --- E. Get Environment Parameters ---
                env_id = core.get_attribute(env_node, 'env_id')
                
                # --- F. Generate Code ---
                code = self.generate_script(run_name, env_id, run_params, agent_params, arch_params)
                self.add_file(f'{run_name}.py', code)

        commit_info = self.util.save(root_node, self.commit_hash, 'master', 'Generated RL Training Code')
        logger.info('Plugin completed.')

    def get_pointer_node(self, node, pointer_name):
        """Helper to load a node from a pointer path."""
        path = self.core.get_pointer_path(node, pointer_name)
        if path:
            return self.core.load_by_path(self.root_node, path)
        return None

    def generate_script(self, run_name, env_id, run_p, agent_p, arch_p):
        """Generates the Python string."""
        
        # Map string activation to PyTorch code
        act_map = {'ReLU': 'nn.ReLU()', 'Tanh': 'nn.Tanh()', 'Sigmoid': 'nn.Sigmoid()'}
        activation_code = act_map.get(arch_p['activation'], 'nn.ReLU()')
        
        return f"""import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# TRAINING RUN: {run_name}
# AGENT POLICY: {agent_p['policy_type']}
# ==========================================

# Hyperparameters
ENV_ID = "{env_id}"
BATCH_SIZE = {run_p['batch_size']}
TOTAL_TIMESTEPS = {run_p['timesteps']}

GAMMA = {agent_p['discount_factor']}
EPS_START = {agent_p['initial_epsilon']}
EPS_END = {agent_p['final_epsilon']}
EPS_DECAY = {agent_p['epsilon_decay']}
LR = {agent_p['learning_rate']}

LAYER_SIZE = {arch_p['layer_size']}

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Architecture defined in WebGME
        self.layer1 = nn.Linear(n_observations, LAYER_SIZE)
        self.layer2 = nn.Linear(LAYER_SIZE, LAYER_SIZE)
        self.layer3 = nn.Linear(LAYER_SIZE, n_actions)
        self.activation = {activation_code}

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.layer3(x)

def train():
    env = gym.make(ENV_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    steps_done = 0
    
    print(f"Starting training on {{ENV_ID}} for {{TOTAL_TIMESTEPS}} steps...")

    # Simplified Training Loop for Demonstration
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    for t in range(TOTAL_TIMESTEPS):
        # Select Action (Epsilon Greedy)
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \\
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

        # Step
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Move to next state
        state = next_state
        
        # (Optimization step omitted for brevity, but would use BATCH_SIZE here)
        
        if done:
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    print("Training Complete")

if __name__ == '__main__':
    train()
"""