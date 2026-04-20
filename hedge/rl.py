import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
import torch.nn.functional as F
import random
import torch.optim as optim

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_space = ["S_t", "tau_t", "n_t", "K"]
    state_space_get = lambda x: [x.S_t, x.tau_t, x.n_t, x.K]
    @staticmethod
    def state_space_set(x, y:dict) :
        x.S_t, x.tau_t, x.n_t, x.K = (y[k] for k in Config.state_space)

    action_space = [-10.0, -5.0, 0.0, 5.0, 10.0]

class BlackScholesOracle:
    @staticmethod
    def call_price(S, K, tau, r, sigma):
        if tau <= 0:
            return torch.relu(S - K)
        d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
        d2 = d1 - sigma * math.sqrt(tau)
        N = torch.distributions.Normal(torch.tensor(0.0, device=Config.device), torch.tensor(1.0, device=Config.device))
        return S * N.cdf(d1) - K * math.exp(-r * tau) * N.cdf(d2)

class OptionReplicationEnv:
    def __init__(self, S0=100.0, K=100.0, T=10.0/252.0, D=10, sigma=0.01, kappa=0.1, cost_multiplier=1.0, tick_size=0.1, rf=0.04):
        self.init_cond = {
            "S_t": torch.tensor(S0, dtype=torch.float32, device=Config.device),
            "tau_t": torch.tensor(T, dtype=torch.float32, device=Config.device), 
            "n_t": torch.tensor(0.0, dtype=torch.float32, device=Config.device), 
            "K": torch.tensor(K, dtype=torch.float32, device=Config.device),
        }
        # self.D = D                # Trades per day
        # D = 1
        self.dt = torch.tensor(1.0 / (252.0 * D), dtype=torch.float32, device=Config.device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=Config.device)
        self.kappa = kappa
        self.r = torch.tensor(rf, dtype=torch.float32, device=Config.device)
        
        # Frictions
        self.C = cost_multiplier
        self.tick_size = tick_size
        
        # Action space: trade [-10, -5, 0, 5, 10] shares
        self.action_space = torch.tensor(Config.action_space, dtype=torch.float32, device=Config.device)
        self.reset()

    @property
    def state(self):
        return torch.stack(Config.state_space_get(self))

    def reset(self):
        Config.state_space_set(
            self, {k:self.init_cond[k] for k in Config.state_space}
        )
        self.C_t = BlackScholesOracle.call_price(self.S_t, self.K, self.tau_t, self.r, self.sigma)
        return self.state

    def step(self, action_idx):
        trade_qty = torch.squeeze(self.action_space[action_idx])
        
        # 1. Execute Trade & Calculate Cost (Eq. 23)
        n_t = self.n_t + trade_qty
        cost = self.C * self.tick_size * (torch.abs(trade_qty) + 0.01 * trade_qty**2)
        
        # 2. Simulate GBM Transition
        dW = torch.squeeze(torch.randn(1, device=Config.device) * math.sqrt(self.dt))
        S_next = self.S_t * torch.exp((self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * dW)
        tau_next = self.tau_t - self.dt
        
        # 3. Mark to Market Option
        C_next = BlackScholesOracle.call_price(S_next, self.K, tau_next, self.r, self.sigma)
        
        # 4. Wealth Increment (Hedging a short call: long stock, short call)
        # q_t = PnL of stock - PnL of option
        q_t = n_t * (S_next - self.S_t) - 100 * (C_next - self.C_t)
        dw_t = q_t - cost
        
        # 5. Mean-Variance Reward (Eq. 22) # for every timestep, not terminal wealth!
        reward = dw_t - (self.kappa / 2.0) * (dw_t**2)
        
        # Update State
        Config.state_space_set(
            self, 
            {
                "S_t": S_next, 
                "tau_t": tau_next,
                "n_t": n_t, 
                "K": self.init_cond["K"]
            }
        )
        self.C_t = C_next
        
        done = tau_next <= 0
        
        return self.state, reward, done

# if __name__ == "__main__":
#     env = OptionReplicationEnv()
#     state = env.reset()
#     done = False
#     while not done:
#         action = torch.randint(0, len(Config.action_space), (1,))
#         state, reward, done = env.step(action)

class dqn(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=64):
        super().__init__()
        self.gamma = 0.99
        action_dim = len(Config.action_space)

        # Multilayer Perceptron with 5 hidden layers as per Du (2020)
        layers = []
        in_dim = state_dim
        for _ in range(5):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
            
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        # q_net is value function ( Policy = argmax Q )
        self.q_net = nn.Sequential(*layers).to(Config.device)

    def forward(self, state):
        q = self.q_net(state)
        return q
        
    def get_action(self, state, epsilon=0.1):
        # Epsilon-Greedy Exploration
        if random.random() < epsilon:
            return torch.randint(0, len(Config.action_space), (1,), device=Config.device).item()
            
        with torch.no_grad(): # Don't track gradients for action selection
            was_training = self.training
            self.eval() # Use moving averages for BN during single-step inference
            state_in = state.unsqueeze(0) if state.dim() == 1 else state
            q = self.forward(state_in)
            action = torch.argmax(q, dim=-1).squeeze(0) # Safe for both 
            if was_training:
                self.train()
            ## 3. Stabilizing the Moving Targets In DQN, your target value ($Q_{target}$) is generated by the network itself, meaning your targets are constantly moving. Because the scale of your rewards can also shift drastically depending on market conditions, Batch Norm ensures that the Q-network's internal representations don't destabilize as the TD-Error targets bounce around. (This is exactly the same flavor of instability that Pop-Art handles, but Pop-Art handles it at the output layer, while Batch Norm stabilizes the hidden layers!).

        return action.item() if action.dim() == 0 else action


    def td_loss(self, state, action, reward, next_state, done):
        # q = self.forward(state)[action]
        # if done: 
        #     q_target = reward
        # else:
        #     with torch.no_grad():
        #         q_next = self.forward(next_state)
        #     q_target = reward + self.gamma * torch.max(q_next)
        state = torch.as_tensor(state, device=Config.device, dtype=torch.float32)
        action = torch.as_tensor(action, device=Config.device, dtype=torch.long)
        reward = torch.as_tensor(reward, device=Config.device, dtype=torch.float32)
        next_state = torch.as_tensor(next_state, device=Config.device, dtype=torch.float32)
        done = torch.as_tensor(done, device=Config.device, dtype=torch.float32)

        # Modified code for batches
        q_vals = self.forward(state)
        if action.dim() == 1:
            action = action.unsqueeze(1)
        q = q_vals.gather(1, action).squeeze(1)
        with torch.no_grad(): # Don't track gradients for action selection
            q_next_max = torch.max(self.forward(next_state), dim=1).values
        q_target = reward + self.gamma * q_next_max * (1.0 - done)
        
        loss = F.mse_loss(q, q_target)
        return loss

class dqn_popart(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=64, beta=1e-4):
        super().__init__()
        self.gamma = 0.99
        self.action_dim = len(Config.action_space)

        # Shared feature extractor with 5 layers (Du 2020)
        layers = []
        in_dim = state_dim
        for _ in range(5):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers).to(Config.device)

        # Final linear layer for Q-values
        self.final_layer = nn.Linear(hidden_dim, self.action_dim).to(Config.device)

        # Pop-Art statistics
        self.register_buffer('mu', torch.zeros(1, device=Config.device))
        self.register_buffer('sigma', torch.ones(1, device=Config.device))
        self.register_buffer('nu', torch.zeros(1, device=Config.device)) # Second moment
        self.beta = beta

    def forward(self, state, return_normalized=False):
        features = self.feature_extractor(state)
        normalized_q = self.final_layer(features)
        
        if return_normalized:
            return normalized_q
        else:
            return normalized_q * self.sigma + self.mu
            
    def get_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return torch.randint(0, len(Config.action_space), (1,), device=Config.device).item()
            
        with torch.no_grad():
            was_training = self.training
            self.eval()
            state_in = state.unsqueeze(0) if state.dim() == 1 else state
            q = self.forward(state_in, return_normalized=False)
            action = torch.argmax(q, dim=-1).squeeze(0)
            if was_training:
                self.train()
        return action.item() if action.dim() == 0 else action

    def update_stats(self, target):
        with torch.no_grad():
            old_mu = self.mu.clone()
            old_sigma = self.sigma.clone()
            
            batch_mean = target.mean()
            batch_sq_mean = (target ** 2).mean()

            self.mu.data = (1 - self.beta) * self.mu + self.beta * batch_mean
            self.nu.data = (1 - self.beta) * self.nu + self.beta * batch_sq_mean
            
            # Sigma calculation with numerics stability
            self.sigma.data = torch.sqrt(torch.clamp(self.nu - self.mu**2, min=1e-8))

            # Pop-Art weights and biases modification (preserving outputs)
            self.final_layer.weight.data = self.final_layer.weight.data * (old_sigma / self.sigma)
            self.final_layer.bias.data = (self.final_layer.bias.data * old_sigma + old_mu - self.mu) / self.sigma

    def td_loss(self, state, action, reward, next_state, done):
        state = torch.as_tensor(state, device=Config.device, dtype=torch.float32)
        action = torch.as_tensor(action, device=Config.device, dtype=torch.long)
        reward = torch.as_tensor(reward, device=Config.device, dtype=torch.float32)
        next_state = torch.as_tensor(next_state, device=Config.device, dtype=torch.float32)
        done = torch.as_tensor(done, device=Config.device, dtype=torch.float32)

        # 1. Compute unnormalized target
        with torch.no_grad():
            q_next_unnormalized = self.forward(next_state, return_normalized=False)
            q_next_max = torch.max(q_next_unnormalized, dim=1).values
            unnormalized_target = reward + self.gamma * q_next_max * (1.0 - done)

        # 2. Update Pop-Art statistics based on targets
        self.update_stats(unnormalized_target)

        # 3. Compute NORMALIZED target for loss calculation
        with torch.no_grad():
            normalized_target = (unnormalized_target - self.mu) / self.sigma

        # 4. Compute loss on predictions mapped from current layers
        q_vals_norm = self.forward(state, return_normalized=True)
        if action.dim() == 1:
            action = action.unsqueeze(1)
        q = q_vals_norm.gather(1, action).squeeze(1)

        loss = F.mse_loss(q, normalized_target)
        return loss

# if __name__=="__main__":
#     cfg = dict(
#         S0=100.0, K=100.0, T=21.0/252.0, sigma=0.01, kappa=0.1, cost_multiplier=1.0, tick_size=0.1
#     )

#     env = OptionReplicationEnv(
#         **cfg
#     )
#     history = []
#     agent = dqn(state_dim = 4, hidden_dim =64)
#     optimizer = optim.Adam(agent.parameters(), lr= 0.001)

#     NUM_EPISODES = 1024

#     # ==========================================
#     # 1. Simulating the environment to gather Data
#     # ==========================================
#     for _ in range(NUM_EPISODES):
#         state = env.reset()
#         done = False
#         while not done:
#             action = agent.get_action(state, epsilon=0.1)
#             next_state, reward, done = env.step(action)
            
#             # IMPORTANT: Save 'next_state' and 'done' into history
#             # we use float() for reward and done to ensure they are python floats, not tensors
#             history.append((state, action, float(reward), next_state, float(done)))
#             state = next_state
            
#     # ==========================================
#     # 2. Ensemble Sampling (Batch Training)
#     # ==========================================
#     # Sample 10% from the collected trajectories
#     sample_size = int(len(history) * 0.1)
    
#     if len(history) >= sample_size and sample_size > 0:
#         # 1. Sample exactly `sample_size` transitions (breaks correlation)
#         batch = random.sample(history, sample_size)
        
#         # 2. Unpack the batch into separated lists using `zip(*batch)`
#         states, actions, rewards, next_states, dones = zip(*batch)
        
#         # 3. Stack lists into batched PyTorch Tensors
#         states_t      = torch.stack(states)                            # Shape: [sample_size, 4]
#         actions_t     = torch.tensor(actions, dtype=torch.long)        # Shape: [sample_size]
#         rewards_t     = torch.tensor(rewards, dtype=torch.float32)     # Shape: [sample_size]
#         next_states_t = torch.stack(next_states)                       # Shape: [sample_size, 4]
#         dones_t       = torch.tensor(dones, dtype=torch.float32)       # Shape: [sample_size]
        
#         # 4. Perform a SINGLE batched forward pass and loss calculation
#         loss = agent.td_loss(states_t, actions_t, rewards_t, next_states_t, dones_t)
        
#         # 5. One backprop step for the entire batch
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(f"Collected total {len(history)} transitions.")
#         print(f"Trained on {sample_size} samples (10%). Loss: {loss.item():.4f}")


class PPOActorCritic(nn.Module):
    def __init__(self, state_dim=4, action_dim=5, hidden_dim=64):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head: Policy (pi) - logits for categorical distribution
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic head: Value function (V)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
        
    def get_action_and_value(self, state):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)


