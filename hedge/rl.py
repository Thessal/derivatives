import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
import torch.nn.functional as F
import random
import torch.optim as optim

class Config:
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
        N = torch.distributions.Normal(0, 1)
        return S * N.cdf(d1) - K * math.exp(-r * tau) * N.cdf(d2)

class OptionReplicationEnv:
    def __init__(self, S0=100.0, K=100.0, T=10.0/252.0, D=10, sigma=0.01, kappa=0.1, cost_multiplier=1.0, tick_size=0.1, rf=0.04):
        self.init_cond = {
            "S_t": torch.tensor(S0),
            "tau_t": torch.tensor(T), 
            "n_t": torch.tensor(0.0), 
            "K": torch.tensor(K),
        }
        # self.D = D                # Trades per day
        # D = 1
        self.dt = torch.tensor(1.0 / (252.0 * D))
        self.sigma = torch.tensor(sigma)
        self.kappa = kappa
        self.r = torch.tensor(rf)
        
        # Frictions
        self.C = cost_multiplier
        self.tick_size = tick_size
        
        # Action space: trade [-10, -5, 0, 5, 10] shares
        self.action_space = torch.tensor(Config.action_space)
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
        dW = torch.squeeze(torch.randn(1) * math.sqrt(self.dt))
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

        # q_net is value function ( Policy = argmax Q ) 
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        q = self.q_net(state)
        return q
        
    def get_action(self, state, epsilon=0.1):
        # Epsilon-Greedy Exploration
        if random.random() < epsilon:
            return torch.randint(0, len(Config.action_space), (1,)).item()
            
        with torch.no_grad(): # Don't track gradients for action selection
            q = self.forward(state)
            action = torch.argmax(q, dim=-1) # Safe for both 

        return action


    def td_loss(self, state, action, reward, next_state, done):
        # q = self.forward(state)[action]
        # if done: 
        #     q_target = reward
        # else:
        #     with torch.no_grad():
        #         q_next = self.forward(next_state)
        #     q_target = reward + self.gamma * torch.max(q_next)

        # Modified code for batches
        q_vals = self.forward(state)
        if action.dim() == 1:
            action = action.unsqueeze(1)
        q = q_vals.gather(1, action.long()).squeeze(1)
        with torch.no_grad(): # Don't track gradients for action selection
            q_next_max = torch.max(self.forward(next_state), dim=1).values
        q_target = reward + self.gamma * q_next_max * (1.0 - done)
        
        loss = F.mse_loss(q, q_target)
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


