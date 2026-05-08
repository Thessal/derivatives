## NOTE: this model estimates delta directly, instead of trading amount. So, this is simplified version without trading cost consideration of Du(2020) 

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

    action_space = torch.arange(0, 101, 1).tolist()

class BlackScholesOracle:
    @staticmethod
    def call_price(S, K, tau, r, sigma):
        if tau <= 0:
            return torch.relu(S - K)
        # sigma = 0.01 (daily volatility )
        # tau = T / 252 (year)
        r = r / 252 # daily rate
        tau = tau * 252 
        d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * (tau)) / (sigma * math.sqrt(tau))
        d2 = d1 - sigma * math.sqrt(tau)
        N = torch.distributions.Normal(torch.tensor(0.0, device=Config.device), torch.tensor(1.0, device=Config.device))
        return S * N.cdf(d1) - K * math.exp(-r * tau) * N.cdf(d2)

    @staticmethod
    def call_delta(S, K, tau, r, sigma):
        if float(tau) <= 0:
            return torch.tensor(1.0 if float(S) > float(K) else 0.0, device=Config.device)
        r = r / 252 # daily rate
        tau = tau * 252 
        d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * (tau)) / (sigma * math.sqrt(tau))
        N = torch.distributions.Normal(torch.tensor(0.0, device=Config.device), torch.tensor(1.0, device=Config.device))
        return N.cdf(d1)

class OptionReplicationEnv:
    def __init__(self, S0=100.0, K=100.0, T=10.0/252.0, D=10, sigma=0.01, kappa=0.1, rf=0.00):
        self.init_cond = {
            "S_t": torch.tensor(S0, dtype=torch.float32, device=Config.device),
            "tau_t": torch.tensor(T, dtype=torch.float32, device=Config.device), 
            "n_t": torch.tensor(0.0, dtype=torch.float32, device=Config.device), 
            "K": torch.tensor(K, dtype=torch.float32, device=Config.device),
        }
        self.D = D                # Trades per day
        self.dt = torch.tensor(1.0 / D, dtype=torch.float32, device=Config.device)  # Unit is day

        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=Config.device)
        self.kappa = kappa
        self.r = torch.tensor(rf, dtype=torch.float32, device=Config.device)
        
        # Action space: target delta [0, 1, ..., 100]
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
        S_t, tau_t, n_t, K = Config.state_space_get (self)
        
        # Action directly dictates the new inventory (delta)
        n_new = torch.squeeze(self.action_space[action_idx])
        
        # Prevent absurd inventory levels just in case (though action space is 0-100)
        n_new = torch.clamp(n_new, min=0.0, max=100.0)
        
        # 2. Simulate GBM Transition
        dW = torch.squeeze(torch.randn(1, device=Config.device) * math.sqrt(self.dt))
        sigma = self.sigma 
        S_next = S_t * torch.exp((self.r - 0.5 * (sigma**2)) * self.dt + sigma * dW)
        tau_next = tau_t - self.dt / 252.0
        
        # 3. Mark to Market Option
        C_next = BlackScholesOracle.call_price(S_next, K, tau_next, self.r, sigma)
        
        # 4. Wealth Increment (Hedging a short call: long stock, short call)
        # q_t = PnL of stock - PnL of option
        q_t = n_new * (S_next - S_t) - 100 * (C_next - self.C_t)
        dw_t = q_t # No transaction cost
        
        # 5. Mean-Variance Reward
        reward = dw_t - (self.kappa / 2.0) * (dw_t**2)
        
        # Update State
        Config.state_space_set(
            self, 
            {
                "S_t": S_next, 
                "tau_t": tau_next,
                "n_t": n_new, 
                "K": self.init_cond["K"]
            }
        )
        self.C_t = C_next
        
        done = tau_next <= 0
        
        return self.state, reward, done


class dqn(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=64):
        super().__init__()
        self.gamma = 0.9 # 0.8 to 0.9 
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

        self.q_net_slow = nn.Sequential(*layers).to(Config.device) # target agent 
        self.target_agent_update_counter = 0
        self.target_agent_update_freq = 100

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

        return action.item() if action.dim() == 0 else action


    def td_loss(self, state, action, reward, next_state, done):

        self.target_agent_update_counter += 1
        if self.target_agent_update_counter % self.target_agent_update_freq == 0:
            self.q_net_slow.load_state_dict(self.q_net.state_dict())

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
            q_next_max = torch.max(self.q_net_slow.forward(next_state), dim=1).values
        q_target = reward + self.gamma * q_next_max * (1.0 - done)
        
        # loss = F.mse_loss(q, q_target)
        loss = F.huber_loss(q, q_target, delta=1.0) 
        return loss


class dqn_popart(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=64, beta=1e-4):
        super().__init__()
        self.gamma = 0.9
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

        # Target network components
        layers_slow = []
        in_dim = state_dim
        for _ in range(5):
            layers_slow.append(nn.Linear(in_dim, hidden_dim))
            layers_slow.append(nn.BatchNorm1d(hidden_dim))
            layers_slow.append(nn.ReLU())
            in_dim = hidden_dim
            
        self.feature_extractor_slow = nn.Sequential(*layers_slow).to(Config.device)
        self.final_layer_slow = nn.Linear(hidden_dim, self.action_dim).to(Config.device)
        
        self.feature_extractor_slow.load_state_dict(self.feature_extractor.state_dict())
        self.final_layer_slow.load_state_dict(self.final_layer.state_dict())
        
        self.target_agent_update_counter = 0
        self.target_agent_update_freq = 100

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

        self.target_agent_update_counter += 1
        if self.target_agent_update_counter % self.target_agent_update_freq == 0:
            self.feature_extractor_slow.load_state_dict(self.feature_extractor.state_dict())
            self.final_layer_slow.load_state_dict(self.final_layer.state_dict())

        # 1. Compute unnormalized target
        with torch.no_grad():
            features_next = self.feature_extractor_slow(next_state)
            q_next_norm = self.final_layer_slow(features_next)
            q_next_unnormalized = q_next_norm * self.sigma + self.mu
            
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

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim=4, action_dim=101, hidden_dim=128):
        super().__init__()
        
        # Separate Actor and Critic to prevent huge value loss from destroying policy
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # State normalization constants: [S, tau, n, K] ~ [100, 1, 100, 100]
        self.register_buffer('state_scale', torch.tensor([100.0, 1.0, 100.0, 100.0]))

    def forward(self, state):
        # Normalize state to help network converge
        norm_state = state / self.state_scale
        
        logits = self.actor(norm_state)
        value = self.critic(norm_state)
        return logits, value
        
    def get_action_and_value(self, state):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

if __name__=="__main__":
    cfg = dict(
        S0=100.0, K=100.0, T=10.0/252.0, D=5, sigma=0.01, kappa=0.1
    )

    env = OptionReplicationEnv(**cfg)
    
    # Initialize PPO Agent
    action_dim = len(Config.action_space)
    agent = PPOActorCritic(state_dim=4, action_dim=action_dim, hidden_dim=128).to(Config.device)
    try:
        agent.load_state_dict(torch.load('ppo_agent2_long.pth'))
    except:
        pass
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)

    # PPO hyperparameters
    num_epochs = 400
    num_episodes = 150
    ppo_epochs = 10
    mini_batch_size = 128
    gamma = 0.90
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")

        states = []
        actions = []
        logprobs = []
        rewards = []
        values = []
        dones = []

        # 1. Simulating the environment to gather Data
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                state_t = state.unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, value = agent.get_action_and_value(state_t)
                
                next_state, reward, done = env.step(action.item())
                
                states.append(state)
                actions.append(action.squeeze())
                logprobs.append(log_prob.squeeze())
                rewards.append(torch.tensor(reward, dtype=torch.float32, device=Config.device))
                values.append(value.squeeze())
                dones.append(torch.tensor(float(done), dtype=torch.float32, device=Config.device))
                
                state = next_state

        # Convert to tensors
        states_t = torch.stack(states)
        actions_t = torch.stack(actions)
        logprobs_t = torch.stack(logprobs)
        rewards_t = torch.stack(rewards)
        values_t = torch.stack(values)
        dones_t = torch.stack(dones)

        # Compute GAE and Returns // GAE: Generalized Advantage Estimation 
        with torch.no_grad():
            advantages = torch.zeros_like(rewards_t, device=Config.device)
            lastgaelam = 0
            for t in reversed(range(len(rewards_t))):
                if t == len(rewards_t) - 1:
                    nextnonterminal = 1.0 - dones_t[t]
                    nextvalues = 0.0
                else:
                    nextnonterminal = 1.0 - dones_t[t]
                    nextvalues = values_t[t + 1]
                
                delta = rewards_t[t] + gamma * nextvalues * nextnonterminal - values_t[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_t

        # 2. PPO Update
        b_inds = torch.arange(len(states_t))
        for ppo_epoch in range(ppo_epochs):
            b_inds = b_inds[torch.randperm(len(states_t))]
            for start in range(0, len(states_t), mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]

                mb_states = states_t[mb_inds]
                mb_actions = actions_t[mb_inds]
                mb_logprobs = logprobs_t[mb_inds]
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds]

                if len(mb_advantages) > 1:
                    mb_advantages_norm = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                else:
                    mb_advantages_norm = mb_advantages

                logits, v = agent(mb_states)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy()
                v = v.squeeze(-1)

                logratio = new_logprobs - mb_logprobs
                ratio = logratio.exp()
                
                pg_loss1 = -mb_advantages_norm * ratio
                pg_loss2 = -mb_advantages_norm * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = F.mse_loss(v, mb_returns)
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
                
        print(f"  PPO Epoch {ppo_epochs} Loss: {loss.item():.4f}")

    # Save the trained model weights
    torch.save(agent.state_dict(), 'ppo_agent2_long.pth')

    # ==========================================
    # 3. Evaluation & Comparison with Theoretical Delta
    # ==========================================
    print("\n--- Evaluation: Agent Target Delta vs Theoretical BS Delta ---")
    
    # We evaluate at tau = T/2
    eval_tau = (10.0 / 252.0) / 2.0
    eval_K = 100.0
    eval_sigma = 0.01
    eval_r = 0.0
    
    # Test across a range of stock prices
    test_prices = [90.0, 95.0, 98.0, 100.0, 102.0, 105.0, 110.0]
    
    print(f"{'Stock Price':>12} | {'Theoretical Delta':>18} | {'Agent Target Delta':>18}")
    print("-" * 55)
    
    agent.eval()
    for S_val in test_prices:
        # 1. Theoretical Delta (Hedging 100 options)
        S_tensor = torch.tensor(S_val, dtype=torch.float32, device=Config.device)
        K_tensor = torch.tensor(eval_K, dtype=torch.float32, device=Config.device)
        tau_tensor = torch.tensor(eval_tau, dtype=torch.float32, device=Config.device)
        r_tensor = torch.tensor(eval_r, dtype=torch.float32, device=Config.device)
        sigma_tensor = torch.tensor(eval_sigma, dtype=torch.float32, device=Config.device)
        
        bs_delta = BlackScholesOracle.call_delta(S_tensor, K_tensor, tau_tensor, r_tensor, sigma_tensor)
        theoretical_pos = bs_delta.item() * 100.0
        
        # 2. Agent Target Delta
        # State: [S_t, tau_t, n_t, K]
        # We start with n_t = 0
        S_val_2 = (S_val - 100) * 252 + 100
        state = torch.tensor([S_val_2, eval_tau, 0.0, eval_K], dtype=torch.float32, device=Config.device)
        state = state.unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = agent(state)
            import numpy as np 
            probs = np.exp(logits) / (np.exp(logits).sum())
            # print(probs * np.linspace(0,1,101) )
            print(100 * (probs * np.linspace(0,1,101) ).sum() )
 
            # We take the deterministic action (argmax) for the optimal delta hedge
            # action_idx = torch.argmax(logits, dim=-1).item()
            target_delta = 100 * (probs * np.linspace(0,1,101) ).sum() 
            
        # target_delta = Config.action_space[action_idx]
        agent_pos = target_delta
        
        print(f"{S_val:>12.2f} | {theoretical_pos:>18.2f} | {agent_pos:>18.2f}")
