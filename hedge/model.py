import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from definitions import S0
import numpy as np 

# --- 1. Agent Architecture ---
class PPOAgent(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
            )
        
        # Head 1: To stop or not to stop? (2 actions: 0=Continue, 1=Stop)
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) 
        )
        
        # Head 2: If we continue, what hedge ratio?
        self.hedge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        features = self.shared(x)
        stop_logits = self.stop_head(features)
        hedge_logits = self.hedge_head(features)
        value = self.critic(features).squeeze(-1)
        
        return stop_logits, hedge_logits, value

def build_state(x, ttms):
    # For this simplified example, state is log(S) and ttm
    return torch.vstack([(x.ravel() - np.log(S0))*20, ttms.ravel()]).T

def explode_state(states, bs, sim_step):
    x = states[:,0] / 20 + np.log(S0)
    ttm = states[:,1]
    return x.reshape((sim_step-1,bs)), ttm.reshape((sim_step-1,bs))

def compute_rewards(agent, action_dim, bs, sim_step, E, dc, ds, curr_s, states, kappa=0.1, enable_stop=False):

    # Sample actions from current policy
    stop_logits, hedge_logits, values = agent(states)
    stop_dist = Categorical(logits=stop_logits)
    stop_action = stop_dist.sample()
    stop_log_prob = stop_dist.log_prob(stop_action)
    hedge_dist = Categorical(logits=hedge_logits)
    hedge_action = hedge_dist.sample()
    hedge_log_prob = hedge_dist.log_prob(hedge_action).reshape((sim_step-1,bs))
    
    stop_entropy = stop_dist.entropy().reshape((sim_step-1,bs))
    hedge_entropy = hedge_dist.entropy().reshape((sim_step-1,bs))

    # Reshape into time x trial 
    values = values.reshape((sim_step-1,bs))
    stop_action = stop_action.reshape((sim_step-1,bs))
    if not enable_stop:
        stop_action = torch.zeros_like(stop_action, device=stop_action.device)
    stop_log_prob = stop_log_prob.reshape((sim_step-1,bs))
    hedge_action = hedge_action.reshape((sim_step-1,bs)) * (1/(action_dim - 1))
    hedge_log_prob = hedge_log_prob.reshape((sim_step-1,bs))

    # Rewards
    stop_mask = (stop_action == 1)
    skip_mask = torch.roll(stop_mask, 1, dims=0)
    skip_mask[0] = 0
    skip_mask = torch.cummax(skip_mask, dim=0)[0] # after exercise
    stop_mask = stop_mask & (~skip_mask) # exercise action
    cont_mask = (~stop_mask) & (~skip_mask) # continue

    reward_continue = dc - hedge_action * ds
    exercise_payout = curr_s - E
    exercise_payout[exercise_payout<0] = 0
    previous_rewards_sum = (torch.cumsum(reward_continue, dim=0) - reward_continue)
    reward_stop = exercise_payout - previous_rewards_sum
    # Du(2020) seems to use return, but we don't include tcost... 
    # So we modify the target from more pnl, to perfect hedge
    # reward_continue = reward_continue - kappa * reward_continue * reward_continue # risk aversion
    reward_continue = - torch.abs(reward_continue)
    # reward_continue = reward_continue - 0.1 * (reward_continue - (r- q)) *2 # should we do this?
 
    rewards = torch.full((sim_step-1, bs), torch.nan)
    rewards[cont_mask] = reward_continue[cont_mask]
    rewards[stop_mask] = reward_stop[stop_mask]

    # skip_mask : true if already exercised and excluded from calculating loss
    return stop_action, stop_log_prob, hedge_action, hedge_log_prob, values, skip_mask, rewards, stop_entropy, hedge_entropy

def compute_returns(rewards, values, exercise_mask, gamma=0.8, lam = 0.97):
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    running_return = torch.zeros(rewards.shape[1], device=rewards.device)
    running_gae = torch.zeros(rewards.shape[1], device=rewards.device)
    
    for t in reversed(range(rewards.shape[0])):
        reward = torch.where(exercise_mask[t], 0, rewards[t])
        running_return = reward + gamma * running_return
        returns[t] = running_return
        if t == rewards.shape[0] - 1:
            next_value = 0
        else:
            next_value = torch.where(exercise_mask[t+1], 0, values[t+1])
            
        delta = torch.where(exercise_mask[t], 0, reward + gamma * next_value - values[t])
        running_gae = delta + gamma * lam * running_gae
        advantages[t] = running_gae
    returns = returns.detach()
    advantages = advantages.detach()
    return returns, advantages

def calc_policy_loss(stop_action, hedge_log_prob, stop_log_prob, exercise_mask, advantages, clip_ratio=0.2, c2=0.01, enable_stop=False):
    # merge two log probs, to calculate importance sampling term
    if enable_stop:
        log_prob = torch.where(stop_action==1, stop_log_prob, stop_log_prob + hedge_log_prob)
    else: 
        log_prob = hedge_log_prob
    old_log_prob = log_prob.detach()
    ratios = torch.exp(log_prob - old_log_prob)
    
    # Policy Loss (Clipped Surrogate)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.minimum(surr1, surr2)
    
    return policy_loss[~exercise_mask].ravel().mean() 

def calc_entropy_bonus(stop_action, stop_entropy, hedge_entropy, exercise_mask):
    # Entropy Bonus
    # entropy = torch.where(stop_action==1, stop_entropy, stop_entropy + hedge_entropy)
    # Do not promote stopping 
    entropy = hedge_entropy
    return entropy[~exercise_mask].ravel().mean() 

def calc_value_loss(values, returns, exercise_mask): 
    # return nn.functional.mse_loss(values[~exercise_mask], returns[~exercise_mask]) # reduction='mean'
    return nn.functional.l1_loss(values[~exercise_mask], returns[~exercise_mask]) # reduction='mean'
    # return nn.functional.huber_loss(values[~exercise_mask], returns[~exercise_mask], delta=1) # reduction='mean'
