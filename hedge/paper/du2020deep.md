I. Core Reinforcement Learning Equations

1. Cumulative Expected Reward (Return):


$$G_t := \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

2. Action-Value Function (Q-function):


$$Q^\pi(s, a) := \mathbb{E}^\pi[G_t | s_t = s, a_t = a]$$

3. State-Value Function:


$$V^\pi(s) := \mathbb{E}^\pi[G_t | s_t = s] = \mathbb{E}^ \pi[Q^\pi(s, \pi(s))]$$

4. Bellman Equation for Optimal Q-Function:


$$Q^*(s, a) = \mathbb{E}\left[R_{t+1} + \gamma \max_{a'} Q^*(s_{t+1}, a') \Big| s_t = s, a_t = a\right]$$

II. Deep Q-Learning (DQN) & Pop-Art

5. DQN Loss Function:


$$L_i(\theta_i) = \mathbb{E}_{(s,a,R,s')\sim U(D)} \left[ L_\delta \left( R + \gamma \max_{a'} Q(s', a'; \theta_i^-) - Q(s, a; \theta_i) \right) \right]$$

6. Huber Loss:


$$L_\delta(x) = \begin{cases} \frac{1}{2}x^2 & \text{for } |x| \le \delta \\ \delta(|x| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

7. Pop-Art Modified Q-Function:


$$\tilde{Q}(s, a; \theta) = W Q(s, a; \theta) + b$$

8. Pop-Art Parameter Updates:


$$\Sigma_{new} \leftarrow \Sigma, \quad W_{new} \leftarrow \Sigma_{new}^{-1}\Sigma W, \quad b_{new} \leftarrow \Sigma_{new}^{-1}(\Sigma b + \mu - \mu_{new})$$

III. Proximal Policy Optimization (PPO)

9. Policy Gradient Estimator:


$$\nabla_\theta J(\theta) = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t \right]$$

10. PPO Loss Function:


$$L_{PPO}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]$$

11. Clipped Surrogate Objective:


$$L_t^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t, \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right)\hat{A}_t \right) \right]$$

IV. Option Replication & Hedging Framework

12. Agent State Space:


$$\mathcal{S} = \{(S_t, \tau, n_t, K) \mid S_t > 0, \tau > 0, n_t \in \mathbb{Z}, K \in \mathbb{R}_+\}$$

13. Agent Action Space:


$$\mathcal{A} = \{-100 \cdot L, \dots, 100 \cdot L\}$$

14. Mean-Variance Objective Function:


$$\max_\pi \mathbb{E}[w_T] - \frac{\kappa}{2} \text{Var}[w_T]$$

15. One-Period Reward Function:


$$R_t = \delta w_t - \frac{\kappa}{2}(\delta w_t)^2$$

16. Trading Cost Function:


$$\text{cost}(n) = C \cdot \text{TickSize} \cdot |n| + 0.01 n^2$$

17. Delta Hedging Baseline Policy:


$$n_t^{DH} = -100 \cdot \text{round}(\Delta(s_t))$$

V. Implementation Details

Environment & Simulation:

Built in OpenAI Gym using Python and PyTorch.

Price paths simulated via Geometric Brownian Motion (GBM).

One episode = 50 observations ($T \times D = 10 \text{ days} \times 5 \text{ trades/day}$).

Market Parameters:

$S_0 = 100$, daily lognormal volatility $\sigma = 0.01$.

Strike range $K \in \{98, 99, 100, 101, 102\}$.

TickSize = 0.1; cost multipliers $C \in \{0, 1, 3, 5\}$.

Network Architecture:

Multilayer Perceptron (MLP) with 5 hidden layers.

ReLU activation with Batch Normalization applied before each ReLU.

Optimization & Hyperparameters:

Adam optimizer for DQN/PPO, SGD for Pop-Art ($LR = 0.0001$).

Gradient clipping norm threshold = 1.

DQN discount factor $\gamma \in [0.8, 0.9]$.

PPO loss constants: $c_1 = 0.5, c_2 = 0.2$.