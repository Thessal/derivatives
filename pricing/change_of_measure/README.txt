It is not easy to estimate expectaion value from as process with heavy tail, using Monte Carlo method.
(This problem is not limited to heavy tailed distribution. Look for variance reduction methods for Monte Carlo method)

RL based researches are often based on assumtions like: 
Expectation value exists, so the optimization have proper goal.
The continunous random process can be properly estimated using \Delta t. (So, daily rebalancing approximately works)

These are good abtraction, and it is linked to the reason why often financial simulation is done in 'risk neutral world'. 
For a hedger, change of measure is not a problem because the exposure (e.g. delta) is zero.
To simply put, E_P[\delta] = E_Q[\delta dP/dQ] = 0 when delta = 0.

So I guess PPO provides efficiently for delta hedging, relative to DQN.
But note that alpha seeker have to explot the systemic error induced by the method often used by other traders. 
