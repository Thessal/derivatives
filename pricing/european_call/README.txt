## On risk neutral pricing

GBM: 
d(log(S)) = \mu dt + \sigma dW
log(S(T)) = log(S(t)) + (\mu - \sigma^2/2)(T-t) + \sigma \sqrt(T-t) Y 
Where Y ~ N(0,1). Y is a random variable.
That means P[log(S(T)) = log(S(t)) + (\mu - \sigma^2/2)(T-t) + \sigma \sqrt(T-t) Y | t ] = \frac{\d N(x)}{\d x}

Drift here is risk free rate, because the risk and the premium is caused by option traders.
Stock traders provide liquidity for the informed traders and they are paid for the risk. 
Scenario:
Information arrives to the institutional trader at time t. They know E[S(t+dt) exp(-r dt)] >> S(t)
They could buy stock, but let's assume that they longs calls for leverage and risk management. They absorbs liquidity in option market.
Delta hedgers provide liquidity in option market and absorbs liquidity in stock market.
Delta hedgers are less informed than institutional traders, so E[C(T)|t] = E[C(T)|t-dt]
But delta hedgers now have short position in option so they buy stocks at time t.
Uninformed market maker in stock market often fails, even though there is reversion. They provided liquidity and not get paid because they bought expired information.
AS A RESULT, risk in stock market is generated and long-term passive stock holders suffer pain and get paid (averaged into equity risk premium).
To summarize, delta hedgers be risk neutral (\mu = r_f), but tipped by the institutional traders.
It explains why \mu prediction does not work well in stock market for retail traders.
On the other hand, volatility changing information is another story.

So, let's assuming \mu = r_f because we don't have information and we are risk neutral.
