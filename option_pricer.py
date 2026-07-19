"""
Monte Carlo European Option Pricer
Prices a European call option by simulating geometric Brownian motion paths
under the risk-neutral measure, and validates against the analytical
Black-Scholes price.
"""

import numpy as np
from scipy.stats import norm


def monte_carlo_call(S0, K, r, sigma, T, M=1_000_000, N=252, seed=None):
    """
    Price a European call via Monte Carlo simulation of GBM paths.

    S0    : initial stock price
    K     : strike price
    r     : risk-free rate (risk-neutral drift)
    sigma : volatility
    T     : time to expiry (years)
    M     : number of simulated paths
    N     : number of time steps
    Returns: (price, standard_error)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    S = np.full(M, S0)
    for _ in range(N):
        Z = np.random.normal(size=M)
        S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    discounted_payoffs = np.exp(-r * T) * np.maximum(S - K, 0)
    price = discounted_payoffs.mean()
    std_error = discounted_payoffs.std() / np.sqrt(M)
    return price, std_error


def black_scholes_call(S0, K, r, sigma, T):
    """Analytical Black-Scholes price for a European call."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


if __name__ == "__main__":
    # Parameters
    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.2, 1.0

    mc_price, se = monte_carlo_call(S0, K, r, sigma, T, seed=42)
    bs_price = black_scholes_call(S0, K, r, sigma, T)

    print(f"Monte Carlo price:   {mc_price:.4f}")
    print(f"Standard error:      {se:.4f}")
    print(f"95% CI:              [{mc_price - 1.96*se:.4f}, {mc_price + 1.96*se:.4f}]")
    print(f"Black-Scholes price: {bs_price:.4f}")
    print(f"Difference:          {abs(mc_price - bs_price):.4f}")
