# Monte Carlo European Option Pricer

Prices a European call option by simulating geometric Brownian motion (GBM)
under the risk-neutral measure, and validates the result against the
analytical Black-Scholes price.

## Overview

The price of a European call is the discounted expected payoff under the
risk-neutral measure. This project estimates that expectation two ways and
checks they agree:

1. **Monte Carlo simulation** — simulate many GBM price paths, compute the
   payoff `max(S_T - K, 0)` for each, average, and discount back to today.
2. **Black-Scholes formula** — the closed-form analytical price, used as a
   benchmark.

## The model

Under the risk-neutral measure the stock follows GBM, discretised as:

    S_{t+dt} = S_t * exp( (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z ),   Z ~ N(0,1)

The drift is the **risk-free rate `r`**, not the stock's real expected return —
required by no-arbitrage, since any other drift would allow a riskless profit
by replicating the option with stock and cash.

The call price is:

    C = exp(-r*T) * E[ max(S_T - K, 0) ]

## Validation

The Monte Carlo estimate is a sample mean, so it carries a standard error of
`std(payoffs) / sqrt(M)`. As the number of paths `M` increases, the estimate
converges to the Black-Scholes price and the standard error shrinks like
`1/sqrt(M)` (so 100x the paths gives ~10x the precision).

Example run (S0=100, K=105, r=0.05, sigma=0.2, T=1, M=1,000,000):

| Method        | Price  |
|---------------|--------|
| Monte Carlo   | 8.0285 |
| Black-Scholes | 8.0214 |

Agreement to within one cent, with the analytical price inside the 95%
confidence interval.

## Usage

    python option_pricer.py

## Requirements

    numpy
    scipy

## Possible extensions

- Variance reduction (antithetic variates, control variates) to cut the
  standard error without more paths
- Pricing put options and put-call parity checks
- The Greeks (delta, gamma, vega) via finite differences
- Path-dependent options (Asian, barrier) where no closed form exists
