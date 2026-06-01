import numpy as np
from options_types import Option
from models import MODEL_PARAMETERS

def BlackScholes_price(option: Option, calc_delta: bool = False) -> float:
    return option.payoff_BlackScholes(calc_delta)

def binomial_beta_price(option: Option, model_name: str, m: int = 1, n: int = 0, i: int = 0, calc_delta: bool = False) -> float:
    return option.payoff_BetaBinomial(model_name, m, n, i, calc_delta)

def monte_carlo_price(option: Option, m: int, N: int, calc_delta: bool = False) -> float:
    np.random.seed(42)
    def _mc_simulate(s0_bump: float) -> float:
        """Auxiliary: simulate at bumped spot price"""
        dt = option.T / m
        dW = np.random.normal(scale=np.sqrt(dt), size=(N, m))
        W = np.cumsum(dW, axis=1)
        time_steps = np.broadcast_to(np.linspace(dt, option.T, m), (N, m))
        ST = s0_bump * np.exp((option.r - 0.5*option.vol**2)*time_steps + option.vol*W)
        # Use only final prices at maturity (terminal spot)
        ST_final = ST[:, -1]  # Shape: (N,)
        payoffs = option.payoff(s=ST_final)
        return np.mean(payoffs) * np.exp(-option.r * option.T)
    
    if not calc_delta:
        # Standard price calculation
        return _mc_simulate(option.s0)
    
    # Delta via central finite differences (bump-and-revalue)
    bump = 0.01 * option.s0  # 1% bump of spot
    price_up = _mc_simulate(option.s0 + bump)
    price_dn = _mc_simulate(option.s0 - bump)
    delta = (price_up - price_dn) / (2 * bump)
    return delta


def binomial_price(option: Option, model_name: str, m: int = 1, calc_delta: bool = False) -> float:
    def _binomial_core(s0_bump: float) -> float:
        """Auxiliary: compute binomial price at bumped spot"""
        model_key = model_name.lower()
        u, d, q = MODEL_PARAMETERS[model_key](s0_bump, option.K, option.T, option.r, option.vol, m)
        dt = option.T / m
        df = np.exp(-option.r * dt)
        i = np.arange(0, m + 1)
        ST = s0_bump * ((u/d)**i) * (d**m)
        payoffs = option.payoff(s=ST)
        for j in reversed(range(m)):
            payoffs = (q * payoffs[1:] + (1 - q) * payoffs[:-1]) * df
        return payoffs[0]
    
    if not calc_delta:
        # Standard price calculation
        return _binomial_core(option.s0)
    
    # Delta via central finite differences (bump-and-revalue)
    bump = 0.01 * option.s0  # 1% bump of spot
    price_up = _binomial_core(option.s0 + bump)
    price_dn = _binomial_core(option.s0 - bump)
    delta = (price_up - price_dn) / (2 * bump)
    return delta


def portfolio_monte_carlo_price(
    option: Option,
    S0: np.ndarray,
    correlation: np.ndarray,
    volatilities: np.ndarray,
    m: int = 1,
    N: int = 50000,
    calc_delta: bool = False
) -> float:
    """
    Monte Carlo pricing for portfolio options (basket, index, equal-weighted multi-asset).
    
    Parameters:
    -----------
    option : Option object
        The option to price (VanillaOption, DigitalOption, BarrierOption, etc.)
    
    S0 : ndarray or list
        Initial prices of each asset [S1_0, S2_0, ..., Sn_0]
    
    correlation : ndarray
        Correlation matrix (n x n), symmetric positive definite
    
    volatilities : ndarray or list
        Volatility vector [σ1, σ2, ..., σn]
    
    m : int, optional
        Number of time steps (default 1 for European portfolio options)
    
    N : int, optional
        Number of Monte Carlo simulations (default 50000)
    
    calc_delta : bool, optional
        Whether to calculate delta via bumping (default False)
    
    Returns:
    --------
    float : Option price or delta
    
    Notes:
    ------
    - Portfolio value computed as equal-weighted basket: V(t) = mean(S_i(t))
    - Correlated asset price paths using Cholesky decomposition
    - Payoff evaluated on portfolio value at maturity
    """
    np.random.seed(42)
    S0 = np.array(S0, dtype=float)
    volatilities = np.array(volatilities, dtype=float)
    correlation = np.array(correlation, dtype=float)
    n_assets = len(S0)
    # Build covariance matrix from correlation and volatilities
    cov_matrix = np.diag(volatilities) @ correlation @ np.diag(volatilities)
    
    def _portfolio_mc_simulate(s0_bump: np.ndarray) -> float:
        """Simulate correlated asset paths and compute portfolio option price"""
        dt = option.T / m
        # Cholesky decomposition for correlation structure
        L = np.linalg.cholesky(cov_matrix)
        # Generate correlated random normals (shape: m, n_assets, N)
        Z = np.random.normal(0, 1, size=(m, n_assets, N))
        W = np.einsum('ij,tjk->tik', L, Z) * np.sqrt(dt)
        # Compute log-returns with drift
        drift = (option.r - 0.5 * np.diag(cov_matrix)).reshape(n_assets, 1)
        log_drifts = drift * dt
        # Cumulative log-returns over all time steps
        cumulative_log_returns = np.cumsum(log_drifts[np.newaxis, :, :] + W, axis=0)
        # Terminal asset prices (shape: n_assets, N)
        S_terminal = s0_bump.reshape(n_assets, 1) * np.exp(cumulative_log_returns[-1, :, :])
        # Portfolio value: equal-weighted basket
        weights = np.ones(n_assets) / n_assets
        portfolio_values = weights @ S_terminal  # Shape: (N,)
        # Compute option payoffs on portfolio value
        payoffs = option.payoff(s=portfolio_values)
        # Discount and average
        return np.mean(payoffs) * np.exp(-option.r * option.T)
    if not calc_delta:
        # Standard price calculation
        return _portfolio_mc_simulate(S0)
    # Delta via central finite differences (bump-and-revalue all assets equally)
    bump = 0.01 * S0  # 1% bump of each spot price
    price_up = _portfolio_mc_simulate(S0 + bump)
    price_dn = _portfolio_mc_simulate(S0 - bump)
    delta = (price_up - price_dn) / (2 * np.mean(bump))
    return delta
