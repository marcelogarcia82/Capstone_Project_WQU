# ============================================================================
# BINOMIAL MODELS: Parameter Calculation Functions
# ============================================================================
# These functions compute u, d, q for different binomial models
# All functions return: (u, d, q) tuple
import numpy as np

def garcia_parameters(S0: float, K: float, T: float, r: float, sigma: float, m: int) -> tuple:
    """
    Garcia Model Parameters.    
    u = e^θ, d = e^(-θ)
    θ = arccosh(1 + ε)
    ε = (r²*δ_τ² + σ²*δ_τ) / [2*(1+r*δ_τ)]
    q = [(1+r*δ_τ) - d] / (u - d)
    """
    delta_tau = T / m
    epsilon = (r**2 * delta_tau**2 + sigma**2 * delta_tau) / (2.0 * (1.0 + r * delta_tau))
    theta = np.arccosh(1.0 + epsilon)    
    u = np.exp(theta)
    d = 1.0 / u
    gfactor = 1.0 + r * delta_tau    
    q = (gfactor - d) / (u - d)    
    return u, d, q

def crr_parameters(S0: float, K: float, T: float, r: float, sigma: float, m: int) -> tuple:
    """
    Cox-Ross-Rubinstein (CRR) Model Parameters (continuous compounding).    
    u = exp(σ*sqrt(δ_τ))
    d = 1/u
    q = (exp(r*δ_τ) - d) / (u - d)
    """
    delta_tau = T / m    
    u = np.exp(sigma * np.sqrt(delta_tau))
    d = 1.0 / u    
    nu = r - 0.5*sigma**2
    q = 1/2+nu*np.sqrt(delta_tau)/(2*sigma)    
    return u, d, q

def jarrow_rudd_parameters(S0: float, K: float, T: float, r: float, sigma: float, m: int) -> tuple:
    """
    Jarrow-Rudd Model Parameters.    
    u = exp((r - σ²/2)*δ_τ + σ*sqrt(δ_τ))
    d = exp((r - σ²/2)*δ_τ - σ*sqrt(δ_τ))
    q = 0.5 (martingale property)
    """
    delta_tau = T / m
    sqrt_delta = np.sqrt(delta_tau)
    nu = r - 0.5*sigma**2
    drift = nu * delta_tau    
    u = np.exp(drift + sigma * sqrt_delta)
    d = np.exp(drift - sigma * sqrt_delta)    
    q = 0.5    
    return u, d, q

def tian_parameters(S0: float, K: float, T: float, r: float, sigma: float, m: int) -> tuple:
    """
    Tian Model Parameters.    
    Designed to match the third moment of the continuous distribution.
    See: Tian, Y., "A Modified Lattice Approach To Option Pricing," JOD, 1999
    """
    delta_tau = T / m    
    r_ = np.exp(r * delta_tau)
    sigma_ = np.exp(sigma**2 * delta_tau)
    u = (r_*sigma_/2)*(sigma_ + 1 + np.sqrt(sigma_**2 + 2*sigma_ - 3))
    d = (r_*sigma_/2)*(sigma_ + 1 - np.sqrt(sigma_**2 + 2*sigma_ - 3))
    q = (r_ - d) / (u - d)
    return u, d, q

def trigeorgis_parameters(S0: float, K: float, T: float, r: float, sigma: float, m: int) -> tuple:
    """
    Trigeorgis Model Parameters.
    Modified version to match log-normal distribution accurately.
    See: Trigeorgis, L., "A Log-transformed Binomial Numerical Analysis Method"
    """
    delta_tau = T / m
    nu = r- 0.5 * sigma**2
    sqrt_term = np.sqrt(sigma**2 * delta_tau + (nu**2) * delta_tau**2)
    u = np.exp(sqrt_term)
    d = np.exp(-sqrt_term)
    q = 1/2 + nu * delta_tau / (2 * sqrt_term)
    return u, d, q

def Jabbour_Kramin_Young_parameters(S0: float, K: float, T: float, r: float, sigma: float, m: int) -> tuple:
    """
    Jabbour-Kramin-Young Model Parameters.
    A more recent model designed to improve convergence for certain option types.
    See: Jabbour, Kramin, Young (2023), "A New Binomial Tree Model for Option Pricing"
    """
    delta_tau = T / m
    sqrt_delta = np.sqrt(delta_tau)
    q = 0.5 + sigma*sqrt_delta/(2*np.sqrt(4+sigma**2*delta_tau))
    nu = r - 0.5*sigma**2
    sqrt_q = np.sqrt(q*(1-q))
    u = np.exp(nu*delta_tau + sigma*sqrt_delta*(1-q)/sqrt_q)
    d = np.exp(nu*delta_tau - sigma*sqrt_delta*q/sqrt_q)
    return u, d, q

# Dictionary for easy access
MODEL_PARAMETERS = {
    'garcia': garcia_parameters,
    'crr': crr_parameters,
    'jarrow_rudd': jarrow_rudd_parameters,
    'tian': tian_parameters,
    'trigeorgis': trigeorgis_parameters,
    'jky': Jabbour_Kramin_Young_parameters
}
