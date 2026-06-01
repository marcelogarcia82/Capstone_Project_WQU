from abc import ABC, abstractmethod
from dataclasses import dataclass
from models import MODEL_PARAMETERS
from typing import Any
from scipy.stats import norm, binom
from scipy.special import betainc, betaincc
import numpy as np

class Option(ABC):
    """Type of Options"""
    @abstractmethod
    def payoff_BetaBinomial(self, model_name: str, m: int, n: int = 0, i: int = 0) -> float:
        ...
    @abstractmethod
    def payoff_BlackScholes(self) -> float:
        ...
    @abstractmethod
    def payoff(self):
        ...

@dataclass
class DigitalOption(Option):
    """
    Representation of a digital option
    """
    s0: float
    r: float
    T: float
    K: float
    vol: float
    call: bool = True
    digital_type: str = 'asset_or_nothing'

    def payoff_BetaBinomial(self, model_name: str, m: int, n: int = 0, i: int = 0, calc_delta = False) -> float:
        sgn = 1 if self.call else -1
        model_key = model_name.lower()
        u, d, q = MODEL_PARAMETERS[model_key](self.s0, self.K, self.T, self.r, self.vol, m)
        delta_tau = self.T / m
        gf = 1.0 + self.r * delta_tau
        df = 1/gf
        tilde_q = q * u * df
        alpha = np.floor((m / 2.0) + np.log(self.K / self.s0) / (2.0 * np.log(u)))
        S_n = self.s0 * (u ** i) * (d ** (n - i))
        a = int(alpha - i + 1)
        b = int(m - n - a + 1)
        if a <= 0 or b <= 0: return 0.0
        beta_q = betainc(a, b, q) if self.call else betaincc(a, b, q)
        beta_tilde_q = betainc(a, b, tilde_q) if self.call else betaincc(a, b, tilde_q)
        premium = S_n*beta_tilde_q if self.digital_type == 'asset_or_nothing' else beta_q*df ** (m-n)
        delta_cash = ((betainc(a-1, b, q) if self.call else betaincc(a-1, b, q))- (betainc(a, b-1, q) if self.call else betaincc(a, b-1, q)))/(S_n*(u-d)*gf**(m-n-1))
        #delta_cash = sgn*binom.pmf(a-1, a+b-2, q)/(S_n*(u-d)*gf**(m-n-1)))
        delta_asset = (u*(betainc(a-1, b, tilde_q) if self.call else betaincc(a-1, b, tilde_q))-d*(betainc(a, b-1, tilde_q) if self.call else betaincc(a, b-1, tilde_q)))/(u-d)
        #delta_asset = beta_tilde_q+sgn*binom.pmf(a-1, a+b-2, tilde_q)/((u-d)*gf) 
        delta = delta_asset if self.digital_type == 'asset_or_nothing' else delta_cash
        return delta if calc_delta else premium
    
    def payoff_BlackScholes(self, calc_delta = False) -> float:
        sgn = 1 if self.call else -1
        sigma = self.vol*np.sqrt(self.T)
        d1 = (np.log(self.s0/self.K) + self.r*self.T)/ sigma + 0.5*sigma
        d2 = d1 - self.vol*np.sqrt(self.T)
        premium = self.s0*norm.cdf(sgn*d1) if self.digital_type == 'asset_or_nothing' else np.exp(-self.r*self.T)*norm.cdf(sgn*d2)
        delta = norm.cdf(sgn*d1, 0, 1) + sgn*norm.pdf(sgn*d1, 0, 1) / sigma if self.digital_type == 'asset_or_nothing' \
                else np.exp(-self.r*self.T)*sgn*norm.pdf(d2, 0, 1) / (self.s0 * sigma)
        return delta if calc_delta else premium
    
    def payoff(self, s: np.ndarray) -> np.ndarray:
        sgn = 1 if self.call else -1
        condition = np.maximum(0, sgn*(s - self.K))
        payout = s if self.digital_type == 'asset_or_nothing' else 1
        payoff_digital = np.where(condition, payout, 0)
        return payoff_digital


@dataclass
class VanillaOption(Option):
    """
    Representation of a vanilla option
    """
    s0: float
    r: float
    T: float
    K: float
    vol: float
    call: bool = True
    
    def payoff_BetaBinomial(self, model_name: str, m: int, n: int = 0, i: int = 0, calc_delta = False) -> float:
        sgn = 1 if self.call else -1
        digital_asset = DigitalOption(self.s0, self.r, self.T, self.K, self.vol, self.call, 'asset_or_nothing').payoff_BetaBinomial(model_name, m, n, i, calc_delta)
        digital_cash = DigitalOption(self.s0, self.r, self.T, self.K, self.vol, self.call, 'cash_or_nothing').payoff_BetaBinomial(model_name, m, n, i, calc_delta)
        return sgn*(digital_asset - self.K * digital_cash) 
        
    def payoff_BlackScholes(self, calc_delta = False) -> float:
        sgn = 1 if self.call else -1
        digital_asset = DigitalOption(self.s0, self.r, self.T, self.K, self.vol, self.call, 'asset_or_nothing').payoff_BlackScholes(calc_delta)
        digital_cash = DigitalOption(self.s0, self.r, self.T, self.K, self.vol, self.call, 'cash_or_nothing').payoff_BlackScholes(calc_delta)
        return sgn*(digital_asset - self.K * digital_cash)
    
    def payoff(self, s: np.ndarray) -> np.ndarray:
        sgn = 1 if self.call else -1
        payoffs_vanilla = np.maximum(sgn*(s - self.K), 0)
        return payoffs_vanilla

@dataclass
class BarrierOption(Option):
    """
    Representation of a barrier option
    """
    s0: float
    r: float
    T: float
    K: float
    barrier: float
    vol: float
    call: bool = True
    barrier_direction: str = 'up'
    barrier_type: str = 'out'
    def payoff_BetaBinomial(self, model_name: str, m: int, n: int = 0, i: int = 0, calc_delta = False) -> float:
        sgn = 1 if self.call else -1
        assert sgn*(self.barrier-self.K)>0, "Barrier must be above strike for call and below strike for put"
        premium_vanilla_K = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, call=self.call).payoff_BetaBinomial(model_name, m, n, i, calc_delta)
        premium_vanilla_barrier = VanillaOption(self.s0, self.r, self.T, self.barrier, self.vol, call=self.call).payoff_BetaBinomial(model_name, m, n, i, calc_delta)
        premium_digital_cash = DigitalOption(self.s0, self.r, self.T, self.barrier, self.vol, self.call, 'cash_or_nothing').payoff_BetaBinomial(model_name, m, n, i, calc_delta)
        digital_1 = premium_vanilla_K - premium_vanilla_barrier - sgn * (self.barrier - self.K) * premium_digital_cash
        digital_2 = premium_vanilla_barrier + sgn * (self.barrier - self.K) * premium_digital_cash
        return digital_1 if \
                            ((self.call) and (self.barrier_direction == 'up') and (self.barrier_type == 'out'))\
                            or ((self.call) and (self.barrier_direction == 'down') and (self.barrier_type == 'in'))\
                            or ((not self.call) and (self.barrier_direction == 'down') and (self.barrier_type == 'out')) \
                            or ((not self.call) and (self.barrier_direction == 'up') and (self.barrier_type == 'in')) else digital_2
        
    def payoff_BlackScholes(self, calc_delta = False) -> float:
        sgn = 1 if self.call else -1
        assert sgn*(self.barrier-self.K)>0, "Barrier must be above strike for call and below strike for put"
        premium_vanilla_K = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, call=self.call).payoff_BlackScholes(calc_delta)
        premium_vanilla_barrier = VanillaOption(self.s0, self.r, self.T, self.barrier, self.vol, call=self.call).payoff_BlackScholes(calc_delta)
        premium_digital_cash = DigitalOption(self.s0, self.r, self.T, self.barrier, self.vol, self.call, 'cash_or_nothing').payoff_BlackScholes(calc_delta)
        digital_1 = premium_vanilla_K - premium_vanilla_barrier - sgn * (self.barrier - self.K) * premium_digital_cash
        digital_2 = premium_vanilla_barrier + sgn * (self.barrier - self.K) * premium_digital_cash
        return digital_1 if \
                            ((self.call) and (self.barrier_direction == 'up') and (self.barrier_type == 'out'))\
                            or ((self.call) and (self.barrier_direction == 'down') and (self.barrier_type == 'in'))\
                            or ((not self.call) and (self.barrier_direction == 'down') and (self.barrier_type == 'out')) \
                            or ((not self.call) and (self.barrier_direction == 'up') and (self.barrier_type == 'in')) else digital_2
    def payoff(self, s: np.ndarray) -> np.ndarray:
        sgn = 1 if self.call else -1
        assert sgn*(self.barrier-self.K)>0, "Barrier must be above strike for call and below strike for put"
        sgnb = 1 if self.barrier_direction == 'up' else -1
        payoffs_vanilla = np.maximum(sgn*(s - self.K), 0)
        condition = np.maximum(0, sgnb*(s - self.barrier))
        payout_true = payoffs_vanilla if self.barrier_type == 'in' else 0
        payout_false = 0 if self.barrier_type == 'in' else payoffs_vanilla
        payoffs_barrier = np.where(condition, payout_true, payout_false)
        return payoffs_barrier

@dataclass
class TestPutCallParity(Option):
    """
    Representation of a barrier option
    """
    s0: float
    r: float
    T: float
    K: float
    vol: float
    def payoff_BetaBinomial(self, model_name: str, m: int, n: int = 0, i: int = 0, calc_delta = False) -> float:
        premium_call = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, call=True).payoff_BetaBinomial(model_name, m, n, i, calc_delta)
        premium_put = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, call=False).payoff_BetaBinomial(model_name, m, n, i, calc_delta)
        return self.s0 + premium_put - premium_call - self.K*(1.0 + self.r * (self.T / m)) ** (-m+n)
        
    def payoff_BlackScholes(self, calc_delta = False) -> float:
        premium_call = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, call=True).payoff_BlackScholes(calc_delta)
        premium_put = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, call=False).payoff_BlackScholes(calc_delta)
        return self.s0 + premium_put-premium_call - self.K*np.exp(-self.r*self.T)
    
    def payoff(self, s: np.ndarray) -> np.ndarray:
        premium_call = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, call=True).payoff(s)
        premium_put = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, call=False).payoff(s)
        return s + premium_put - premium_call - self.K

@dataclass
class TestDigitals(Option):
    """
    Representation of a barrier option
    """
    s0: float
    r: float
    T: float
    K: float
    vol: float
    call: bool = True
    
    def payoff_BetaBinomial(self, model_name: str, m: int, n: int = 0, i: int = 0, calc_delta = False) -> float:
        sgn = 1 if self.call else -1
        premium_digital_asset = DigitalOption(self.s0, self.r, self.T, self.K, self.vol, self.call, 'asset_or_nothing').payoff_BetaBinomial(model_name, m, n, i, calc_delta)
        premium_digital_cash = DigitalOption(self.s0, self.r, self.T, self.K, self.vol, self.call, 'cash_or_nothing').payoff_BetaBinomial(model_name, m, n, i, calc_delta)
        premium_vanilla = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, self.call).payoff_BetaBinomial(model_name, m, n, i, calc_delta)
        return premium_vanilla - sgn*(premium_digital_asset - self.K*premium_digital_cash)    
    def payoff_BlackScholes(self, calc_delta = False) -> float:
        sgn = 1 if self.call else -1
        premium_digital_asset = DigitalOption(self.s0, self.r, self.T, self.K, self.vol, self.call, 'asset_or_nothing').payoff_BlackScholes(calc_delta)
        premium_digital_cash = DigitalOption(self.s0, self.r, self.T, self.K, self.vol, self.call, 'cash_or_nothing').payoff_BlackScholes(calc_delta)
        premium_vanilla = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, self.call).payoff_BlackScholes(calc_delta)
        return premium_vanilla - sgn*(premium_digital_asset - self.K*premium_digital_cash)

    def payoff(self, s: np.ndarray) -> np.ndarray:
        sgn = 1 if self.call else -1
        payoff_digital_asset = DigitalOption(self.s0, self.r, self.T, self.K, self.vol, self.call, 'asset_or_nothing').payoff(s)
        payoff_digital_cash = DigitalOption(self.s0, self.r, self.T, self.K, self.vol, self.call, 'cash_or_nothing').payoff(s)
        payoff_vanilla = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, call=self.call).payoff(s)
        return payoff_vanilla - sgn*(payoff_digital_asset - self.K*payoff_digital_cash)

@dataclass
class TestBarriers(Option):
    """
    Representation of a barrier option
    """
    s0: float
    r: float
    T: float
    K: float
    barrier: float
    vol: float
    call: bool = True
    def payoff_BetaBinomial(self, model_name: str, m: int, n: int = 0, i: int = 0, calc_delta = False) -> float:
        sgn = 1 if self.call else -1
        assert sgn*(self.barrier-self.K)>0, "Barrier must be above strike for call and below strike for put"
        premium_vanilla_strike = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, self.call).payoff_BetaBinomial(model_name, m, n, i, calc_delta)
        premium_vanilla_barrier = VanillaOption(self.s0, self.r, self.T, self.barrier, self.vol, self.call).payoff_BetaBinomial(model_name, m, n, i, calc_delta)
        premium_digital_cash = DigitalOption(self.s0, self.r, self.T, self.barrier, self.vol, self.call, 'cash_or_nothing').payoff_BetaBinomial(model_name, m, n, i, calc_delta)
        digital_out = premium_vanilla_strike - premium_vanilla_barrier - sgn * (self.barrier - self.K) * premium_digital_cash
        digital_in = premium_vanilla_strike - digital_out
        return digital_in + digital_out - premium_vanilla_strike    
    
    def payoff_BlackScholes(self, calc_delta = False) -> float:
        sgn = 1 if self.call else -1
        assert sgn*(self.barrier-self.K)>0, "Barrier must be above strike for call and below strike for put"
        premium_vanilla_strike = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, self.call).payoff_BlackScholes(calc_delta)
        premium_vanilla_barrier = VanillaOption(self.s0, self.r, self.T, self.barrier, self.vol, self.call).payoff_BlackScholes(calc_delta)
        premium_digital_cash = DigitalOption(self.s0, self.r, self.T, self.barrier, self.vol, self.call, 'cash_or_nothing').payoff_BlackScholes(calc_delta)
        digital_out = premium_vanilla_strike - premium_vanilla_barrier - sgn * (self.barrier - self.K) * premium_digital_cash
        digital_in = premium_vanilla_strike - digital_out
        return digital_in + digital_out - premium_vanilla_strike
    
    def payoff(self, s: np.ndarray) -> np.ndarray:
        sgn = 1 if self.call else -1
        assert sgn*(self.barrier-self.K)>0, "Barrier must be above strike for call and below strike for put"
        payoff_vanilla_strike = VanillaOption(self.s0, self.r, self.T, self.K, self.vol, self.call).payoff(s)
        payoff_vanilla_barrier = VanillaOption(self.s0, self.r, self.T, self.barrier, self.vol, self.call).payoff(s)
        payoff_digital_cash = DigitalOption(self.s0, self.r, self.T, self.barrier, self.vol, self.call, 'cash_or_nothing').payoff(s)
        digital_out = payoff_vanilla_strike - payoff_vanilla_barrier - sgn * (self.barrier - self.K) * payoff_digital_cash
        digital_in = payoff_vanilla_strike - digital_out
        return digital_in + digital_out - payoff_vanilla_strike