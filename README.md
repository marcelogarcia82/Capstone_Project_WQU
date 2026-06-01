# Capstone Project - WQU

## Overview

This capstone project implements and analyzes the **Garcia Binomial Model** for option pricing, comparing it with other established binomial models CRR and Monte Carlo simulation methods.

The project demonstrates that the Garcia model provides superior convergence properties for certain parameter regimes and maintains computational efficiency for complex derivative pricing scenarios, including barrier options, digital options, and multi-asset portfolio options.

## Key Features

### Option Pricing Models
- **Garcia Beta-Binomial Model**: Fast, accurate convergence with O(m) time complexity
- **Binomial Tree Models**: Traditional binomial tree implementation
- **Monte Carlo Simulation**: High-precision reference pricing method
- **Black-Scholes**: Analytical benchmark for European options

### Option Types Supported
- **Vanilla Options**: European call and put options
- **Digital Options**: Cash-or-nothing and asset-or-nothing options
- **Barrier Options**: Up-out, down-out, up-in, down-in configurations
- **Portfolio Options**: Multi-asset basket pricing with correlation structure

### Binomial Models Implemented
1. **Garcia** - Custom model optimized for faster convergence
2. **CRR** - Cox-Ross-Rubinstein
3. **Jarrow-Rudd** - JR model
4. **Tian** - Tian model (matches third moment)
5. **Trigeorgis** - Log-transformed approach
6. **JKY** - Jabbour-Kramin-Young model

## Project Structure

```
Capstone_Project_WQU/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── options_types.py                       # Option class definitions (Vanilla, Digital, Barrier)
├── models.py                              # Binomial model parameter calculations
├── methods.py                             # Pricing methods (Black-Scholes, Binomial, MC)
├── graphs.py                              # Plotting and visualization utilities
├── generate_dissertation_figures.py       # Main figure generation script
├── convergence_analysis.ipynb             # Convergence analysis notebook
├── sanity_check.ipynb                     # Sanity check and validation tests
├── Dissertation_Figures/                  # Generated figures (PNG, PDF)
└── Dissertation_Data/                     # Generated data (JSON, CSV)
```

## Installation

### Requirements
- Python 3.7+
- Dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:
```bash
git clone https://github.com/marcelogarcia82/Capstone_Project_WQU.git
cd Capstone_Project_WQU
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Figure Generation

Generate all dissertation figures and data:

```bash
python generate_dissertation_figures.py
```

This script generates:
- **Section 1**: Garcia Beta-Binomial vs Binomial Tree comparison
- **Section 2**: Garcia vs CRR convergence analysis
- **Section 3**: Delta convergence analysis
- **Section 4**: Computational performance comparison
- **Section 5**: Portfolio analysis (Beta-Binomial vs Monte Carlo)
- **Section 6**: Portfolio convergence figures with MC benchmark
- **Section 7**: Portfolio Delta analysis with MC benchmark
- **Section 8**: Portfolio timing vs number of MC paths
- **Section 9**: Portfolio timing vs number of assets

### Using in Python

#### Simple Pricing Example

```python
from options_types import VanillaOption, DigitalOption, BarrierOption
from methods import BlackScholes_price, binomial_beta_price

# Create a vanilla call option
option = VanillaOption(s0=100, r=0.05, T=1.0, K=100, vol=0.2, call=True)

# Price using Black-Scholes
bs_price = BlackScholes_price(option)
print(f"Black-Scholes Price: {bs_price:.6f}")

# Price using Garcia model with 100 steps
garcia_price = binomial_beta_price(option, model_name='garcia', m=100)
print(f"Garcia Price: {garcia_price:.6f}")

# Price using CRR model
crr_price = binomial_beta_price(option, model_name='crr', m=100)
print(f"CRR Price: {crr_price:.6f}")
```

#### Digital Option Pricing

```python
from options_types import DigitalOption
from methods import binomial_beta_price

# Asset-or-nothing digital call
digital = DigitalOption(s0=100, r=0.05, T=1.0, K=100, vol=0.2, 
                        call=True, digital_type='asset_or_nothing')

price = binomial_beta_price(digital, model_name='garcia', m=500)
print(f"Digital Price: {price:.6f}")
```

#### Barrier Option Pricing

```python
from options_types import BarrierOption
from methods import binomial_beta_price

# Up-and-out call
barrier = BarrierOption(s0=100, r=0.05, T=1.0, K=100, barrier=120, vol=0.2,
                       call=True, barrier_direction='up', barrier_type='out')

price = binomial_beta_price(barrier, model_name='garcia', m=500)
print(f"Barrier Price: {price:.6f}")
```

#### Portfolio Option Pricing

```python
from options_types import VanillaOption
from methods import portfolio_monte_carlo_price
import numpy as np

# Portfolio parameters
S0 = np.array([100, 120, 80, 95])
K = 90
T = 1
r = 0.05
volatilities = np.array([0.25, 0.35, 0.20, 0.40])
correlation = np.array([
    [1.0, 0.1, 0.4, 0.5],
    [0.1, 1.0, 0.3, 0.2],
    [0.4, 0.3, 1.0, 0.6],
    [0.5, 0.2, 0.6, 1.0]
])

# Create portfolio option
portfolio_option = VanillaOption(s0=np.mean(S0), r=r, T=T, K=K, vol=0.3, call=True)

# Price using Monte Carlo
price = portfolio_monte_carlo_price(
    option=portfolio_option,
    S0=S0,
    correlation=correlation,
    volatilities=volatilities,
    N=100000
)
print(f"Portfolio Price (MC): {price:.6f}")
```

## File Descriptions

### `options_types.py`
Defines option classes using OOP design:
- `Option` - Abstract base class
- `VanillaOption` - European call/put
- `DigitalOption` - Binary options (cash/asset-or-nothing)
- `BarrierOption` - Barrier options (knock-in/out)
- Test classes for validation

### `models.py`
Implements binomial model parameter calculations:
- `garcia_parameters()` - Garcia model
- `crr_parameters()` - Cox-Ross-Rubinstein
- `jarrow_rudd_parameters()` - Jarrow-Rudd
- `tian_parameters()` - Tian model
- `trigeorgis_parameters()` - Trigeorgis model
- `Jabbour_Kramin_Young_parameters()` - JKY model

### `methods.py`
Core pricing methods:
- `BlackScholes_price()` - Analytical solution
- `binomial_beta_price()` - Beta-binomial pricing (fast)
- `binomial_price()` - Tree-based pricing
- `monte_carlo_price()` - MC simulation
- `portfolio_monte_carlo_price()` - Multi-asset MC pricing

### `graphs.py`
Visualization utilities:
- `plot_convergence_to_bs()` - Single option convergence
- `plot_convergence_multiple()` - Multi-option subplots
- `plot_timing_comparison()` - Performance benchmarking
- `plot_beta_vs_mc_portfolio()` - Portfolio method comparison
- `plot_timing_beta_vs_mc_portfolio()` - Timing comparison

### `generate_dissertation_figures.py`
Main script generating all dissertation figures and data tables in high resolution (300 DPI).

## Output

The script generates:
- **Figures**: High-resolution PNG and PDF files in `Dissertation_Figures/`
- **Data**: JSON and CSV files in `Dissertation_Data/` for LaTeX integration

## Key Results

### Garcia Model Performance
✅ **Faster Convergence**: Garcia model shows superior convergence for certain parameter regimes  
✅ **Beta-Binomial Speed**: O(m) time complexity enables rapid pricing  
✅ **Accuracy**: Comparable or better error rates vs CRR and other models  
✅ **Portfolio Efficiency**: Maintains constant-time complexity for multi-asset options

## Technical Details

### Black-Scholes Benchmark
Used as reference for European option pricing. Prices computed using standard normal CDF:
- Call: S₀N(d₁) - K·e^(-rT)·N(d₂)
- Put: K·e^(-rT)·N(-d₂) - S₀·N(-d₁)

### Beta-Binomial Pricing
Uses incomplete beta function for fast computation:
- Digital option pricing via regularized incomplete beta functions
- Vanilla options decomposed into digital components
- Barrier options via payoff decomposition

### Monte Carlo Simulation
- Correlated asset paths via Cholesky decomposition
- Equal-weighted basket portfolio value
- Central finite differences for delta calculation

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{garcia2026capstone,
  author = {Garcia, Marcelo},
  title = {Exact Binomial Pricing via Exact Moment Matching and Regularized Beta Functions},
  school = {WorldQuant University},
  year = {2026}
}
```

## License

This project is part of a capstone thesis at WorldQuant University.

## Contact

For questions or inquiries, please contact: [marcelogarcia82@gmail.com]

---

**Last Updated**: June 2026  
**Project Status**: Complete
