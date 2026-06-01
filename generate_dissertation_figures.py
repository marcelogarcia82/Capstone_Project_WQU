
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
import time
from datetime import datetime

from options_types import VanillaOption, DigitalOption, BarrierOption
from methods import (
    BlackScholes_price,
    binomial_beta_price,
    monte_carlo_price,
    binomial_price,
    portfolio_monte_carlo_price
)
from models import MODEL_PARAMETERS
from graphs import plot_convergence_to_bs, plot_convergence_multiple, plot_timing_comparison, plot_beta_vs_mc_portfolio

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

# Create output directories
OUTPUT_DIR = Path("Dissertation_Figures")
DATA_DIR = Path("Dissertation_Data")
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# High-resolution settings for dissertation quality
DPI_DISSERTATION = 300
FIGSIZE_SINGLE = (12, 7)
FIGSIZE_DOUBLE = (16, 10)
FIGSIZE_QUAD = (16, 12)

# Color schemes
COLOR_GARCIA = '#1f77b4'      # Blue
COLOR_CRR = '#ff7f0e'          # Orange
COLOR_JR = '#2ca02c'           # Green
COLOR_TREE = '#d62728'         # Red
COLOR_MC = '#9467bd'           # Purple

# Professional color palettes
COLORS_GARCIA_BETA_TREE = [COLOR_GARCIA, COLOR_TREE]
COLORS_MODEL_COMPARISON = [COLOR_GARCIA, COLOR_CRR]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_figure(fig, filename, dpi=DPI_DISSERTATION, tight=True):
    """Save figure in multiple formats"""
    if tight:
        fig.tight_layout()
    filepath_png = OUTPUT_DIR / f"{filename}.png"
    filepath_pdf = OUTPUT_DIR / f"{filename}.pdf"
    fig.savefig(filepath_png, dpi=dpi, bbox_inches='tight', transparent=False)
    fig.savefig(filepath_pdf, dpi=dpi, bbox_inches='tight', transparent=False)
    print(f"✅ Saved: {filepath_png}")
    return filepath_png

def save_data_table(data_dict, filename):
    """Save data as JSON and CSV for LaTeX integration"""
    filepath_json = DATA_DIR / f"{filename}.json"
    filepath_csv = DATA_DIR / f"{filename}.csv"
    
    with open(filepath_json, 'w') as f:
        json.dump(data_dict, f, indent=2, default=str)
    
    if isinstance(data_dict, dict) and 'data' in data_dict:
        df = pd.DataFrame(data_dict['data'])
        df.to_csv(filepath_csv, index=False)
    
    print(f"✅ Saved data: {filepath_json}")
    return filepath_json

def compute_convergence_data(option, models_config, steps_range, calc_delta=False):
    """Compute convergence data for multiple models"""
    bs_price = BlackScholes_price(option, calc_delta=calc_delta)
    results = {}
    
    for cfg in models_config:
        errors = []
        prices = []
        for m in steps_range:
            if cfg['method'] == 'binomial_beta':
                price = binomial_beta_price(option, cfg['model_name'], m, calc_delta=calc_delta)
            elif cfg['method'] == 'binomial_tree':
                price = binomial_price(option, cfg['model_name'], m, calc_delta=calc_delta)
            prices.append(price)
            errors.append(abs(price - bs_price))
        results[cfg['name']] = {
            'prices': prices,
            'errors': errors,
            'bs_price': bs_price
        }
    
    return results, bs_price

# ============================================================================
# SECTION 1: GARCIA BETA-BINOMIAL vs BINOMIAL TREE
# ============================================================================

def generate_garcia_beta_tree_comparison():
    """
    Figure 1: Garcia Model - Beta-Binomial vs Binomial Tree Comparison
    Demonstrates that beta-binomial method produces same results as tree but faster
    """
    print("\n" + "="*70)
    print("SECTION 1: GARCIA BETA-BINOMIAL vs BINOMIAL TREE COMPARISON")
    print("="*70)
    
    K = 100
    s0 = 0.8 * K
    r = 0.15
    T = 1
    vol = 0.10
    steps_range = np.arange(1, 101, 1)
    
    # Digital Options
    digital_cash_call = DigitalOption(s0=s0, r=r, T=T, K=K, vol=vol, call=True, digital_type='cash_or_nothing')
    digital_cash_put = DigitalOption(s0=s0, r=r, T=T, K=K, vol=vol, call=False, digital_type='cash_or_nothing')
    digital_asset_call = DigitalOption(s0=s0, r=r, T=T, K=K, vol=vol, call=True, digital_type='asset_or_nothing')
    digital_asset_put = DigitalOption(s0=s0, r=r, T=T, K=K, vol=vol, call=False, digital_type='asset_or_nothing')
    
    # Vanilla Options
    vanilla_call = VanillaOption(s0=s0, r=r, T=T, K=K, vol=vol, call=True)
    vanilla_put = VanillaOption(s0=s0, r=r, T=T, K=K, vol=vol, call=False)
    
    # Barrier Options
    barrier_call_uo = BarrierOption(s0=s0, r=r, T=T, K=K, barrier=K+10, vol=vol, call=True, barrier_direction='up', barrier_type='out')
    barrier_put_do = BarrierOption(s0=s0, r=r, T=T, K=K, barrier=K-10, vol=vol, call=False, barrier_direction='down', barrier_type='out')
    
    options_list = [
        (digital_cash_call, "Digital Cash Call"),
        (digital_cash_put, "Digital Cash Put"),
        (digital_asset_call, "Digital Asset Call"),
        (digital_asset_put, "Digital Asset Put"),
        (vanilla_call, "Vanilla Call"),
        (vanilla_put, "Vanilla Put"),
        (barrier_call_uo, "Barrier Call Up-Out"),
        (barrier_put_do, "Barrier Put Down-Out")
    ]
    
    models_config = [
        {'name': 'GARCIA (Beta)', 'method': 'binomial_beta', 'model_name': 'garcia'},
        {'name': 'GARCIA (Tree)', 'method': 'binomial_tree', 'model_name': 'garcia'},
    ]
    
    results_summary = {
        'section': 'Garcia Beta-Binomial vs Tree',
        'date': datetime.now().isoformat(),
        'parameters': {
            'S0': float(s0),
            'K': float(K),
            'r': float(r),
            'T': float(T),
            'vol': float(vol),
            'steps_tested': list(steps_range)
        },
        'options': []
    }
    
    for option, option_name in options_list:
        print(f"\n📊 Processing: {option_name}")
        results, bs_price = compute_convergence_data(option, models_config, steps_range, calc_delta=False)
        
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=100)
        
        for cfg, color in zip(models_config, COLORS_GARCIA_BETA_TREE):
            model_name = cfg['name']
            errors = results[model_name]['errors']
            ax.plot(steps_range, errors, linewidth=2.5, marker='o', markersize=4, 
                   label=model_name, color=color, alpha=0.8)
        
        ax.set_xlabel('Number of Steps (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Absolute Pricing Error', fontsize=12, fontweight='bold')
        ax.set_title(f'{option_name}: Beta-Binomial vs Binomial Tree (Garcia)', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_yscale('log')
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        filename = f"01_Garcia_BetaTree_{option_name.replace(' ', '_')}"
        save_figure(fig, filename)
        plt.close(fig)
        
        # Store summary data
        results_summary['options'].append({
            'name': option_name,
            'bs_price': float(results[model_name]['bs_price']),
            'max_error_beta': float(np.max(results['GARCIA (Beta)']['errors'])),
            'max_error_tree': float(np.max(results['GARCIA (Tree)']['errors'])),
            'min_error_beta': float(np.min(results['GARCIA (Beta)']['errors'])),
            'min_error_tree': float(np.min(results['GARCIA (Tree)']['errors'])),
        })
    
    save_data_table(results_summary, "01_Garcia_BetaTree_Comparison")
    print(f"\n✅ Completed Section 1: {len(options_list)} figures generated")

# ============================================================================
# SECTION 2: GARCIA vs CRR - CONVERGENCE ANALYSIS
# ============================================================================

def generate_garcia_crr_convergence():
    """
    Figure 2: Convergence Comparison - Garcia vs CRR to Black-Scholes
    Highlights faster convergence of Garcia model in certain parameter regimes
    """
    print("\n" + "="*70)
    print("SECTION 2: GARCIA vs CRR CONVERGENCE ANALYSIS")
    print("="*70)
    
    K = 100
    s0 = 0.8 * K
    r = 0.15
    T = 1
    vol = 0.10
    steps_range = np.arange(50, 201, 1)
    
    # Options
    digital_asset_call = DigitalOption(s0=s0, r=r, T=T, K=K, vol=vol, call=True, digital_type='asset_or_nothing')
    digital_asset_put = DigitalOption(s0=s0, r=r, T=T, K=K, vol=vol, call=False, digital_type='asset_or_nothing')
    vanilla_call = VanillaOption(s0=s0, r=r, T=T, K=K, vol=vol, call=True)
    vanilla_put = VanillaOption(s0=s0, r=r, T=T, K=K, vol=vol, call=False)
    
    options_list = [
        (digital_asset_call, "Digital Asset Call"),
        (digital_asset_put, "Digital Asset Put"),
        (vanilla_call, "Vanilla Call"),
        (vanilla_put, "Vanilla Put")
    ]
    
    models_config = [
        {'name': 'GARCIA (Beta)', 'method': 'binomial_beta', 'model_name': 'garcia'},
        {'name': 'CRR (Beta)', 'method': 'binomial_beta', 'model_name': 'crr'},
    ]
    
    results_summary = {
        'section': 'Garcia vs CRR Convergence',
        'date': datetime.now().isoformat(),
        'parameters': {
            'S0': float(s0),
            'K': float(K),
            'r': float(r),
            'T': float(T),
            'vol': float(vol),
            'steps_range': [int(s) for s in steps_range]
        },
        'convergence_comparison': []
    }
    
    for option, option_name in options_list:
        print(f"\n📊 Processing: {option_name}")
        results, bs_price = compute_convergence_data(option, models_config, steps_range, calc_delta=False)
        
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=100)
        
        for cfg, color in zip(models_config, COLORS_MODEL_COMPARISON):
            model_name = cfg['name']
            errors = results[model_name]['errors']
            ax.plot(steps_range, errors, linewidth=2.5, marker='o', markersize=4,
                   label=model_name, color=color, alpha=0.8)
        
        ax.set_xlabel('Number of Steps (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Absolute Error to Black-Scholes', fontsize=12, fontweight='bold')
        ax.set_title(f'{option_name}: Garcia vs CRR Convergence', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_yscale('log')
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        filename = f"02_Convergence_GarciaVsCRR_{option_name.replace(' ', '_')}"
        save_figure(fig, filename)
        plt.close(fig)
        
        # Compute convergence rates at different step points
        garcia_errors = results['GARCIA (Beta)']['errors']
        crr_errors = results['CRR (Beta)']['errors']
        
        convergence_data = {
            'option': option_name,
            'bs_price': float(bs_price),
            'garcia_error_at_50': float(garcia_errors[0]),
            'garcia_error_at_150': float(garcia_errors[100]) if len(garcia_errors) > 100 else None,
            'crr_error_at_50': float(crr_errors[0]),
            'crr_error_at_150': float(crr_errors[100]) if len(crr_errors) > 100 else None,
        }
        
        # Calculate which converges faster at step 100
        if len(garcia_errors) > 50 and len(crr_errors) > 50:
            garcia_100 = garcia_errors[50]
            crr_100 = crr_errors[50]
            convergence_data['faster_at_100'] = 'GARCIA' if garcia_100 < crr_100 else 'CRR'
            convergence_data['error_ratio_100'] = float(garcia_100 / crr_100)
        
        results_summary['convergence_comparison'].append(convergence_data)
    
    save_data_table(results_summary, "02_Garcia_vs_CRR_Convergence")
    print(f"\n✅ Completed Section 2: {len(options_list)} convergence figures generated")

# ============================================================================
# SECTION 3: GREEK ANALYSIS (DELTA)
# ============================================================================

def generate_delta_analysis():
    """
    Figure 3: Delta Convergence Analysis for Garcia vs CRR
    """
    print("\n" + "="*70)
    print("SECTION 3: GREEK ANALYSIS - DELTA")
    print("="*70)
    
    K = 100
    s0 = 0.8 * K
    r = 0.15
    T = 1
    vol = 0.10
    steps_range = np.arange(50, 201, 1)
    
    # Options for Delta analysis
    digital_call = DigitalOption(s0=s0, r=r, T=T, K=K, vol=vol, call=True, digital_type='asset_or_nothing')
    vanilla_call = VanillaOption(s0=s0, r=r, T=T, K=K, vol=vol, call=True)
    
    options_list = [
        (digital_call, "Digital Call Delta"),
        (vanilla_call, "Vanilla Call Delta")
    ]
    
    models_config = [
        {'name': 'GARCIA (Beta)', 'method': 'binomial_beta', 'model_name': 'garcia'},
        {'name': 'CRR (Beta)', 'method': 'binomial_beta', 'model_name': 'crr'},
    ]
    
    for option, option_name in options_list:
        print(f"\n📊 Processing: {option_name}")
        results, bs_delta = compute_convergence_data(option, models_config, steps_range, calc_delta=True)
        
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=100)
        
        for cfg, color in zip(models_config, COLORS_MODEL_COMPARISON):
            model_name = cfg['name']
            errors = results[model_name]['errors']
            ax.plot(steps_range, errors, linewidth=2.5, marker='o', markersize=4,
                   label=model_name, color=color, alpha=0.8)
        
        ax.set_xlabel('Number of Steps (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Absolute Delta Error', fontsize=12, fontweight='bold')
        ax.set_title(f'{option_name}: Garcia vs CRR', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_yscale('log')
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        filename = f"03_Delta_{option_name.replace(' ', '_')}"
        save_figure(fig, filename)
        plt.close(fig)
    
    print(f"\n✅ Completed Section 3: {len(options_list)} delta figures generated")

# ============================================================================
# SECTION 4: PERFORMANCE ANALYSIS
# ============================================================================

def generate_performance_analysis():
    """
    Figure 4: Computational Performance - Beta-Binomial vs Binomial Tree
    """
    print("\n" + "="*70)
    print("SECTION 4: PERFORMANCE ANALYSIS")
    print("="*70)
    
    vanilla_call = VanillaOption(s0=80, r=0.15, T=1.0, K=100, vol=0.10, call=True)
    
    methods_timing = [
        {'name': 'Beta-Binomial', 'method': 'binomial_beta', 'model_name': 'garcia'},
        {'name': 'Binomial Tree', 'method': 'binomial_tree', 'model_name': 'garcia'}
    ]
    
    print("\n⏱️  Running timing comparison (this may take a moment)...")
    fig, ax, timing_results = plot_timing_comparison(
        vanilla_call,
        methods_timing,
        steps_range=np.arange(100, 1001, 100),
        scale='linear',
        title='Vanilla Call: Beta-Binomial vs Binomial Tree Performance',
        benchmark_model='garcia',
        num_trials=3
    )
    
    filename = "04_Performance_Timing_Comparison"
    save_figure(fig, filename)
    plt.close(fig)
    
    # Save timing data
    timing_data = {
        'section': 'Performance Analysis',
        'date': datetime.now().isoformat(),
        'option': 'Vanilla Call (S0=80, K=100, T=1, r=0.15, vol=0.10)',
        'results': timing_results
    }
    save_data_table(timing_data, "04_Performance_Data")
    
    print(f"✅ Completed Section 4: Performance analysis generated")

# ============================================================================
# SECTION 5: PORTFOLIO ANALYSIS
# ============================================================================

def generate_portfolio_analysis():
    """
    Figure 5: Portfolio Analysis - Beta-Binomial vs Monte Carlo
    """
    print("\n" + "="*70)
    print("SECTION 5: PORTFOLIO ANALYSIS - BETA-BINOMIAL vs MONTE CARLO")
    print("="*70)
    
    # Portfolio parameters
    S0_portfolio = np.array([100, 120, 80, 95])
    K = 90
    B = 150
    T = 1
    r = 0.05
    volatilities = np.array([0.25, 0.35, 0.20, 0.40])
    
    correlation = np.array([
        [1.0, 0.1, 0.4, 0.5],
        [0.1, 1.0, 0.3, 0.2],
        [0.4, 0.3, 1.0, 0.6],
        [0.5, 0.2, 0.6, 1.0]
    ])
    
    # Calculate portfolio volatility
    weights = np.ones(len(volatilities)) / len(volatilities)
    cov_matrix = np.diag(volatilities) @ correlation @ np.diag(volatilities)
    var_portfolio = weights @ cov_matrix @ weights
    vol_portfolio = np.sqrt(var_portfolio)
    s0_portfolio = np.mean(S0_portfolio)
    
    print(f"\n📊 Portfolio Parameters:")
    print(f"   S0 (individual assets): {S0_portfolio}")
    print(f"   Portfolio S0 (equal-weighted): {s0_portfolio:.2f}")
    print(f"   Portfolio Volatility: {vol_portfolio:.4f}")
    print(f"   Strike: {K}, Barrier: {B}")
    
    # Create portfolio options
    portfolio_call = VanillaOption(s0=s0_portfolio, r=r, T=T, K=K, vol=vol_portfolio, call=True)
    portfolio_digital = DigitalOption(s0=s0_portfolio, r=r, T=T, K=K, vol=vol_portfolio, 
                                     call=True, digital_type='asset_or_nothing')
    portfolio_barrier = BarrierOption(s0=s0_portfolio, r=r, T=T, K=K, barrier=B, vol=vol_portfolio, 
                                     call=True, barrier_direction='up', barrier_type='out')
    
    options_list = [portfolio_call, portfolio_digital, portfolio_barrier]
    option_names = ['Vanilla Call', 'Digital Asset Call', 'Barrier Call Up-Out']
    
    results_summary = {
        'section': 'Portfolio Analysis',
        'date': datetime.now().isoformat(),
        'portfolio_params': {
            'num_assets': len(S0_portfolio),
            'asset_prices': S0_portfolio.tolist(),
            'portfolio_s0': float(s0_portfolio),
            'portfolio_vol': float(vol_portfolio),
            'strike': float(K),
            'barrier': float(B),
            'rate': float(r),
            'time': float(T),
            'volatilities': volatilities.tolist(),
            'correlation': correlation.tolist()
        },
        'option_comparison': []
    }
    
    for option, option_name in zip(options_list, option_names):
        print(f"\n📊 Processing: {option_name}")
        
        # Black-Scholes benchmark
        bs_price = BlackScholes_price(option, calc_delta=False)
        
        # Monte Carlo (using large number of paths for accuracy)
        mc_price = portfolio_monte_carlo_price(
            option=option,
            S0=S0_portfolio,
            correlation=correlation,
            volatilities=volatilities,
            m=1,
            N=1_000_000,
            calc_delta=False
        )
        
        # Beta-Binomial with very large m for high precision
        beta_price = binomial_beta_price(option, model_name='garcia', m=1_000_000, calc_delta=False)
        
        print(f"   Black-Scholes: {bs_price:.6f}")
        print(f"   Monte Carlo (1M paths): {mc_price:.6f}")
        print(f"   Beta-Binomial (100k steps): {beta_price:.6f}")
        print(f"   MC-to-Beta error: {abs(mc_price - beta_price):.6e}")
        
        option_comparison = {
            'option': option_name,
            'bs_price': float(bs_price),
            'mc_price': float(mc_price),
            'beta_price': float(beta_price),
            'mc_bs_error': float(abs(mc_price - bs_price)),
            'beta_bs_error': float(abs(beta_price - bs_price)),
            'beta_mc_error': float(abs(beta_price - mc_price))
        }
        results_summary['option_comparison'].append(option_comparison)
    
    save_data_table(results_summary, "05_Portfolio_Analysis")
    print(f"\n✅ Completed Section 5: Portfolio analysis completed")

# ============================================================================
# SECTION 6: PORTFOLIO FIGURES - BETA vs MONTE CARLO
# ============================================================================

def generate_portfolio_figures():
    """
    Figures 5a-5c: Portfolio Analysis - Beta-Binomial vs Monte Carlo Convergence
    Using CORRECT portfolio volatility calculation:
    weights = np.array(len(volatilities)*[1/len(volatilities)])
    S0_portfolio = np.dot(weights, SM)
    cov_matrix = np.diag(volatilities)@correlation@np.diag(volatilities)
    var_b = weights@cov_matrix@weights
    vol_portfolio = np.sqrt(var_b)
    """
    print("\n" + "="*70)
    print("SECTION 6: PORTFOLIO FIGURES - BETA vs MONTE CARLO (MC BENCHMARK)")
    print("="*70)
    
    # Portfolio parameters
    SM = np.array([150, 120, 80, 95])  # Individual asset prices
    K = 90
    B = 150
    T = 1
    r = 0.05
    volatilities = np.array([0.25, 0.35, 0.20, 0.40])
    
    correlation = np.array([
        [1.0, 0.1, 0.4, 0.5],
        [0.1, 1.0, 0.3, 0.2],
        [0.4, 0.3, 1.0, 0.6],
        [0.5, 0.2, 0.6, 1.0]
    ])
    
    # Calculate portfolio parameters using CORRECT formula
    weights = np.array(len(volatilities)*[1/len(volatilities)])  # Equal weights
    S0_portfolio = np.dot(weights, SM)  # Use dot product, not mean
    cov_matrix = np.diag(volatilities) @ correlation @ np.diag(volatilities)
    var_b = weights @ cov_matrix @ weights
    vol_portfolio = np.sqrt(var_b)
    
    print(f"\n📊 Portfolio Parameters (CORRECT CALCULATION):")
    print(f"   Individual S0: {SM}")
    print(f"   Weights: {weights}")
    print(f"   S0 (dot product): {S0_portfolio:.4f}")
    print(f"   Volatilities: {volatilities}")
    print(f"   Portfolio Vol (from covariance): {vol_portfolio:.4f}")
    print(f"   Strike: {K}, Barrier: {B}")
    
    # Create portfolio options with correct S0 and volatility
    portfolio_vanilla = VanillaOption(
        s0=S0_portfolio, r=r, T=T, K=K, vol=vol_portfolio, call=True
    )
    portfolio_digital = DigitalOption(
        s0=S0_portfolio, r=r, T=T, K=K, vol=vol_portfolio, 
        call=True, digital_type='asset_or_nothing'
    )
    portfolio_barrier = BarrierOption(
        s0=S0_portfolio, r=r, T=T, K=K, barrier=B, vol=vol_portfolio, 
        call=True, barrier_direction='up', barrier_type='out'
    )
    
    options_list = [portfolio_vanilla, portfolio_digital, portfolio_barrier]
    option_names = ['Vanilla Call', 'Digital Asset Call', 'Barrier Call Up-Out']
    filenames = ['05_Portfolio_BetaVsMC_Vanilla', '05_Portfolio_BetaVsMC_Digital', '05_Portfolio_BetaVsMC_Barrier']
    
    portfolio_results = {
        'section': 'Portfolio Figures - Beta vs MC',
        'calculation_method': 'Correct covariance-based volatility',
        'portfolio_params': {
            'num_assets': len(volatilities),
            'individual_prices': SM.tolist(),
            'weights': weights.tolist(),
            'portfolio_s0': float(S0_portfolio),
            'portfolio_vol': float(vol_portfolio),
            'strike': float(K),
            'barrier': float(B),
            'rate': float(r),
            'time': float(T),
            'volatilities': volatilities.tolist(),
            'correlation': correlation.tolist()
        },
        'figures': []
    }
    
    # Generate convergence figures for each option type
    for option, option_name, filename in zip(options_list, option_names, filenames):
        print(f"\n🔄 Generating figure: {option_name}")
        print(f"   Using Monte Carlo (1M paths) as benchmark")
        
        try:
            # Generate convergence figure: Beta-Binomial vs MC
            fig, ax, results = plot_beta_vs_mc_portfolio(
                option=option,
                S0=SM,
                correlation=correlation,
                volatilities=volatilities,
                mc_benchmark_paths=1_000_000,
                is_calc_delta=False,
                beta_steps_range=np.arange(10_000, 1_000_000, 50_000),
                title=f'Portfolio {option_name}:\nBeta-Binomial vs Monte Carlo (1M paths)',
                figsize=(14, 7),
                error_type='absolute',
                scale='log'
            )
            
            # Save figure
            save_figure(fig, filename)
            plt.close(fig)
            
            # Extract results
            if results:
                figure_info = {
                    'option': option_name,
                    'mc_benchmark_paths': 1_000_000,
                    'mc_benchmark_price': float(results.get('mc_benchmark', np.nan)),
                    'beta_steps_tested': [int(x) for x in results.get('beta_steps', [])],
                    'beta_prices': [float(x) for x in results.get('beta_prices', [])],
                    'beta_errors': [float(x) for x in results.get('beta_errors', [])],
                    'min_error': float(np.nanmin(results.get('beta_errors', [np.nan]))),
                    'max_error': float(np.nanmax(results.get('beta_errors', [np.nan])))
                }
                portfolio_results['figures'].append(figure_info)
                print(f"   ✅ Figure saved: {filename}.pdf")
                print(f"      MC Benchmark: {figure_info['mc_benchmark_price']:.8f}")
                print(f"      Beta Min Error: {figure_info['min_error']:.2e}")
                print(f"      Beta Max Error: {figure_info['max_error']:.2e}")
        except Exception as e:
            print(f"   ❌ Error generating figure: {e}")
    
    # Save all results
    save_data_table(portfolio_results, "06_Portfolio_BetaVsMC_Figures")
    print(f"\n✅ Completed Section 6: Portfolio figures with Monte Carlo benchmark")
    
    return portfolio_results

def generate_portfolio_delta_figures():
    """
    Figures 6a-6c: Portfolio Delta Analysis - Beta-Binomial vs Monte Carlo Convergence
    Using CORRECT portfolio volatility calculation:
    weights = np.array(len(volatilities)*[1/len(volatilities)])
    S0_portfolio = np.dot(weights, SM)
    cov_matrix = np.diag(volatilities)@correlation@np.diag(volatilities)
    var_b = weights@cov_matrix@weights
    vol_portfolio = np.sqrt(var_b)
    """
    print("\n" + "="*70)
    print("SECTION 7: PORTFOLIO DELTA FIGURES - BETA vs MONTE CARLO (MC BENCHMARK)")
    print("="*70)
    
    # Portfolio parameters
    SM = np.array([150, 120, 80, 95])  # Individual asset prices
    K = 90
    B = 150
    T = 1
    r = 0.05
    volatilities = np.array([0.25, 0.35, 0.20, 0.40])
    
    correlation = np.array([
        [1.0, 0.1, 0.4, 0.5],
        [0.1, 1.0, 0.3, 0.2],
        [0.4, 0.3, 1.0, 0.6],
        [0.5, 0.2, 0.6, 1.0]
    ])
    
    # Calculate portfolio parameters using CORRECT formula
    weights = np.array(len(volatilities)*[1/len(volatilities)])  # Equal weights
    S0_portfolio = np.dot(weights, SM)  # Use dot product, not mean
    cov_matrix = np.diag(volatilities) @ correlation @ np.diag(volatilities)
    var_b = weights @ cov_matrix @ weights
    vol_portfolio = np.sqrt(var_b)
    
    print(f"\n📊 Portfolio Parameters (CORRECT CALCULATION):")
    print(f"   Individual S0: {SM}")
    print(f"   Weights: {weights}")
    print(f"   S0 (dot product): {S0_portfolio:.4f}")
    print(f"   Volatilities: {volatilities}")
    print(f"   Portfolio Vol (from covariance): {vol_portfolio:.4f}")
    print(f"   Strike: {K}, Barrier: {B}")
    
    # Create portfolio options with correct S0 and volatility
    portfolio_vanilla = VanillaOption(
        s0=S0_portfolio, r=r, T=T, K=K, vol=vol_portfolio, call=True
    )
    portfolio_digital = DigitalOption(
        s0=S0_portfolio, r=r, T=T, K=K, vol=vol_portfolio, 
        call=True, digital_type='asset_or_nothing'
    )
    portfolio_barrier = BarrierOption(
        s0=S0_portfolio, r=r, T=T, K=K, barrier=B, vol=vol_portfolio, 
        call=True, barrier_direction='up', barrier_type='out'
    )
    
    options_list = [portfolio_vanilla, portfolio_digital, portfolio_barrier]
    option_names = ['Vanilla Call', 'Digital Asset Call', 'Barrier Call Up-Out']
    filenames = ['06_Portfolio_Delta_BetaVsMC_Vanilla', '06_Portfolio_Delta_BetaVsMC_Digital', '06_Portfolio_Delta_BetaVsMC_Barrier']
    
    portfolio_results = {
        'section': 'Portfolio Figures - Beta vs MC',
        'calculation_method': 'Correct covariance-based volatility',
        'portfolio_params': {
            'num_assets': len(volatilities),
            'individual_prices': SM.tolist(),
            'weights': weights.tolist(),
            'portfolio_s0': float(S0_portfolio),
            'portfolio_vol': float(vol_portfolio),
            'strike': float(K),
            'barrier': float(B),
            'rate': float(r),
            'time': float(T),
            'volatilities': volatilities.tolist(),
            'correlation': correlation.tolist()
        },
        'figures': []
    }
    
    # Generate convergence figures for each option type
    for option, option_name, filename in zip(options_list, option_names, filenames):
        print(f"\n🔄 Generating figure: {option_name}")
        print(f"   Using Monte Carlo (1M paths) as benchmark")
        
        try:
            # Generate convergence figure: Beta-Binomial vs MC
            fig, ax, results = plot_beta_vs_mc_portfolio(
                option=option,
                S0=SM,
                correlation=correlation,
                volatilities=volatilities,
                mc_benchmark_paths=1_000_000,
                is_calc_delta=True,
                beta_steps_range=np.arange(10_000, 1_000_000, 50_000),
                title=f'Portfolio Delta {option_name}:\nBeta-Binomial vs Monte Carlo (1M paths)',
                figsize=(14, 7),
                error_type='absolute',
                scale='log'
            )
            
            # Save figure
            save_figure(fig, filename)
            plt.close(fig)
            
            # Extract results
            if results:
                figure_info = {
                    'option': option_name,
                    'mc_benchmark_paths': 1_000_000,
                    'mc_benchmark_price': float(results.get('mc_benchmark', np.nan)),
                    'beta_steps_tested': [int(x) for x in results.get('beta_steps', [])],
                    'beta_prices': [float(x) for x in results.get('beta_prices', [])],
                    'beta_errors': [float(x) for x in results.get('beta_errors', [])],
                    'min_error': float(np.nanmin(results.get('beta_errors', [np.nan]))),
                    'max_error': float(np.nanmax(results.get('beta_errors', [np.nan])))
                }
                portfolio_results['figures'].append(figure_info)
                print(f"   ✅ Figure saved: {filename}.pdf")
                print(f"      MC Benchmark: {figure_info['mc_benchmark_price']:.8f}")
                print(f"      Beta Min Error: {figure_info['min_error']:.2e}")
                print(f"      Beta Max Error: {figure_info['max_error']:.2e}")
        except Exception as e:
            print(f"   ❌ Error generating figure: {e}")
    
    # Save all results
    save_data_table(portfolio_results, "07_Portfolio_Delta_BetaVsMC_Figures")
    print(f"\n✅ Completed Section 7: Portfolio Delta figures with Monte Carlo benchmark")
    
    return portfolio_results

# ============================================================================
# SECTION 8: PORTFOLIO TIMING vs NUMBER OF MC PATHS
# ============================================================================

def generate_portfolio_timing_vs_paths():
    """
    Figure 7: Performance Analysis - MC Timing vs Paths vs Beta-Binomial O(1)
    Shows how MC time scales with number of paths.
    """
    print("\n" + "="*70)
    print("SECTION 8: PORTFOLIO TIMING vs NUMBER OF MC PATHS")
    print("="*70)
    
    # Portfolio parameters
    SM = np.array([150, 120, 80, 95])
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
    
    # Calculate portfolio parameters
    weights = np.array(len(volatilities)*[1/len(volatilities)])
    S0_portfolio = np.dot(weights, SM)
    cov_matrix = np.diag(volatilities) @ correlation @ np.diag(volatilities)
    var_b = weights @ cov_matrix @ weights
    vol_portfolio = np.sqrt(var_b)
    
    # Create option
    portfolio_vanilla = VanillaOption(s0=S0_portfolio, r=r, T=T, K=K, vol=vol_portfolio, call=True)
    
    # Path ranges to test - Beta steps scale with MC paths for consistency
    mc_paths_list = np.array([1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000], dtype=int)
    beta_steps_list = mc_paths_list.copy()  # Same values as MC paths for consistency
    
    mc_times = []
    beta_times = []
    
    print(f"\n⏱️  Testing MC with different path counts (vs Beta with same step counts)...")
    print(f"    Portfolio S0: {S0_portfolio:.2f}, Vol: {vol_portfolio:.4f}")
    print(f"    Note: Beta steps scale with MC paths for consistent comparison")
    
    # Test Monte Carlo and Beta-Binomial with matching step/path counts
    for paths, beta_steps in zip(mc_paths_list, beta_steps_list):
        print(f"\n    Testing paths={paths:>7,} with beta_steps={beta_steps:>7,}:")
        
        # Test Beta-Binomial
        beta_trial_times = []
        for trial in range(3):
            start = time.time()
            beta_price = binomial_beta_price(portfolio_vanilla, model_name='garcia', m=beta_steps, calc_delta=False)
            beta_trial_times.append((time.time() - start) * 1000)
        beta_time_avg = np.mean(beta_trial_times)
        beta_times.append(beta_time_avg)
        
        # Test Monte Carlo
        mc_trial_times = []
        for trial in range(3):  # 3 trials for averaging
            start = time.time()
            mc_price = portfolio_monte_carlo_price(
                option=portfolio_vanilla,
                S0=SM,
                correlation=correlation,
                volatilities=volatilities,
                m=1,
                N=paths,
                calc_delta=False
            )
            elapsed = (time.time() - start) * 1000
            mc_trial_times.append(elapsed)
        
        avg_time = np.mean(mc_trial_times)
        mc_times.append(avg_time)
        speedup = avg_time / beta_time_avg
        print(f"       MC: {avg_time:8.3f} ms | Beta: {beta_time_avg:8.3f} ms | Speedup: {speedup:7.2f}x")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot MC scaling
    ax.plot(mc_paths_list / 1000, mc_times, 'o-', linewidth=2.5, markersize=8, 
            label='Monte Carlo (linear in N)', color='#ff7f0e')
    
    # Plot Beta scaling
    ax.plot(mc_paths_list / 1000, beta_times, 's-', linewidth=2.5, markersize=8, 
            label='Beta-Binomial (linear in steps)', color='#1f77b4')
    
    # Add speedup annotation at maximum
    max_speedup = np.max(np.array(mc_times) / np.array(beta_times))
    speedup_at_1M = mc_times[-1] / beta_times[-1]
    ax.text(900, max(mc_times) * 0.5, f'{speedup_at_1M:.1f}x speedup\nat 1M paths/steps', 
            fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Formatting
    ax.set_xlabel('Number of Paths/Steps (thousands)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Computation Time (milliseconds)', fontsize=13, fontweight='bold')
    ax.set_title('Portfolio Vanilla Call:\nMonte Carlo vs Beta-Binomial (Proportional Scaling)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_xlim(0, 1050)
    
    save_figure(fig, "07_Portfolio_Timing_vs_Paths")
    plt.close(fig)
    
    # Save data
    timing_data = {
        'section': 'Portfolio Timing vs MC Paths',
        'analysis_type': 'MC paths scaling vs Beta steps (proportional)',
        'portfolio_params': {
            'num_assets': len(volatilities),
            'individual_prices': SM.tolist(),
            'portfolio_s0': float(S0_portfolio),
            'portfolio_vol': float(vol_portfolio)
        },
        'mc_paths': mc_paths_list.tolist(),
        'beta_steps': beta_steps_list.tolist(),
        'mc_times_ms': mc_times,
        'beta_times_ms': beta_times,
        'speedup_at_1M': float(speedup_at_1M)
    }
    save_data_table(timing_data, "08_Portfolio_Timing_vs_Paths")
    
    print(f"\n✅ Saved: Dissertation_Figures/07_Portfolio_Timing_vs_Paths.pdf")
    print(f"✅ Completed Section 8: Portfolio timing vs paths analysis (proportional scaling)")
    
    return timing_data

# ============================================================================
# SECTION 9: PORTFOLIO TIMING vs NUMBER OF ASSETS
# ============================================================================

def generate_portfolio_timing_vs_assets():
    """
    Figure 8: Performance Analysis - MC Timing vs Number of Assets vs Beta-Binomial O(1)
    Shows how computation scales with portfolio size.
    """
    print("\n" + "="*70)
    print("SECTION 9: PORTFOLIO TIMING vs NUMBER OF ASSETS")
    print("="*70)
    
    asset_counts = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
    mc_paths = 100_000  # Fixed MC paths
    beta_steps = 100_000  # Fixed Beta steps
    
    mc_times = []
    beta_times = []
    
    T = 1
    r = 0.05
    K_ratio = 0.9  # Strike as 90% of portfolio S0
    
    print(f"\n⏱️  Testing scaling with number of assets...")
    print(f"    MC Paths: {mc_paths:,}")
    print(f"    Beta Steps: {beta_steps:,}")
    
    for num_assets in asset_counts:
        print(f"\n    Testing {num_assets} assets:")
        
        # Generate random asset parameters
        np.random.seed(42 + num_assets)  # For reproducibility
        SM = 100 * np.ones(num_assets)  # All S0 = 100
        volatilities = np.random.uniform(0.15, 0.45, num_assets)
        
        # Generate random correlation matrix (ensure positive definite)
        while True:
            L = np.random.uniform(-0.5, 0.5, (num_assets, num_assets))
            correlation = L @ L.T
            correlation = correlation / np.sqrt(np.diag(correlation))[:, None]
            np.fill_diagonal(correlation, 1.0)
            # Check if positive definite
            try:
                np.linalg.cholesky(correlation)
                break
            except:
                pass
        
        # Calculate portfolio parameters
        weights = np.ones(num_assets) / num_assets
        S0_portfolio = np.dot(weights, SM)
        cov_matrix = np.diag(volatilities) @ correlation @ np.diag(volatilities)
        var_b = weights @ cov_matrix @ weights
        vol_portfolio = np.sqrt(var_b)
        K = K_ratio * S0_portfolio
        
        # Create option
        portfolio_vanilla = VanillaOption(s0=S0_portfolio, r=r, T=T, K=K, vol=vol_portfolio, call=True)
        
        # Test Beta-Binomial
        beta_trial_times = []
        for trial in range(3):
            start = time.time()
            beta_price = binomial_beta_price(portfolio_vanilla, model_name='garcia', m=beta_steps, calc_delta=False)
            beta_trial_times.append((time.time() - start) * 1000)
        beta_time_avg = np.mean(beta_trial_times)
        beta_times.append(beta_time_avg)
        
        # Test Monte Carlo
        mc_trial_times = []
        for trial in range(3):
            start = time.time()
            mc_price = portfolio_monte_carlo_price(
                option=portfolio_vanilla,
                S0=SM,
                correlation=correlation,
                volatilities=volatilities,
                m=1,
                N=mc_paths,
                calc_delta=False
            )
            mc_trial_times.append((time.time() - start) * 1000)
        mc_time_avg = np.mean(mc_trial_times)
        mc_times.append(mc_time_avg)
        
        speedup = mc_time_avg / beta_time_avg
        print(f"       MC:   {mc_time_avg:7.3f} ms | Beta:  {beta_time_avg:7.3f} ms | Speedup: {speedup:6.2f}x")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot MC scaling
    ax.plot(asset_counts, mc_times, 'o-', linewidth=2.5, markersize=10, 
            label=f'Monte Carlo ({mc_paths:,} paths, linear scaling)', color='#ff7f0e')
    
    # Plot Beta constant time
    ax.axhline(y=np.mean(beta_times), color='#1f77b4', linestyle='--', linewidth=2.5, 
               label=f'Beta-Binomial O(1) ≈ {np.mean(beta_times):.3f} ms', alpha=0.8)
    
    # Add speedup annotations
    max_speedup = np.max(np.array(mc_times) / np.mean(beta_times))
    ax.text(8, mc_times[-1]*0.5, f'{max_speedup:.0f}x speedup\nat 10 assets', 
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Formatting
    ax.set_xlabel('Number of Assets in Portfolio', fontsize=13, fontweight='bold')
    ax.set_ylabel('Computation Time (milliseconds)', fontsize=13, fontweight='bold')
    ax.set_title('Portfolio Pricing Scaling:\nMonte Carlo vs Beta-Binomial O(1) Complexity', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_xticks(asset_counts)
    ax.set_xlim(1.5, 10.5)
    
    save_figure(fig, "08_Portfolio_Timing_vs_Assets")
    plt.close(fig)
    
    # Save data
    timing_data = {
        'section': 'Portfolio Timing vs Number of Assets',
        'analysis_type': 'Scaling with portfolio size',
        'mc_paths': int(mc_paths),
        'beta_steps': int(beta_steps),
        'asset_counts': asset_counts.tolist(),
        'mc_times_ms': mc_times,
        'beta_times_ms': beta_times,
        'max_speedup': float(max_speedup)
    }
    save_data_table(timing_data, "09_Portfolio_Timing_vs_Assets")
    
    print(f"✅ Saved: Dissertation_Figures/08_Portfolio_Timing_vs_Assets.pdf")
    print(f"✅ Completed Section 9: Portfolio timing vs assets analysis")
    
    return timing_data

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute all figure generation"""
    print("\n" + "="*70)
    print("DISSERTATION FIGURE GENERATION - GARCIA MODEL")
    print("="*70)
    print(f"Output Directory: {OUTPUT_DIR.absolute()}")
    print(f"Data Directory: {DATA_DIR.absolute()}")
    print(f"High Resolution: {DPI_DISSERTATION} DPI")
    print("="*70)
    
    try:
        generate_garcia_beta_tree_comparison()
        
        generate_garcia_crr_convergence()
        
        generate_delta_analysis()
        
        generate_performance_analysis()
        
        generate_portfolio_analysis()
        
        generate_portfolio_figures()  # Portfolio Beta vs MC Figures
        
        generate_portfolio_delta_figures()  # Portfolio Delta Beta vs MC Figures
        
        generate_portfolio_timing_vs_paths()  # NEW: MC Timing vs Paths
        
        generate_portfolio_timing_vs_assets()  # NEW: MC Timing vs Assets
        
        print("\n" + "="*70)
        print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
        print("="*70)
        print(f"\nFigures saved to: {OUTPUT_DIR.absolute()}")
        print(f"Data saved to: {DATA_DIR.absolute()}")
        print("\nGenerated files are ready for LaTeX integration and further analysis.")
        
    except Exception as e:
        print(f"\n❌ Error during figure generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
