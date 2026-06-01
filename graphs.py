import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import time

from options_types import VanillaOption, DigitalOption, BarrierOption
from methods import (
    BlackScholes_price,
    binomial_beta_price,
    monte_carlo_price,
    binomial_price,
    portfolio_monte_carlo_price
)
from models import MODEL_PARAMETERS

def plot_convergence_to_bs(
    option,
    models_config,
    steps_range=None,
    error_type='absolute',
    scale='linear',
    figsize=(12, 7),
    title=None,
    colors=None,
    linestyles=None,
    markers=None,
    linewidth=2.5,
    markersize=8,
    grid=True,
    save_path=None,
    dpi=300,
    calc_delta=False,
    show_mean_lines=False,
    show_performance_summary=True
):
    """
    Plot convergence of pricing models to Black-Scholes benchmark.
    
    Parameters:
    -----------
    option : Option object
        The option to price (VanillaOption, DigitalOption, BarrierOption)
    
    models_config : list of dicts
        List of model configurations. Each dict must contain:
        - 'name': str, display name (e.g., 'GARCIA', 'CRR')
        - 'method': str, one of ['binomial_beta', 'monte_carlo', 'binomial_tree']
        - 'model_name': str (required for binomial_beta and binomial_tree), e.g., 'GARCIA', 'CRR'
        - 'mc_steps': int (optional for monte_carlo), number of time steps (default 100)
        - 'mc_paths': int (optional for monte_carlo), number of paths (default 50000)
        
    steps_range : list of int, optional
        Range of steps to test. Default: [10, 25, 50, 100, 200, 500, 1000]
    
    error_type : str
        'absolute' (default) or 'relative' error
    
    scale : str
        'linear' (default) or 'log' scale for both axes
    
    figsize : tuple
        Figure size (width, height) in inches
    
    title : str, optional
        Plot title. Auto-generated if None
    
    colors : list or dict, optional
        Colors for each model. Can be list or dict {model_name: color}
    
    linestyles : list or dict, optional
        Linestyles for each model
    
    markers : list or dict, optional
        Markers for each model
    
    linewidth : float
        Width of lines in plot
    
    markersize : float
        Size of markers
    
    grid : bool
        Whether to show grid
    
    save_path : str, optional
        Path to save figure. If None, only display
    
    dpi : int
        Resolution for saved figure (default 300, dissertation quality)
    
    calc_delta : bool
        Whether to calculate delta instead of price (default False)
    
    show_mean_lines : bool
        Whether to show horizontal lines representing mean error for each model (default False)
    
    show_performance_summary : bool
        Whether to print performance summary to console (default True)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    results_dict : dict with convergence data and mean performance
    """
    
    # Default steps range
    if steps_range is None:
        steps_range = [10, 25, 50, 100, 200, 500, 1000]
    
    # Calculate Black-Scholes benchmark (exact)
    bs_price = BlackScholes_price(option, calc_delta=calc_delta)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # Default colors (professional palette)
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    if colors is None:
        colors = {cfg['name']: default_colors[i % len(default_colors)] 
                 for i, cfg in enumerate(models_config)}
    
    # Default linestyles
    default_linestyles = ['-', '--', '-.', ':']
    if linestyles is None:
        linestyles = {cfg['name']: default_linestyles[i % len(default_linestyles)] 
                     for i, cfg in enumerate(models_config)}
    
    # Default markers
    default_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    if markers is None:
        markers = {cfg['name']: default_markers[i % len(default_markers)] 
                  for i, cfg in enumerate(models_config)}
    
    results_dict = {}
    mean_errors = {}  # Store mean errors for ranking
    
    # Plot convergence for each model
    for cfg in models_config:
        model_name = cfg['name']
        method = cfg['method']
        errors = []
        
        for m in steps_range:
            try:
                if method == 'binomial_beta':
                    price = binomial_beta_price(option, model_name=cfg['model_name'], m=m, calc_delta=calc_delta)
                elif method == 'monte_carlo':
                    mc_steps = cfg.get('mc_steps', 100)
                    mc_paths = cfg.get('mc_paths', 50000)
                    price = monte_carlo_price(option, m=mc_steps, N=mc_paths, calc_delta=calc_delta)
                elif method == 'binomial_tree':
                    price = binomial_price(option, model_name=cfg['model_name'], m=m, calc_delta=calc_delta)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Calculate error
                if error_type == 'relative':
                    error = abs(price - bs_price) / abs(bs_price)
                elif error_type == 'absolute':  # absolute
                    error = abs(price - bs_price)
                else:
                    error = price - bs_price
                
                errors.append(error)
            except Exception as e:
                print(f"Error computing {model_name} at m={m}: {e}")
                errors.append(np.nan)
        
        # Calculate mean error (excludes NaN values)
        mean_error = np.nanmean(errors)
        mean_errors[model_name] = mean_error
        
        results_dict[model_name] = {
            'steps': steps_range, 
            'errors': errors,
            'mean_error': mean_error,
            'min_error': np.nanmin(errors),
            'max_error': np.nanmax(errors)
        }
        
        # Plot line
        ax.plot(
            steps_range, errors,
            color=colors[model_name],
            linestyle=linestyles[model_name],
            marker=markers[model_name],
            linewidth=linewidth,
            markersize=markersize,
            label=model_name,
            alpha=0.85
        )
        
        # Optionally add mean line
        if show_mean_lines:
            ax.axhline(y=mean_error, color=colors[model_name], 
                      linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Configure axes
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Steps (log scale)', fontsize=13, fontweight='bold')
        if error_type == 'relative':
            ax.set_ylabel('Relative Error vs BS (log scale)', fontsize=13, fontweight='bold')
        elif error_type == 'absolute':
            ax.set_ylabel('Absolute Error vs BS (log scale)', fontsize=13, fontweight='bold')
        else:
            ax.set_ylabel('Error vs BS (log scale)', fontsize=13, fontweight='bold')
    else:
        ax.set_xlabel('Number of Steps', fontsize=13, fontweight='bold')
        if error_type == 'relative':
            ax.set_ylabel('Relative Error vs BS', fontsize=13, fontweight='bold')
        elif error_type == 'absolute':
            ax.set_ylabel('Absolute Error vs BS', fontsize=13, fontweight='bold')
        else:
            ax.set_ylabel('Error vs BS', fontsize=13, fontweight='bold')
    
    # Title
    if title is None:
        title = f"Convergence to Black-Scholes Benchmark ({error_type.capitalize()} Error)"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    
    # Grid
    if grid:
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    # Formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    # Print performance summary
    if show_performance_summary:
        print("\n" + "="*70)
        print("📊 CONVERGENCE PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Error Type: {error_type.upper()} | Range: {min(steps_range)}-{max(steps_range)} steps\n")
        
        # Sort models by mean error (best to worst)
        sorted_models = sorted(mean_errors.items(), key=lambda x: x[1])
        
        for rank, (model_name, mean_err) in enumerate(sorted_models, 1):
            min_err = results_dict[model_name]['min_error']
            max_err = results_dict[model_name]['max_error']
            medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
            print(f"{medal} #{rank}. {model_name:25s} | Mean: {mean_err:.6e} | Min: {min_err:.6e} | Max: {max_err:.6e}")
        
        print("="*70 + "\n")
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    plt.show()
    
    return fig, ax, results_dict

def plot_convergence_multiple(
    options_list,
    models_config,
    steps_range=None,
    error_type='absolute',
    scale='linear',
    figsize=None,
    titles=None,
    colors=None,
    linestyles=None,
    markers=None,
    linewidth=2.5,
    markersize=8,
    grid=True,
    save_path=None,
    dpi=300,
    calc_delta=False,
    show_mean_lines=False,
    show_performance_summary=True,
    risk_factor=1.0,
    risk_method='linear'
):
    """
    Plot convergence of pricing models for multiple options (with subplots).
    
    Parameters:
    -----------
    options_list : list of Option objects
        List of options to plot (e.g., [digital_call, digital_put, digital_asset_call, ...])
    
    models_config : list of dicts
        Pricing models configuration (same as plot_convergence_to_bs)
    
    steps_range : list of int, optional
        Range of steps to test. Default: [10, 25, 50, 100, 200, 500, 1000]
    
    error_type : str
        'absolute' (default) or 'relative' error
    
    scale : str
        'linear' (default) or 'log' scale
    
    figsize : tuple, optional
        Figure size. If None, auto-calculated based on number of options
        Examples: (12, 7) for 1x1, (16, 10) for 2x2, (18, 12) for 2x3
    
    titles : list of str, optional
        Titles for each subplot. If None, auto-generated
    
    colors : dict, optional
        Colors for models (same as plot_convergence_to_bs)
    
    linestyles : dict, optional
        Linestyles for models (same as plot_convergence_to_bs)
    
    markers : dict, optional
        Markers for models (same as plot_convergence_to_bs)
    
    linewidth : float
        Width of lines
    
    markersize : float
        Size of markers
    
    grid : bool
        Whether to show grid
    
    save_path : str, optional
        Path to save figure
    
    dpi : int
        Resolution for saved figure
    
    calc_delta : bool
        Calculate delta instead of price
    
    show_mean_lines : bool
        Whether to show horizontal lines representing mean error for each model (default False)
    
    show_performance_summary : bool
        Whether to print performance summary to console (default True)
    
    risk_factor : float, optional
        Risk weighting factor for ranking (default 1.0).
        - risk_factor=0: rank only by mean
        - risk_factor=1: equal weight to mean and std
        - risk_factor>1: penalizes variance more heavily
    
    risk_method : str, optional
        Method for combining mean and std into risk score (default 'linear'):
        - 'linear': risk_score = |mean| + risk_factor × std
          Simple additive combination. Better for intuitive interpretation.
        - 'l2': risk_score = √(mean² + (risk_factor × std)²)
          Euclidean distance. More mathematically rigorous, penalizes outliers more.
    
    Returns:
    --------
    fig, axes, results_dict : figure, axes array, convergence data with means, stds and variances
    """
    
    if steps_range is None:
        steps_range = [10, 25, 50, 100, 200, 500, 1000]
    
    if risk_method not in ['linear', 'l2']:
        raise ValueError("risk_method must be 'linear' or 'l2'")
    
    # Type conversion for robustness - handle tuples, lists, etc.
    try:
        if isinstance(risk_factor, (tuple, list)):
            risk_factor = float(risk_factor[0]) if len(risk_factor) > 0 else 1.0
        else:
            risk_factor = float(risk_factor)
    except (ValueError, IndexError, TypeError):
        risk_factor = 1.0
    
    n_options = len(options_list)
    
    # Auto-determine grid layout
    if n_options == 1:
        n_rows, n_cols = 1, 1
    elif n_options == 2:
        n_rows, n_cols = 1, 2
    elif n_options == 3:
        n_rows, n_cols = 1, 3
    elif n_options == 4:
        n_rows, n_cols = 2, 2
    elif n_options == 6:
        n_rows, n_cols = 2, 3
    else:
        n_cols = min(3, (n_options + 1) // 2)
        n_rows = (n_options + n_cols - 1) // n_cols
    
    # Auto-calculate figsize
    if figsize is None:
        figsize = (7 * n_cols, 5.5 * n_rows)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=100)
    
    # Ensure axes is always 2D
    if n_options == 1:
        axes = np.array([[axes]])
    elif min(n_rows, n_cols) == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    # Setup colors, linestyles, markers (same logic)
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    if colors is None:
        colors = {cfg['name']: default_colors[i % len(default_colors)] 
                 for i, cfg in enumerate(models_config)}
    
    default_linestyles = ['-', '--', '-.', ':']
    if linestyles is None:
        linestyles = {cfg['name']: default_linestyles[i % len(default_linestyles)] 
                     for i, cfg in enumerate(models_config)}
    
    default_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    if markers is None:
        markers = {cfg['name']: default_markers[i % len(default_markers)] 
                  for i, cfg in enumerate(models_config)}
    
    results_dict = {}
    
    # Auto-generate titles if not provided
    if titles is None:
        titles = [f"Option {i+1}" for i in range(n_options)]
    
    # Plot each option
    for idx, option in enumerate(options_list):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Calculate Black-Scholes benchmark
        bs_price = BlackScholes_price(option, calc_delta=calc_delta)
        
        option_results = {}
        mean_errors_per_option = {}
        std_errors_per_option = {}
        
        # Plot convergence for each model
        for cfg in models_config:
            model_name = cfg['name']
            method = cfg['method']
            errors = []
            
            for m in steps_range:
                try:
                    if method == 'binomial_beta':
                        price = binomial_beta_price(option, model_name=cfg['model_name'], m=m, calc_delta=calc_delta)
                    elif method == 'monte_carlo':
                        mc_steps = cfg.get('mc_steps', 100)
                        mc_paths = cfg.get('mc_paths', 50000)
                        price = monte_carlo_price(option, m=mc_steps, N=mc_paths, calc_delta=calc_delta)
                    elif method == 'binomial_tree':
                        price = binomial_price(option, model_name=cfg['model_name'], m=m, calc_delta=calc_delta)
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    
                    # Calculate error
                    if error_type == 'relative':
                        error = abs(price - bs_price) / abs(bs_price)
                    elif error_type == 'absolute':  # absolute
                        error = abs(price - bs_price)
                    else:
                        error = price - bs_price
                    errors.append(error)
                except Exception as e:
                    print(f"Error computing {model_name} at m={m}: {e}")
                    errors.append(np.nan)
            
            # Calculate mean, std, and variance
            mean_error = np.nanmean(errors)
            std_error = float(np.nanstd(errors))  # Convert to float for safety
            var_error = std_error ** 2
            mean_errors_per_option[model_name] = mean_error
            std_errors_per_option[model_name] = std_error
            
            option_results[model_name] = {
                'steps': steps_range, 
                'errors': errors,
                'mean_error': mean_error,
                'std_error': std_error,
                'var_error': var_error,
                'min_error': np.nanmin(errors),
                'max_error': np.nanmax(errors)
            }
            
            # Plot line
            ax.plot(
                steps_range, errors,
                color=colors[model_name],
                linestyle=linestyles[model_name],
                marker=markers[model_name],
                linewidth=linewidth,
                markersize=markersize,
                label=model_name,
                alpha=0.85
            )
            
            # Optionally add mean line
            if show_mean_lines:
                ax.axhline(y=mean_error, color=colors[model_name], 
                          linestyle=':', alpha=0.5, linewidth=1.5)
        
        results_dict[f"option_{idx}"] = {
            'title': titles[idx],
            'models': option_results,
            'mean_errors_per_model': mean_errors_per_option,
            'std_errors_per_model': std_errors_per_option
        }
        
        # Configure axes
        if scale == 'log':
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Steps (log scale)', fontsize=11, fontweight='bold')
            if error_type == 'relative':
                ax.set_ylabel('Relative Error (log)', fontsize=11, fontweight='bold')
            elif error_type == 'absolute':
                ax.set_ylabel('Absolute Error (log)', fontsize=11, fontweight='bold')
            else:
                ax.set_ylabel('Error (log)', fontsize=11, fontweight='bold')
        else:
            ax.set_xlabel('Steps', fontsize=11, fontweight='bold')
            if error_type == 'relative':
                ax.set_ylabel('Relative Error', fontsize=11, fontweight='bold')
            elif error_type == 'absolute':
                ax.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
            else:
                ax.set_ylabel('Error', fontsize=11, fontweight='bold')
        
        # Title
        ax.set_title(titles[idx], fontsize=12, fontweight='bold', pad=12)
        
        # Legend (only on first subplot to avoid clutter)
        if idx == 0:
            ax.legend(fontsize=9, loc='best', framealpha=0.95, edgecolor='black')
        
        # Grid
        if grid:
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        
        # Formatting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)
    
    # Hide empty subplots
    total_subplots = n_rows * n_cols
    for idx in range(n_options, total_subplots):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f"Convergence Analysis: {n_options} Options", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Print performance summary for multiple options
    if show_performance_summary:
        print("\n" + "="*140)
        print("📊 CONVERGENCE PERFORMANCE SUMMARY (MULTIPLE OPTIONS)")
        print("="*140)
        print(f"Error Type: {error_type.upper()} | Range: {min(steps_range)}-{max(steps_range)} steps | Risk Factor: {risk_factor} | Risk Method: {risk_method.upper()}\n")
        
        for idx in range(n_options):
            option_key = f"option_{idx}"
            print(f"\n{titles[idx]:50s}")
            print("-" * 140)
            
            mean_errors = results_dict[option_key]['mean_errors_per_model']
            std_errors = results_dict[option_key]['std_errors_per_model']
            
            # Calculate risk score based on method
            risk_scores = {}
            for model_name in mean_errors.keys():
                mean_err = mean_errors[model_name]
                std_err = float(std_errors[model_name])  # Ensure float type
                
                # For price difference (not absolute/relative), use absolute value of mean
                if error_type not in ['absolute', 'relative']:
                    mean_abs = abs(mean_err)
                else:
                    mean_abs = mean_err
                
                if risk_method == 'linear':
                    # Linear: |mean| + k × std
                    risk_scores[model_name] = mean_abs + risk_factor * std_err
                elif risk_method == 'l2':
                    # L2 (Euclidean): √(mean² + (k × std)²)
                    risk_scores[model_name] = np.sqrt(mean_abs**2 + (risk_factor * std_err)**2)
            
            sorted_models = sorted(risk_scores.items(), key=lambda x: x[1])
            
            for rank, (model_name, risk_score) in enumerate(sorted_models, 1):
                model_data = results_dict[option_key]['models'][model_name]
                mean_err = mean_errors[model_name]
                std_err = std_errors[model_name]
                var_err = model_data['var_error']
                medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
                print(f"  {medal} #{rank}. {model_name:25s} | Mean: {mean_err:.6e} | Std: {std_err:.6e} | Var: {var_err:.6e} | Risk: {risk_score:.6e}")
        
        print("\n" + "="*140 + "\n")
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    plt.show()
    return fig, axes, results_dict

def plot_timing_comparison(
    option,
    methods_config,
    steps_range=None,
    scale='linear',
    figsize=(12, 7),
    title=None,
    colors=None,
    linestyles=None,
    markers=None,
    linewidth=2.5,
    markersize=8,
    grid=True,
    save_path=None,
    dpi=300,
    num_trials=3,
    benchmark_model='GARCIA'
):
    """
    Plot computational time comparison between pricing methods.
    
    Parameters:
    -----------
    option : Option object
        The option to price
    
    methods_config : list of dicts
        List of method configurations. Each dict must contain:
        - 'name': str, display name (e.g., 'Beta-Binomial', 'Monte Carlo')
        - 'method': str, one of ['binomial_beta', 'monte_carlo', 'binomial_tree']
        - 'model_name': str (for binomial_beta/tree), e.g., 'GARCIA', 'CRR'
        - 'mc_steps': int (for monte_carlo), number of time steps
        - 'mc_paths': int (for monte_carlo), number of paths
        
    steps_range : list of int, optional
        Range of steps to test. Default: [100, 500, 1000, 5000, 10000]
    
    scale : str
        'linear' (default) or 'log' for both axes
    
    figsize : tuple
        Figure size in inches
    
    title : str, optional
        Plot title. Auto-generated if None
    
    colors : dict or list, optional
        Colors for each method
    
    linestyles : dict or list, optional
        Linestyles for each method
    
    markers : dict or list, optional
        Markers for each method
    
    linewidth : float
        Width of lines
    
    markersize : float
        Size of markers
    
    grid : bool
        Whether to show grid
    
    save_path : str, optional
        Path to save figure
    
    dpi : int
        Resolution for saved figure
    
    num_trials : int
        Number of repeated trials for timing (for averaging)
    
    benchmark_model : str
        Which model to use as reference (default 'GARCIA')
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    timing_dict : dict with timing data
    """
    
    # Default steps range (careful with large binomial tree values!)
    if steps_range is None:
        steps_range = [100, 500, 1000, 5000, 10000]
    else:
        # Warn if binomial tree steps are too large
        for cfg in methods_config:
            if cfg['method'] == 'binomial_tree' and max(steps_range) > 5000:
                print(f"⚠️  WARNING: Binomial tree with {max(steps_range)} steps can be VERY slow!")
                print("   Consider using smaller steps_range or fewer methods.")
    
    # Setup plot
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # Default colors
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    if colors is None:
        colors = {cfg['name']: default_colors[i % len(default_colors)] 
                 for i, cfg in enumerate(methods_config)}
    
    # Default linestyles
    default_linestyles = ['-', '--', '-.']
    if linestyles is None:
        linestyles = {cfg['name']: default_linestyles[i % len(default_linestyles)] 
                     for i, cfg in enumerate(methods_config)}
    
    # Default markers
    default_markers = ['o', 's', '^']
    if markers is None:
        markers = {cfg['name']: default_markers[i % len(default_markers)] 
                  for i, cfg in enumerate(methods_config)}
    
    timing_dict = {}
    
    # Timing measurements
    for cfg in methods_config:
        method_name = cfg['name']
        method = cfg['method']
        times = []
        
        for m in steps_range:
            trial_times = []
            
            for trial in range(num_trials):
                try:
                    if method == 'binomial_beta':
                        start = time.perf_counter()
                        BlackScholes_price(option) if cfg['model_name'] == 'BS' else binomial_beta_price(
                            option, model_name=cfg['model_name'], m=m
                        )
                        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
                    
                    elif method == 'monte_carlo':
                        mc_steps = cfg.get('mc_steps', 100)
                        mc_paths = cfg.get('mc_paths', 50000)
                        start = time.perf_counter()
                        monte_carlo_price(option, m=mc_steps, N=mc_paths)
                        elapsed = (time.perf_counter() - start) * 1000
                    
                    elif method == 'binomial_tree':
                        start = time.perf_counter()
                        binomial_price(option, model_name=cfg['model_name'], m=m)
                        elapsed = (time.perf_counter() - start) * 1000
                    
                    trial_times.append(elapsed)
                
                except Exception as e:
                    print(f"⚠️  Error timing {method_name} at m={m}: {e}")
                    trial_times.append(np.nan)
            
            # Average time across trials
            avg_time = np.nanmean(trial_times) if trial_times else np.nan
            times.append(avg_time)
        
        timing_dict[method_name] = {'steps': steps_range, 'times_ms': times}
        
        # Plot line
        ax.plot(
            steps_range, times,
            color=colors[method_name],
            linestyle=linestyles[method_name],
            marker=markers[method_name],
            linewidth=linewidth,
            markersize=markersize,
            label=method_name,
            alpha=0.85
        )
    
    # Configure axes
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Steps (log scale)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Computation Time (ms, log scale)', fontsize=13, fontweight='bold')
    else:
        ax.set_xlabel('Number of Steps', fontsize=13, fontweight='bold')
        ax.set_ylabel('Computation Time (milliseconds)', fontsize=13, fontweight='bold')
    
    # Title
    if title is None:
        title = f"Computational Performance Comparison (Benchmark: {benchmark_model})"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    
    # Grid
    if grid:
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    # Formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    plt.show()
    
    return fig, ax, timing_dict

def plot_beta_vs_mc_portfolio(
    option,
    S0,
    correlation,
    volatilities,
    mc_benchmark_paths=1_000_000,
    beta_steps_range=None,
    is_calc_delta = False,
    figsize=(14, 6),
    title=None,
    save_path=None,
    dpi=300,
    error_type='absolute',
    scale='log'
):
    """
    Compare Beta-Binomial pricing with Monte Carlo benchmark for portfolios.
    
    Shows convergence of Beta-Binomial to MC as the number of steps increases.
    MC benchmark uses 1M simulations for high precision reference.
    
    Parameters:
    -----------
    option : Option object
        The portfolio option (VanillaOption, DigitalOption, BarrierOption, etc.)
    
    S0 : list or ndarray
        Initial asset prices [S1_0, S2_0, ..., Sn_0]
    
    correlation : ndarray
        Correlation matrix (n x n), symmetric positive definite
    
    volatilities : list or ndarray
        Volatility vector [σ1, σ2, ..., σn]
    
    mc_benchmark_paths : int
        MC simulations for benchmark (default 1M for high precision)
    
    beta_steps_range : list, optional
        Beta model steps to test (default: log-spaced from 10 to 5000)
    
    figsize : tuple
        Figure size (default: 14x6)
    
    title : str, optional
        Plot title
    
    save_path : str, optional
        Path to save figure
    
    dpi : int
        Resolution (default 300)
    
    error_type : str, optional
        Type of error to display: 'absolute', 'relative', or 'difference' (default 'absolute')
        - 'absolute': |price - benchmark|
        - 'relative': |price - benchmark| / |benchmark|
        - 'difference': price - benchmark (can be negative, best with scale='linear')
    
    scale : str, optional
        Scale for axes: 'log' (default) or 'linear'
        - 'log': logarithmic scale (better for absolute/relative errors)
        - 'linear': linear scale (better for difference errors centered at 0)
    
    Returns:
    --------
    tuple : (fig, ax, results_dict)
        - fig: matplotlib figure
        - ax: matplotlib axes
        - results_dict: contains mc_benchmark, beta_steps, beta_prices, beta_errors
    """
    if beta_steps_range is None:
        beta_steps_range = np.arange(1_000, 10_000, 1_000)#[10, 25, 50, 100, 200, 500, 1000, 2000]
    
    # Compute MC benchmark (high precision reference)
    print(f"Computing MC Benchmark ({mc_benchmark_paths} paths)...")
    mc_benchmark = portfolio_monte_carlo_price(
        option=option,
        S0=S0,
        correlation=correlation,
        volatilities=volatilities,
        m=1,
        N=mc_benchmark_paths,
        calc_delta=is_calc_delta
    )
    print(f"✅ MC Benchmark: {mc_benchmark:.8f}")
    
    # Compute Beta-Binomial prices for increasing steps
    beta_prices = []
    beta_errors = []
    
    weights = np.array(len(volatilities)*[1/len(volatilities)])
    S0_portfolio = np.dot(weights, S0)
    cov_matrix = np.diag(volatilities)@correlation@np.diag(volatilities)
    var_b = weights@cov_matrix@weights
    vol_portfolio = np.sqrt(var_b)
    
    print("\nComputing Beta-Binomial prices...")
    for steps in beta_steps_range:
        # Create equivalent single-asset option for beta pricing
        # Handle BarrierOption (requires barrier parameter in constructor)
        if hasattr(option, 'barrier'):
            portfolio_option = option.__class__(
                s0=S0_portfolio,
                r=option.r,
                T=option.T,
                K=option.K,
                barrier=option.barrier,
                vol=vol_portfolio,
                call=option.call if hasattr(option, 'call') else True,
                barrier_direction=option.barrier_direction if hasattr(option, 'barrier_direction') else 'up',
                barrier_type=option.barrier_type if hasattr(option, 'barrier_type') else 'out'
            )
        else:
            # Standard VanillaOption, DigitalOption, etc.
            portfolio_option = option.__class__(
                s0=S0_portfolio,
                r=option.r,
                T=option.T,
                K=option.K,
                vol=vol_portfolio,
                call=option.call if hasattr(option, 'call') else True
            )
        
        # Copy digital type if it exists (for DigitalOption)
        if hasattr(option, 'digital_type'):
            portfolio_option.digital_type = option.digital_type
        
        try:
            price = binomial_beta_price(
                portfolio_option,
                model_name='garcia',
                m=steps,
                calc_delta=is_calc_delta
            )
            beta_prices.append(price)
            if error_type == 'absolute':
                error = abs(price - mc_benchmark)
            elif error_type == 'relative':
                error = abs(price - mc_benchmark) / abs(mc_benchmark)
            else:
                error = price-mc_benchmark
            beta_errors.append(error)
            #print(f"  Steps={steps:4d}: Price={price:.8f}, Error={error:.8e}")
        except Exception as e:
            print(f"  Error at steps={steps}: {e}")
            beta_prices.append(np.nan)
            beta_errors.append(np.nan)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # Plot convergence errors
    ax.plot(
        beta_steps_range, beta_errors,
        color='#1f77b4', linestyle='-', marker='o',
        linewidth=2.5, markersize=8, label='Beta-Binomial Error',
        alpha=0.85
    )
    
    # Horizontal reference line at benchmark (y=0 for difference, y=0 for others too)
    ax.axhline(y=0, color='#d62728', linestyle='--', linewidth=2.5, 
               label='MC Benchmark (zero line)', alpha=0.8)
    
    # Configure axes based on scale parameter
    if scale.lower() == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
        ylabel_suffix = '(log scale)'
    elif scale.lower() == 'linear':
        # Linear scale - no special configuration needed
        ylabel_suffix = '(linear scale)'
    else:
        raise ValueError(f"scale must be 'log' or 'linear', got '{scale}'")
    
    ax.set_xlabel('Number of Beta-Binomial Steps', fontsize=13, fontweight='bold')
    
    # Y-axis label based on error type
    if error_type == 'absolute':
        ylabel = f'Absolute Error vs MC Benchmark {ylabel_suffix}'
    elif error_type == 'relative':
        ylabel = f'Relative Error (%) {ylabel_suffix}'
    else:  # 'difference'
        ylabel = f'Price Difference: Beta - MC {ylabel_suffix}'
    
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    
    # Title
    if title is None:
        opt_type = option.__class__.__name__
        error_label = 'Difference' if error_type == 'difference' else 'Error'
        title = f"Portfolio {opt_type}: Beta-Binomial Convergence ({error_label}, {scale} scale)"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Legend and grid
    ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    
    if scale.lower() == 'log':
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1, which='both')
    else:
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    # Formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    plt.show()
    
    results = {
        'mc_benchmark': mc_benchmark,
        'beta_steps': beta_steps_range,
        'beta_prices': beta_prices,
        'beta_errors': beta_errors
    }
    
    return fig, ax, results

def plot_timing_beta_vs_mc_portfolio(
    option,
    S0,
    correlation,
    volatilities,
    beta_steps_range=None,
    mc_path_range=None,
    figsize=(15, 6),
    title=None,
    save_path=None,
    dpi=300
):
    """
    Compare computational time: Beta-Binomial (constant) vs MC (explosive).
    
    Demonstrates that:
    - Beta-Binomial: O(m^2) complexity, essentially constant for practical step counts
    - Monte Carlo: O(N) complexity, grows linearly with number of simulations
    
    Parameters:
    -----------
    option : Option object
        The portfolio option
    
    S0 : list or ndarray
        Initial asset prices
    
    correlation : ndarray
        Correlation matrix
    
    volatilities : list or ndarray
        Volatility vector
    
    beta_steps_range : list, optional
        Beta model steps to test (default: [10, 50, 100, 500, 1000, 5000])
    
    mc_path_range : list, optional
        MC simulations to test (default: [10K, 50K, 100K, 500K, 1M])
    
    figsize : tuple
        Figure size (default: 15x6)
    
    title : str, optional
        Plot title
    
    save_path : str, optional
        Path to save figure
    
    dpi : int
        Resolution (default 300)
    
    Returns:
    --------
    tuple : (fig, (ax_left, ax_right), results_dict)
        - fig: matplotlib figure
        - (ax_left, ax_right): linear and log scale axes
        - results_dict: contains timing data
    """
    if beta_steps_range is None:
        beta_steps_range = [10, 50, 100, 500, 1000, 5000]
    
    if mc_path_range is None:
        mc_path_range = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    
    # ============ BETA-BINOMIAL TIMING ============
    print("Measuring Beta-Binomial computational time...")
    beta_times = []
    weights = np.array(len(volatilities)*[1/len(volatilities)])
    S0_portfolio = np.dot(weights, S0)
    cov_matrix = np.diag(volatilities)@correlation@np.diag(volatilities)
    var_b = weights@cov_matrix@weights
    vol_portfolio = np.sqrt(var_b)
    
    for steps in beta_steps_range:
        portfolio_option = option.__class__(
            s0=S0_portfolio, r=option.r, T=option.T, K=option.K,
            vol=vol_portfolio, call=option.call if hasattr(option, 'call') else True
        )
        if hasattr(option, 'barrier'):
            portfolio_option.barrier = option.barrier
            portfolio_option.barrier_direction = option.barrier_direction
            portfolio_option.barrier_type = option.barrier_type
        if hasattr(option, 'digital_type'):
            portfolio_option.digital_type = option.digital_type
        
        start = time.perf_counter()
        binomial_beta_price(portfolio_option, model_name='garcia', m=steps)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        beta_times.append(elapsed)
        print(f"  Steps={steps:5d}: {elapsed:.3f} ms")
    
    # ============ MONTE CARLO TIMING ============
    print("\nMeasuring Monte Carlo computational time...")
    mc_times = []
    
    for paths in mc_path_range:
        start = time.perf_counter()
        portfolio_monte_carlo_price(
            option=option,
            S0=S0,
            correlation=correlation,
            volatilities=volatilities,
            m=1,
            N=paths
        )
        elapsed = (time.perf_counter() - start) * 1000  # ms
        mc_times.append(elapsed)
        print(f"  Paths={paths:7d}: {elapsed:.3f} ms")
    
    # ============ CREATE PLOTS ============
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=100)
    
    # LEFT PLOT: Linear scale
    ax1.plot(beta_steps_range, beta_times, color='#1f77b4', linestyle='-', 
             marker='o', linewidth=2.5, markersize=8, label='Beta-Binomial', alpha=0.85)
    ax1.plot(mc_path_range, mc_times, color='#ff7f0e', linestyle='-', 
             marker='s', linewidth=2.5, markersize=8, label='Monte Carlo', alpha=0.85)
    
    ax1.set_xlabel('MC Paths', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Computational Time Comparison (Linear Scale)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(labelsize=11)
    
    # RIGHT PLOT: Log scale
    ax2.loglog(beta_steps_range, beta_times, color='#1f77b4', linestyle='-', 
               marker='o', linewidth=2.5, markersize=8, label='Beta-Binomial', alpha=0.85)
    ax2.loglog(mc_path_range, mc_times, color='#ff7f0e', linestyle='-', 
               marker='s', linewidth=2.5, markersize=8, label='Monte Carlo', alpha=0.85)
    
    ax2.set_xlabel('Beta Steps / MC Paths (log scale)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (ms, log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Computational Time Comparison (Log Scale)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=1, which='both')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(labelsize=11)
    
    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    plt.show()
    
    results = {
        'beta_steps': beta_steps_range,
        'beta_times_ms': beta_times,
        'mc_paths': mc_path_range,
        'mc_times_ms': mc_times
    }
    
    return fig, (ax1, ax2), results