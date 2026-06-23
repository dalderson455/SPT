import numpy as np
import pandas as pd
from scipy.stats import linregress, gaussian_kde
import matplotlib.pyplot as plt

def general_linear_stoich(spots_df, i_single, n_frames=3, min_frames=3, threshold=150, n_frames_after_laser_on=10, laser_on_frame=None):
    if np.isnan(i_single) or i_single <= 0:
        raise ValueError("i_single must be a positive scalar. Run find_isingle first.")
        
    if 'traj_id' not in spots_df.columns:
        return np.nan, []
        
    raw_results = []
    
    for traj_id, group in spots_df.groupby('traj_id'):
        if traj_id == 0:
            continue
            
        group = group.sort_values('frame')
        
        # Determine laser on frame
        if laser_on_frame is None:
            actual_laser_on_frame = spots_df['frame'].min() 
        else:
            actual_laser_on_frame = laser_on_frame
        
        start_frame = group['frame'].min()
        if start_frame < actual_laser_on_frame or start_frame > actual_laser_on_frame + n_frames_after_laser_on:
            continue
            
        n_available = len(group)
        n_fit = min(n_frames, n_available)
        if n_fit < min_frames:
            continue
            
        rel_time = group['frame'].values[:n_fit] - actual_laser_on_frame
        intensity = group['intensity'].values[:n_fit]
        
        slope, intercept, r_value, p_value, std_err = linregress(rel_time, intensity)
        
        val = intercept / i_single
        if 0 < val < threshold and slope <= 0:
            raw_results.append(val)
            
    stoich_all = np.array(raw_results)
    if len(stoich_all) > 0:
        return np.mean(stoich_all), stoich_all
    return np.nan, stoich_all

def stoich_analyser(spots_df, i_single, kde_bandwidth=5, show_plot=False, save_path=None, **kwargs):
    mean_stoich, stoich_all = general_linear_stoich(spots_df, i_single, **kwargs)
    
    if len(stoich_all) == 0:
        print("Warning: No valid trajectories found for stoichiometry analysis.")
        return {'mean': np.nan, 'median': np.nan, 'mode': np.nan, 'sd': np.nan, 'sem': np.nan, 'stoich_all': []}
        
    mean_val = np.mean(stoich_all)
    median_val = np.median(stoich_all)
    sd_val = np.std(stoich_all)
    sem_val = sd_val / np.sqrt(len(stoich_all))
    
    try:
        kde = gaussian_kde(stoich_all, bw_method=kde_bandwidth / np.std(stoich_all, ddof=1))
        xi = np.linspace(0, np.max(stoich_all) + 10, 500)
        f = kde(xi)
        mode_val = xi[np.argmax(f)]
    except (ValueError, np.linalg.LinAlgError):
        mode_val = np.nan
        
    print(f"Stoichiometry (n={len(stoich_all)}): Mean {mean_val:.2f} | Median {median_val:.2f} | Mode {mode_val:.2f} | SD {sd_val:.2f}")
    
    if show_plot or save_path:
        plt.figure(figsize=(8, 5))
        try:
            plt.plot(xi, f, 'b-', linewidth=2, label='KDE')
            yl = np.max(f) * 1.1
        except NameError:
            plt.hist(stoich_all, bins=20, density=True, color='b', alpha=0.5)
            yl = plt.gca().get_ylim()[1]
            
        plt.vlines(mean_val, 0, yl, colors='r', linestyles='--', label=f'Mean={mean_val:.2f}')
        plt.vlines(median_val, 0, yl, colors='g', linestyles='--', label=f'Median={median_val:.2f}')
        if not np.isnan(mode_val):
            plt.vlines(mode_val, 0, yl, colors='m', linestyles='--', label=f'Mode={mode_val:.2f}')
            
        plt.xlabel('Stoichiometry')
        plt.ylabel('Probability Density')
        plt.title(f'Stoichiometry Distribution (n={len(stoich_all)})')
        plt.legend()
        try:
            plt.xlim(0, np.max(xi))
        except NameError:
            plt.xlim(0, np.max(stoich_all) * 1.1)
        plt.ylim(0, yl)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        if show_plot:
            plt.show()
        else:
            plt.close()
        
    return {
        'mean': mean_val, 'median': median_val, 'mode': mode_val, 
        'sd': sd_val, 'sem': sem_val, 'stoich_all': stoich_all
    }

if __name__ == "__main__":
    pass
