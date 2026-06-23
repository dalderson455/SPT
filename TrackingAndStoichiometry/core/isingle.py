import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def extract_valid_traces(spots_df, min_track_length=15):
    traces = []
    if 'traj_id' not in spots_df.columns:
        return traces
        
    for traj_id, group in spots_df.groupby('traj_id'):
        if traj_id == 0 or len(group) < min_track_length:
            continue
            
        group = group.sort_values('frame')
        intensities = group['intensity'].values
        diffs = np.diff(intensities)
        
        # Reject traces with massive jumps
        if len(diffs) > 0 and np.std(diffs) > 0:
            if np.max(diffs) > 3 * np.std(diffs):
                continue
                
        traces.append({
            'intensities': intensities,
            'frames': group['frame'].values,
            'traj_id': traj_id
        })
    return traces

def get_kde_mode(data):
    if len(data) <= 10:
        return np.nan
    bw = np.median(data) / 5.0
    if bw <= 0:
        bw = np.std(data) / 3.0
    if bw <= 0:
        return np.nan
        
    try:
        kde = gaussian_kde(data, bw_method=bw / np.std(data, ddof=1))
        xi = np.linspace(np.min(data), np.max(data), 200)
        f = kde(xi)
        return xi[np.argmax(f)]
    except (ValueError, np.linalg.LinAlgError):
        return np.median(data)

def ck_filter(x, w=5, r=10):
    n = len(x)
    x = np.array(x, dtype=float)
    
    padded = np.zeros(n + 2*w)
    padded[w:n+w] = x
    for i in range(w):
        padded[w-1-i] = x[i]
        padded[n+w+i] = x[n-1-i]
        
    window_matrix = np.zeros((n+w+1, w))
    for i in range(w):
        window_matrix[:, i] = padded[i : n+w+i+1]
        
    window_mean = np.mean(window_matrix, axis=1)
    window_std = np.std(window_matrix, axis=1, ddof=1)
    
    post_mean = window_mean[:n]
    pre_mean = window_mean[w:n+w]
    post_std = window_std[:n]
    pre_std = window_std[w:n+w]
    
    post_var = post_std**2
    pre_var = pre_std**2
    
    eps = 1e-12
    post_var_r = np.power(post_var + eps, r)
    pre_var_r = np.power(pre_var + eps, r)
    
    denom = post_var_r + pre_var_r
    w_pre = post_var_r / denom
    w_post = pre_var_r / denom
    
    filtered = w_post * post_mean + w_pre * pre_mean
    return filtered

def run_ck_method(traces, w=5, r=10):
    steps = []
    for t in traces:
        y = t['intensities']
        if len(y) < 2*w + 1:
            continue
        filt = ck_filter(y, w, r)
        df = np.diff(filt)
        if np.std(df, ddof=1) == 0:
            continue
            
        step_locs = np.where(np.abs(df) > 2 * np.std(df, ddof=1))[0]
        for loc in step_locs:
            pre_idx = slice(max(0, loc - w + 1), loc + 1)
            post_idx = slice(loc + 1, min(len(y), loc + 1 + w))
            
            y_pre = y[pre_idx]
            y_post = y[post_idx]
            if len(y_pre) >= 2 and len(y_post) >= 2:
                steps.append(np.abs(np.mean(y_pre) - np.mean(y_post)))
                
    return get_kde_mode(steps), np.array(steps)

def run_pdd_method(traces):
    all_diffs = []
    for t in traces:
        all_diffs.extend(np.diff(t['intensities']))
        
    all_diffs = np.array(all_diffs)
    if len(all_diffs) > 50:
        cutoff = -np.median(np.abs(all_diffs)) * 0.5
        neg_diffs = -all_diffs[all_diffs < cutoff]
        return get_kde_mode(neg_diffs), neg_diffs
    return np.nan, np.array([])

def bayesian_changepoints(data, penalty_scale=1.0):
    N = len(data)
    if N < 6:
        return []
        
    v_all = np.var(data, ddof=1)
    if v_all == 0:
        return []
        
    logL_null = -0.5 * N * np.log(2 * np.pi * v_all) - 0.5 * np.sum((data - np.mean(data))**2) / v_all
    penalty = penalty_scale * np.log(N)
    
    best_imp = -np.inf
    best_split = 0
    
    for t in range(3, N - 3):
        s1 = data[:t]
        s2 = data[t:]
        
        v1 = max(np.var(s1, ddof=1), v_all * 1e-6)
        v2 = max(np.var(s2, ddof=1), v_all * 1e-6)
        n1, n2 = len(s1), len(s2)
        
        logL_split = -0.5 * n1 * np.log(2 * np.pi * v1) - 0.5 * (n1 - 1) \
                     -0.5 * n2 * np.log(2 * np.pi * v2) - 0.5 * (n2 - 1)
                     
        imp = logL_split - logL_null - penalty
        if imp > best_imp:
            best_imp = imp
            best_split = t
            
    if best_imp > 0 and best_split > 0:
        left = bayesian_changepoints(data[:best_split], penalty_scale)
        right = bayesian_changepoints(data[best_split:], penalty_scale)
        return left + [best_split] + [r + best_split for r in right]
    return []

def run_bcp_method(traces, penalty_scale=1.0):
    steps = []
    for t in traces:
        y = t['intensities']
        if len(y) < 6:
            continue
            
        cps = bayesian_changepoints(y, penalty_scale)
        bounds = [0] + sorted(cps) + [len(y)]
        
        levels = []
        for j in range(len(bounds) - 1):
            levels.append(np.mean(y[bounds[j]:bounds[j+1]]))
            
        new_steps = np.abs(np.diff(levels))
        steps.extend(new_steps[new_steps > 0])
        
    return get_kde_mode(steps), np.array(steps)

def bootstrap_kde_mode(data, n_boot=1000):
    if len(data) < 10:
        return [], (np.nan, np.nan)
        
    boots = []
    n = len(data)
    for _ in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        boots.append(get_kde_mode(sample))
        
    boots = np.array(boots)
    boots = boots[~np.isnan(boots)]
    if len(boots) > 0:
        ci = np.percentile(boots, [2.5, 97.5])
        return boots, ci
    return [], (np.nan, np.nan)

def compute_consensus(results):
    n_boot = 1000
    b_ck, ci_ck = bootstrap_kde_mode(results['CK_steps'], n_boot)
    b_pdd, ci_pdd = bootstrap_kde_mode(results['PDD_steps'], n_boot)
    b_bcp, ci_bcp = bootstrap_kde_mode(results['BCP_steps'], n_boot)
    
    estimates = np.array([results['CK'], results['PDD'], results['BCP']])
    stds = np.array([np.std(b_ck) if len(b_ck) > 0 else 0,
                     np.std(b_pdd) if len(b_pdd) > 0 else 0, 
                     np.std(b_bcp) if len(b_bcp) > 0 else 0])
    
    valid = ~np.isnan(estimates) & (stds > 0)
    
    if not np.any(valid):
        return np.nan, {'ci': (np.nan, np.nan), 'convergence': np.nan}
        
    if np.sum(valid) == 1:
        idx = np.where(valid)[0][0]
        ci_list = [ci_ck, ci_pdd, ci_bcp]
        return estimates[idx], {'ci': ci_list[idx], 'convergence': np.nan}
        
    valid_est = estimates[valid]
    valid_var = stds[valid]**2
    w = 1.0 / valid_var
    w = w / np.sum(w)
    
    i_single = np.sum(w * valid_est)
    i_single_std = np.sqrt(1.0 / np.sum(1.0 / valid_var))
    ci = (i_single - 1.96 * i_single_std, i_single + 1.96 * i_single_std)
    
    return i_single, {'ci': ci, 'convergence': 1.0}

def find_isingle(spots_df, min_track_length=15, penalty_scale=1.0, ck_window=5, ck_power=10, show_plot=False, save_path=None):
    traces = extract_valid_traces(spots_df, min_track_length)
    if not traces:
        return np.nan, {}
        
    est_ck, ck_steps = run_ck_method(traces, ck_window, ck_power)
    est_pdd, pdd_steps = run_pdd_method(traces)
    est_bcp, bcp_steps = run_bcp_method(traces, penalty_scale)
    
    results = {
        'CK': est_ck, 'CK_steps': ck_steps,
        'PDD': est_pdd, 'PDD_steps': pdd_steps,
        'BCP': est_bcp, 'BCP_steps': bcp_steps
    }
    
    i_single, consensus_res = compute_consensus(results)
    results.update(consensus_res)
    
    print(f"========== iSingle Results ==========")
    print(f"CK:     {results['CK']:.1f}")
    print(f"PDD:    {results['PDD']:.1f}")
    print(f"BCP:    {results['BCP']:.1f}")
    print(f"-------------------------------------")
    print(f"Consensus:   {i_single:.1f}")
    print(f"=====================================")
    
    if show_plot or save_path:
        plt.figure(figsize=(10, 6))
        colors = {'CK': 'blue', 'PDD': 'green', 'BCP': 'orange'}
        max_density = 0
        
        for name, steps in zip(['CK', 'PDD', 'BCP'], [ck_steps, pdd_steps, bcp_steps]):
            if len(steps) > 5:
                try:
                    bw = np.median(steps) / 5.0
                    if bw <= 0: bw = np.std(steps) / 3.0
                    if bw > 0:
                        kde = gaussian_kde(steps, bw_method=bw / np.std(steps, ddof=1))
                        xi = np.linspace(0, max(np.max(steps), i_single * 2), 500)
                        f = kde(xi)
                        max_density = max(max_density, np.max(f))
                        plt.plot(xi, f, color=colors[name], linewidth=2, label=f'{name} KDE')
                        plt.hist(steps, bins=40, density=True, color=colors[name], alpha=0.2)
                except Exception:
                    pass
                    
        yl = max_density * 1.1 if max_density > 0 else 1.0
        
        if not np.isnan(i_single):
            plt.vlines(i_single, 0, yl, colors='r', linestyles='-', linewidth=2, label=f'Consensus={i_single:.1f}')
            if 'ci' in consensus_res and not np.isnan(consensus_res['ci'][0]):
                plt.axvspan(consensus_res['ci'][0], consensus_res['ci'][1], color='r', alpha=0.2, label='95% CI')
                
        plt.xlabel('Intensity Step Size')
        plt.ylabel('Density')
        plt.title('iSingle Estimation: Step Size Distributions')
        plt.legend()
        plt.xlim(0, i_single * 3 if not np.isnan(i_single) else None)
        plt.ylim(0, yl)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    return i_single, results

if __name__ == "__main__":
    pass
