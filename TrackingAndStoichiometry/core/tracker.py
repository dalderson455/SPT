import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_laplace, maximum_filter
from skimage.feature import blob_log
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.spatial.distance import cdist
import tifffile
import time
import warnings
from numba import jit
import concurrent.futures

def set_default_params(p=None):
    if p is None:
        p = {}
    defaults = {
        'disk_radius': 5,
        'subarray_halfwidth': 8,
        'inner_circle_radius': 5,
        'gauss_mask_sigma': 2,
        'error_set': 0.05,
        'SNR_min': 0.4,
        'sigmaFit_min': 0,
        'sigmaFit_max': 5, # matching inner_circle_radius default
        'guess_sigma_Fit': 3,
        'd_01_max': 5,
        'Iratio_01_min': 0.5,
        'Iratio_01_max': 3,
        'gap_frames': 0,
        'satPixelVal': 10**10,
        'LocalisationMethod': 'iterative', # 'iterative' or 'quadratic'
        'DetectionMethod': 'log',
        'log_thresh_multiplier': 5
    }
    for k, v in defaults.items():
        if k not in p:
            p[k] = v
    return p

from scipy.ndimage import correlate, maximum_filter

def fspecial_log(hsize, sigma):
    x, y = np.meshgrid(np.arange(-hsize//2 + 1, hsize//2 + 1),
                       np.arange(-hsize//2 + 1, hsize//2 + 1))
    h = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    h = h * (x**2 + y**2 - 2 * sigma**2) / (sigma**4)
    return h - np.sum(h) / (hsize**2)

def detect_log(frame, p):
    frame = frame.astype(np.float64)
    sigma = max(1, p['disk_radius'] / 2.5)
    
    kSize = int(max(5, np.ceil(6 * sigma)))
    if kSize % 2 == 0:
        kSize += 1
        
    h = fspecial_log(kSize, sigma)
    # MATLAB imfilter with 'replicate' uses correlation
    filtered = -correlate(frame, h, mode='nearest')
    
    medV = np.median(filtered)
    stdV = np.std(filtered)
    bg = filtered[filtered < medV + 2 * stdV]
    
    if len(bg) > 0:
        thresh = np.median(bg) + p.get('log_thresh_multiplier', 5) * np.std(bg)
    else:
        thresh = medV + p.get('log_thresh_multiplier', 5) * stdV
        
    local_max = maximum_filter(filtered, size=3) == filtered
    y_est, x_est = np.nonzero(local_max & (filtered > thresh))
    
    b = int(np.ceil(p['disk_radius']))
    valid = (y_est >= b) & (y_est < frame.shape[0] - b) & (x_est >= b) & (x_est < frame.shape[1] - b)
    
    return y_est[valid], x_est[valid]

@jit(nopython=True, cache=True)
def loc_iterative_core(I, x_est, y_est, d, r, sig, error_set):
    h, w = I.shape
    
    Xs = np.zeros((h, w), dtype=np.float64)
    Ys = np.zeros((h, w), dtype=np.float64)
    
    start_x = x_est - d
    start_y = y_est - d
    
    for i in range(h):
        for j in range(w):
            Xs[i, j] = start_x + j
            Ys[i, j] = start_y + i
            
    xc = float(x_est)
    yc = float(y_est)
    clip = 0
    noConv = 0
    
    Ibg = 0.0
    Isp = 0.0
    bg_std = 0.0
    m_pix = 0
    
    for k in range(300):
        inner_count = 0
        outer_count = 0
        I_outer_sum = 0.0
        I_outer_sum_sq = 0.0
        
        for i in range(h):
            for j in range(w):
                distSq = (Xs[i, j] - xc)**2 + (Ys[i, j] - yc)**2
                if distSq <= r**2:
                    inner_count += 1
                else:
                    outer_count += 1
                    val = I[i, j]
                    I_outer_sum += val
                    I_outer_sum_sq += val**2
                    
        if inner_count == 0 or outer_count == 0:
            clip = 1
            noConv = 1
            break
            
        Ibg = I_outer_sum / outer_count
        bg_var = (I_outer_sum_sq / outer_count) - Ibg**2
        if bg_var > 0:
            bg_std = np.sqrt(bg_var)
        else:
            bg_std = 0.0
            
        Isp = 0.0
        m_pix = inner_count
        
        mask_sum = 0.0
        I3_sum = 0.0
        xn_num = 0.0
        yn_num = 0.0
        
        for i in range(h):
            for j in range(w):
                distSq = (Xs[i, j] - xc)**2 + (Ys[i, j] - yc)**2
                if distSq <= r**2:
                    I2_val = I[i, j] - Ibg
                    Isp += I2_val
                    mask_val = np.exp(-distSq / (2.0 * sig**2))
                    mask_sum += mask_val
                    
        if mask_sum == 0.0:
            noConv = 1
            break
            
        for i in range(h):
            for j in range(w):
                distSq = (Xs[i, j] - xc)**2 + (Ys[i, j] - yc)**2
                if distSq <= r**2:
                    I2_val = I[i, j] - Ibg
                    mask_val = np.exp(-distSq / (2.0 * sig**2))
                    I3_val = I2_val * (mask_val / mask_sum)
                    I3_sum += I3_val
                    xn_num += I3_val * Xs[i, j]
                    yn_num += I3_val * Ys[i, j]
                    
        if I3_sum == 0.0:
            noConv = 1
            break
            
        xn = xn_num / I3_sum
        yn = yn_num / I3_sum
        
        if np.sqrt((xn - xc)**2 + (yn - yc)**2) <= error_set and k >= 4:
            xc = xn
            yc = yn
            break
            
        xc = xn
        yc = yn
        if abs(xc - x_est) > (d - r + 1) or abs(yc - y_est) > (d - r + 1):
            clip = 1
            break
            
        if k == 299:
            noConv = 1
            
    return xc, yc, clip, Ibg, Isp, bg_std, m_pix, noConv

def loc_iterative(img, x_est, y_est, p):
    d = p['subarray_halfwidth']
    r = p['inner_circle_radius']
    sig = p['gauss_mask_sigma']
    
    h, w = img.shape
    x_est = int(np.clip(np.round(x_est), d, w - d - 1))
    y_est = int(np.clip(np.round(y_est), d, h - d - 1))
    
    I = img[y_est-d:y_est+d+1, x_est-d:x_est+d+1].astype(np.float64)
    
    xc, yc, clip, Ibg, Isp, bg_std, m_pix, noConv = loc_iterative_core(
        I, x_est, y_est, d, r, sig, p['error_set']
    )
    
    return xc, yc, clip, Ibg, Isp, bg_std, m_pix, noConv



def link_lap(spots_df, p):
    if spots_df.empty:
        return spots_df
        
    frames = np.sort(spots_df['frame'].unique())
    spots_df['traj_id'] = 0
    next_id = 1
    
    # Dictionary mapping spot index to traj_id
    id_map = {}
    
    for i in range(1, len(frames)):
        idx1 = spots_df.index[spots_df['frame'] == frames[i-1]].tolist()
        idx2 = spots_df.index[spots_df['frame'] == frames[i]].tolist()
        
        if not idx1 or not idx2:
            continue
            
        pos1 = spots_df.loc[idx1, ['x', 'y']].values
        pos2 = spots_df.loc[idx2, ['x', 'y']].values
        
        D = cdist(pos2, pos1)
        D[D > p['d_01_max']] = np.inf
        
        I1 = spots_df.loc[idx1, 'intensity'].values
        I2 = spots_df.loc[idx2, 'intensity'].values
        
        # Intensity ratio gating
        for a in range(len(I2)):
            for b in range(len(I1)):
                if D[a, b] != np.inf:
                    ratio = I1[b] / (I2[a] + 1e-9)
                    if ratio < p['Iratio_01_min'] * 0.5 or ratio > p['Iratio_01_max'] * 2:
                        D[a, b] = np.inf
                        
        LARGE_COST = 1e6
        cost_matrix = D.copy()
        cost_matrix[np.isinf(cost_matrix)] = LARGE_COST
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            if D[r, c] < np.inf:
                spot1_idx = idx1[c]
                spot2_idx = idx2[r]
                
                if id_map.get(spot1_idx, 0) == 0:
                    id_map[spot1_idx] = next_id
                    next_id += 1
                id_map[spot2_idx] = id_map[spot1_idx]
                
    spots_df['traj_id'] = spots_df.index.map(lambda x: id_map.get(x, 0))
    unlinked = spots_df['traj_id'] == 0
    spots_df.loc[unlinked, 'traj_id'] = np.arange(next_id, next_id + unlinked.sum())
    
    return spots_df

def process_frame_task(frame, frame_idx, p):
    if np.max(frame) > p['satPixelVal']:
        return []
        
    y_est, x_est = detect_log(frame, p)
    
    valid_spots = []
    for j in range(len(x_est)):
        xc, yc, clip, Ibg, Isp, bg_std, m_pix, noConv = loc_iterative(frame, x_est[j], y_est[j], p)
        
        snr = Isp / (bg_std * m_pix + 1e-9)
        if noConv or clip or snr <= p['SNR_min']:
            continue
            
        sdx, sdy = p['guess_sigma_Fit'], p['guess_sigma_Fit']
        
        valid_spots.append({
            'x': xc, 'y': yc, 'frame': frame_idx,
            'intensity': Isp, 'bg': Ibg,
            'sigma_x': sdx, 'sigma_y': sdy,
            'snr': snr
        })
    return valid_spots

def track_image(image_data, p=None):
    p = set_default_params(p)
    num_frames = image_data.shape[0] if image_data.ndim == 3 else 1
    
    all_spots = []
    
    t0 = time.time()
    
    if num_frames > 1:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i in range(num_frames):
                frame = image_data[i]
                futures.append(executor.submit(process_frame_task, frame, i + 1, p))
            
            for future in concurrent.futures.as_completed(futures):
                spots = future.result()
                all_spots.extend(spots)
    else:
        frame = image_data
        spots = process_frame_task(frame, 1, p)
        all_spots.extend(spots)
        
    spots_df = pd.DataFrame(all_spots)
    
    if not spots_df.empty:
        spots_df = spots_df.sort_values(by=['frame']).reset_index(drop=True)
        spots_df = link_lap(spots_df, p)
        
    print(f"Tracking complete in {time.time() - t0:.2f} seconds. Tracks found: {spots_df['traj_id'].nunique() if not spots_df.empty else 0}")
    
    return spots_df

if __name__ == "__main__":
    pass
