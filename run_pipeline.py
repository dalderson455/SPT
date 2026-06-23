import os
import argparse
import yaml
import numpy as np
from aicsimageio import AICSImage
import tifffile
from TrackingAndStoichiometry.core import track_image, find_isingle, stoich_analyser

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {'tracking': {}, 'isingle': {}, 'stoichiometry': {}}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run the single-molecule tracking and stoichiometry pipeline on a single image.")
    parser.add_argument("image", help="Path to the input video (.tif, .nd2, .czi, .lif)")
    parser.add_argument("--mask", help="Path to the binary mask TIFF (optional)", default=None)
    parser.add_argument("--save-csv", help="Path to save the tracking dataframe as CSV (optional)", default=None)
    parser.add_argument("--show-plots", action="store_true", help="Display the iSingle and Stoichiometry plots")
    parser.add_argument("--skip-stoichiometry", action="store_true", help="Skip iSingle and Stoichiometry steps")
    parser.add_argument("--config", help="Path to config.yaml", default="config.yaml")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    print(f"\n--- Processing {os.path.basename(args.image)} ---")
    try:
        if args.image.lower().endswith('.tif') or args.image.lower().endswith('.tiff'):
            image = tifffile.imread(args.image, key=slice(None))
        else:
            image = AICSImage(args.image).data.squeeze()
            
        print(f"Loaded image shape: {image.shape}")
        
        mask = None
        if args.mask and os.path.exists(args.mask):
            mask = tifffile.imread(args.mask) > 0
            print(f"Loaded mask shape: {mask.shape}")
    except Exception as e:
        print(f"Error loading image or mask: {e}")
        return

    p = config.get('tracking', {})
    if not p:
        p = {'log_thresh_multiplier': 5.0}
        
    print("Running Tracking...")
    spots_df = track_image(image, p)
    
    if spots_df.empty:
        print("No spots found.")
        return
        
    if mask is not None:
        print(f"Spots before masking: {len(spots_df)}")
        first_spots = spots_df.loc[spots_df.groupby('traj_id')['frame'].idxmin()]
        
        y_int = np.clip(np.round(first_spots['y']).astype(int), 0, mask.shape[0]-1)
        x_int = np.clip(np.round(first_spots['x']).astype(int), 0, mask.shape[1]-1)
        
        valid_traj_mask = mask[y_int, x_int]
        valid_traj_ids = first_spots.loc[valid_traj_mask, 'traj_id']
        
        spots_df = spots_df[spots_df['traj_id'].isin(valid_traj_ids)].copy()
        print(f"Spots after masking: {len(spots_df)}")
        
        if spots_df.empty:
            print("No spots remain after masking.")
            return

    print(f"Found {len(spots_df)} spots and {spots_df['traj_id'].nunique()} trajectories.")
    
    if args.save_csv:
        spots_df.to_csv(args.save_csv, index=False)
        print(f"Saved tracking dataframe to: {args.save_csv}")
    
    if not args.skip_stoichiometry:
        print("Running iSingle Estimation...")
        i_params = config.get('isingle', {})
        
        # Auto-save figures
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        fig_dir = "Output_Figures"
        os.makedirs(fig_dir, exist_ok=True)
        isingle_fig_path = os.path.join(fig_dir, f"{base_name}_iSingle_KDE.png")
        stoich_fig_path = os.path.join(fig_dir, f"{base_name}_Stoichiometry_KDE.png")
        
        i_single, results = find_isingle(
            spots_df, 
            min_track_length=i_params.get('min_track_length', 5),
            penalty_scale=i_params.get('penalty_scale', 1.0),
            ck_window=i_params.get('ck_window', 5),
            ck_power=i_params.get('ck_power', 10),
            show_plot=args.show_plots,
            save_path=isingle_fig_path
        )
        
        if np.isnan(i_single):
            print("Failed to estimate iSingle.")
        else:
            print("Running Stoichiometry Analysis...")
            frame_means = np.mean(image, axis=(1, 2))
            laser_on_frame = int(np.argmax(frame_means))
            print(f"Detected laser on frame: {laser_on_frame}")
            
            s_params = config.get('stoichiometry', {})
            stoich_res = stoich_analyser(
                spots_df, 
                i_single, 
                n_frames=s_params.get('n_frames', 3),
                min_frames=s_params.get('min_frames', 3),
                n_frames_after_laser_on=s_params.get('n_frames_after_laser_on', 10),
                kde_bandwidth=s_params.get('kde_bandwidth', 5),
                threshold=s_params.get('threshold', 150),
                show_plot=args.show_plots, 
                save_path=stoich_fig_path,
                laser_on_frame=laser_on_frame
            )
            print(f"Mean Stoichiometry: {stoich_res['mean']:.2f}")
            print(f"Median Stoichiometry: {stoich_res['median']:.2f}")
            print(f"Plots saved to {fig_dir}/ directory!")


if __name__ == "__main__":
    main()
