# PythonSPT

**Note: This project is currently under active development.**

A lightweight Python implementation of single-molecule tracking and stoichiometry analysis. It uses Laplacian of Gaussian (LoG) detection, iterative localisation, and LAP tracking to resolve sub-pixel molecular trajectories, followed by step-wise photobleaching analysis to estimate molecular stoichiometry.

## Requirements

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage

Use `run_pipeline.py` to analyse a single video (with an optional binary mask).

**Command:**
```bash
python3 run_pipeline.py /path/to/Video.tif --mask /path/to/Mask.tif --save-csv tracking_results.csv
```

**Arguments:**
- `image` (Required): Path to the input video (.tif, .nd2, etc.).
- `--mask` (Optional): Path to a binary mask TIFF. Tracks originating outside the mask will be ignored.
- `--save-csv` (Optional): Saves the raw Pandas dataframe of localised trajectories to the specified CSV file.
- `--skip-stoichiometry` (Optional): Skips the iSingle and Stoichiometry steps.
- `--show-plots` (Optional): Displays the iSingle and Stoichiometry KDE plots (they are also auto-saved to `Output_Figures/`).
- `--config` (Optional): Path to a custom `config.yaml` file.

### CSV Output Format

If you provide the `--save-csv` flag, the pipeline will export the raw trajectories. The columns include:

*   `x` and `y`: Sub-pixel spatial coordinates.
*   `frame`: The video frame number.
*   `intensity`: The total background-corrected integrated intensity of the molecule
*   `bg`: The local background intensity.
*   `sigma_x` and `sigma_y`: Gaussian width estimations.
*   `snr`: Signal-to-Noise Ratio.
*   `traj_id`: The unique trajectory identifier connecting the spots across frames.

## Architecture

* `TrackingAndStoichiometry/core/tracker.py`: Contains the core detection (LoG) and iterative localisation logic.
* `TrackingAndStoichiometry/core/isingle.py`: Extracts the consensus single-fluorophore intensity from the dataset using pairwise distance distribution, Chung-Kennedy filtering and bayesian changepoint analysis.
* `TrackingAndStoichiometry/core/stoichiometry.py`: Fits lines to the initial photobleaching steps of each valid trajectory to extrapolate the intensity at the first illumination frame.
