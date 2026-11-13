# Joint completion for regression of RMST curves

This repository has the code and the data of the paper. Run the commands
from the main folder.

## Files

- `code/run_joint_rmst_simulation.py`: simulation study
- `code/run_colon_real_data.py`: colon cancer application
- `code/prepare_simulation_results.py`: simulation tables and figures
- `code/prepare_boundary_inference_results.py`: inference tables
- `code/prepare_real_data_results.py`: application table and figures
- `code/simulated_subject_shape_diagnostic.py`: shape check at the simulated subjects
- `code/export_colon_trial_data.R`: exports the data from R
- `colon_trial_raw.csv`: colon data from the R package `survival`

## Requirements

Python 3.11 with NumPy, pandas, and SciPy.

```text
python -m pip install -r requirements.txt
```

## Simulation

```text
python code/run_joint_rmst_simulation.py --design main --reps 500 --jobs 4 --support-rate 0.60 --support-scale 0.5 --boundary-inference --out simulation_output
python code/prepare_simulation_results.py simulation_output
python code/prepare_boundary_inference_results.py simulation_output --combined-main
```

The seed is 20260715. For a short test run:

```text
python code/run_joint_rmst_simulation.py --pilot
```

## Application

```text
python code/run_colon_real_data.py
python code/prepare_real_data_results.py
```

The seed is 731947. The results are saved in `real_data_output/`.

## Data

The data come from the R package `survival`
(https://CRAN.R-project.org/package=survival).
