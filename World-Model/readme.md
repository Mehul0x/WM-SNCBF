# Pendulum Dataset Generator

This workspace generates a labeled dataset using the OpenAI Gymnasium Pendulum environment. Each data point consists of:
- An image of the current state.
- The category of the state ("safe", "unsafe", or "buffer").
- The action applied at the current state.
- The image file name of the next state after this action.
- The observation (state vector) of the current state.

## Data Labeling Criteria

- **Safe**:  
  If both `theta` and `theta_dot` are within ±π/12:
  ```
  -π/12 < theta < π/12
  -π/12 < theta_dot < π/12
  ```
- **Unsafe**:  
  If either `theta` or `theta_dot` exceeds ±5π/12:
  ```
  theta > 5π/12 or theta < -5π/12
  theta_dot > 5π/12 or theta_dot < -5π/12
  ```
- **Buffer**:  
  All other states.

## Contents

- `generate_pendulum_dataset.py` — Main script to generate and save the dataset.
- `requirements.txt` — Python dependencies.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the dataset generator:
   ```bash
   python generate_pendulum_dataset.py
   ```

3. Generated images and labels will be saved in the `dataset/` directory.

## Notes

- Each row in `labels.csv` contains the image filename, the category, the action applied, the filename of the next state image, and the current observation.
- You can modify the number of samples or output directory in the script as needed.