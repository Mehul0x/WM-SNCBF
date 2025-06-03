# - An image of the current state.
# - The category of the state ("safe", "unsafe", or "buffer").
# - The action applied at the current state.
# - The image file name of the next state after this action.
# - The observation (state vector) of the current state.

# ## Data Labeling Criteria

# - **Safe**:  
#   If both `theta` and `theta_dot` are within ±π/12:
#   ```
#   -π/12 < theta < π/12
#   -π/12 < theta_dot < π/12
#   ```
# - **Unsafe**:  
#   If either `theta` or `theta_dot` exceeds ±5π/12:
#   ```
#   theta > 5π/12 or theta < -5π/12
#   theta_dot > 5π/12 or theta_dot < -5π/12
#   ```
# - **Buffer**:  
#   All other states.

# - Each row in `labels.csv` contains the image filename, the category, the action applied, the filename of the next state image, and the current observation.
# - You can modify the number of samples or output directory in the script as needed.

import gymnasium as gym
import numpy as np
import math
import os
import csv
from PIL import Image

# Output directories
os.makedirs('World-Model/images', exist_ok=True)

def categorize(theta, theta_dot):
    if (-math.pi/12 < theta < math.pi/12) and (-math.pi/12 < theta_dot < math.pi/12):
        return "safe"
    elif (theta > math.pi*(5/12)) or (theta < -math.pi*(5/12)) or (theta_dot > math.pi*(5/12)) or (theta_dot < -math.pi*(5/12)):
        return "unsafe"
    else:
        return "buffer"

def get_theta_theta_dot(obs):
    # Pendulum: [cos(theta), sin(theta), theta_dot]p
    cos_theta, sin_theta, theta_dot = obs
    theta = math.atan2(sin_theta, cos_theta)
    return theta, theta_dot

def save_img(img_array, img_name):
    img = Image.fromarray(img_array)
    img.save(os.path.join('World-Model/images', img_name))

def main():
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    obs, _ = env.reset()
    total_images = 0
    data = []
    step_num = 0

    # 1. Random actions for 5000 images
    num_random = 2500
    while total_images < num_random:
        img_array = env.render()
        curr_img_name = f"state_{step_num}.png"
        save_img(img_array, curr_img_name)

        theta, theta_dot = get_theta_theta_dot(obs)
        category = categorize(theta, theta_dot)

        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        next_img_array = env.render()
        next_img_name = f"state_{step_num+1}.png"
        save_img(next_img_array, next_img_name)

        data.append({
            "state_image": curr_img_name,
            "category": category,
            "action": action.tolist(),
            "next_state_image": next_img_name,
            "observation": obs.tolist()
        })

        obs = next_obs
        step_num += 1
        total_images += 1

        if terminated or truncated:
            obs, _ = env.reset()

    # 2. P controller for next 600 images
    num_p_control = 2000
    Kp = 2.0  # Proportional gain, tune as needed

    for _ in range(num_p_control):
        theta_init = np.random.uniform(-np.pi/12, np.pi/12)
        theta_dot_init = np.random.uniform(-0.2, 0.2)
        env.unwrapped.state = np.array([theta_init, theta_dot_init])
        img_array = env.render()
        curr_img_name = f"state_{step_num}.png"
        save_img(img_array, curr_img_name)

        theta, theta_dot = get_theta_theta_dot(obs)
        category = categorize(theta, theta_dot)

        # P controller: action = Kp * theta
        action = np.array([-Kp * theta])
        action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, _, terminated, truncated, _ = env.step(action)
        next_img_array = env.render()
        next_img_name = f"state_{step_num+1}.png"
        save_img(next_img_array, next_img_name)

        data.append({
            "state_image": curr_img_name,
            "category": category,
            "action": action.tolist(),
            "next_state_image": next_img_name,
            "observation": obs.tolist()
        })

        obs = next_obs
        step_num += 1

        if terminated or truncated:
            obs, _ = env.reset(seed=123)
            

    # Save metadata as CSV
    with open('World-Model/pendulum_dataset.csv', 'w', newline='') as csvfile:
        fieldnames = ["state_image", "category", "action", "next_state_image", "observation"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f"Dataset generation complete: {len(data)} images (5000 random, 600 P controller).")
    env.close()

if __name__ == "__main__":
    main()