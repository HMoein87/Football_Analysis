# Football Analysis

This repository provides tools and scripts for analyzing football (soccer) matches using computer vision techniques. It includes modules for pitch detection, player and ball tracking, team color assignment, and camera movement estimation. By leveraging advanced methods like YOLO, optical flow, and k-means clustering, this project aims to deliver detailed football match analysis.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Contributing](#contributing)

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/HMoein87/Football_Analysis.git

2. **Navigate to the project directory**:

   ```bash
   cd Football_Analysis  

3. **Install the required dependencies**:

   ```bash
   cd Football_Analysis


## Usage

The primary script for running the analysis is `main_notebook.ipynb`.

### Input Data:

Ensure your input data (e.g., match video or frames) is properly formatted and located in the directories expected by the scripts.




## Modules

The repository contains several modules, each addressing a key component of football match analysis:

### 1. `camera_movement_estimator`

- **Purpose**: Estimates the movement of the camera during the match (e.g., pans, tilts, zooms).
- **How It Works**: 
  - Uses **optical flow**, which calculates the motion of pixels between consecutive frames.
  - Stabilizes tracking algorithms by compensating for these movements.
- **Use Case**: Critical for maintaining consistent tracking of players and the ball even with dynamic camera motion.

---

### 2. `pitch_key_point`

- **Purpose**: Detects key points on the football pitch, such as corners and lines, for perspective transformation.
- **How It Works**:
  - Utilizes **YOLO (You Only Look Once)**, a deep learning-based object detection framework, to recognize and localize pitch markings.
  - Identified key points are used to perform homography transformations, enabling a top-down view of the pitch.
- **Use Case**: Essential for aligning the camera's view with the pitch for spatial analysis.

---

### 3. `player_ball_assigner`

- **Purpose**: Links detected players with the ball to understand player-ball interactions.
- **How It Works**: Combines spatial proximity and movement patterns derived from trackers to determine player-ball associations.
- **Use Case**: Useful for determining passing, possession, and key moments in the game.

---

### 4. `team_color_assigner`

- **Purpose**: Automatically assigns detected players to teams based on their jersey colors.
- **How It Works**:
  - Uses **k-means clustering**, an unsupervised machine learning algorithm, to cluster player colors into two dominant groups (representing two teams).
  - Assigns players to teams based on the dominant color cluster of their jerseys.
- **Use Case**: Ensures accurate team-based statistics and analytics by distinguishing between teams.

---

### 5. `trackers`

- **Purpose**: Implements tracking algorithms for players and the ball.
- **How It Works**: 
  - Incorporates **YOLO** for initial detection of players and the ball.
  - Tracks the detected objects across frames using algorithms like Kalman filters or DeepSORT.
- **Use Case**: Captures continuous player movements and ball trajectories throughout the match for tactical analysis.

---

### 6. `training`

- **Purpose**: Provides scripts and data for training models used in various modules.
- **How It Works**: Includes datasets, preprocessing routines, and training pipelines for models like pitch key point detection and tracking.
- **Use Case**: Extendable for fine-tuning the system with custom data.

---

### 7. `utils`

- **Purpose**: Contains utility functions used across the repository.
- **Examples**:
  - Data preprocessing utilities
  - Visualization tools
  - Helper functions for file management
- **Use Case**: Simplifies common tasks and improves modularity.

---

### 8. `view_transformer`

- **Purpose**: Transforms the perspective of the match camera to a top-down, 2D view of the pitch.
- **How It Works**: 
  - Combines the pitch key points detected using YOLO with homography techniques.
  - Generates a bird's-eye view of the pitch for strategic and spatial analysis.
- **Use Case**: Enables accurate tactical visualization of player and ball movements.


## Contributing

Contributions to this project are highly encouraged! You’re welcome to:

- Fork the repository.
- Implement improvements or add new features.
- Submit a pull request for review.

Additionally, suggestions for new features, optimizations, or other enhancements are greatly appreciated. 

Thank you for supporting and using this project!
