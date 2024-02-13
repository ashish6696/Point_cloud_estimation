# Point Cloud Estimation Project

## Overview

This project is an assignment focused on estimating the most accurate 3D positions for points in a point cloud using multi-view geometry. The estimation process involves capturing images of a checkerboard and the point cloud from five different angles under specified camera settings.

## Objective
The goal is to determine the most accurate 3D positions for the points in the point cloud using multi-view geometry techniques.

## Features

- **Homography Decomposition**: Extracts the [R|t] of the camera between shots.
- **Triangulation**: Generates 3D points from camera pairs.
- **Bundle Adjustment**: Optimizes the 3D points and camera parameters to improve the reconstruction accuracy.
  

## Getting Started

### Prerequisites

- Python 3
- Required Python libraries:
  - numpy
  - opencv
  - json
  - scipy
  - plotly
  - matplotlib

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/ashish6696/Point_cloud_estimation.git
cd Point_cloud_estimation
```

## Configuration

Before running the main script, ensure the camera configuration is correctly set in `data/checkerboard_camera_config.json`. Adjust the parameters according to your camera setup.

## Usage

To start the point cloud estimation process, run the `main.py` script:

```bash
python main.py
```

## Results

The results, including the optimized 3D positions of the point cloud, will be saved in the results directory.
