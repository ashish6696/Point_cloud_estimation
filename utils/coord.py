import numpy as np
import matplotlib.pyplot as plt

def checkerboard(width, height, cols, rows):
    """
    Generates a 2D grid of checkerboard pattern points.

    Parameters:
    - width (float): Width of the checkerboard.
    - height (float): Height of the checkerboard.
    - cols (int): Number of columns in the grid.
    - rows (int): Number of rows in the grid.

    Returns:
    - numpy.ndarray: Array of shape (rows*cols, 4), containing index, x, y, z coordinates
      for each point. z is always 0, indicating points lie on the z=0 plane.
    """
    # Calculate spacing between points
    x_spacing = width / (cols - 1)
    y_spacing = height / (rows - 1)
    # Initialize an array to hold the point coordinates
    checkerboard_points = np.zeros((rows * cols, 4))  # 4 columns: index, x, y, z

    # Fill the array with the checkerboard points
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col  # Calculate the index for the current point
            x = col * x_spacing
            y = row * y_spacing
            z = 0  # All points have z=0 since they're in the plane z=0
            checkerboard_points[index] = [index, x, y, z]

    return checkerboard_points

def plot_checkerboard(checkerboard_points, width, height):
    """
    Plots checkerboard corners and frame axes for visualization.

    Parameters:
    - checkerboard_points (numpy.ndarray): Array of shape (N, 4) with checkerboard corners' [index, x, y, z].
    - width (float): Checkerboard width along X-axis.
    - height (float): Checkerboard height along Y-axis.

    Visualizes checkerboard corners and X, Y axes on a plot. Points are shown in black,
    X-axis in red, and Y-axis in blue. Includes axis labels and a legend.
    """


    # Extract x and y coordinates
    x_coords = checkerboard_points[:, 1]
    y_coords = checkerboard_points[:, 2]

    # Plot checkerboard points
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, color='black', label='Checkerboard Corners')  # Plot points

    # Plot frame axes
    # X-axis
    plt.plot([0, width], [0, 0], color='red', linewidth=2, label='X-axis')
    # Y-axis
    plt.plot([0, 0], [0, height], color='blue', linewidth=2, label='Y-axis')

    # Set plot characteristics
    plt.title('Checkerboard Layout with Frame Axes')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    # Checkerboard dimensions and point count
    width, height = 1, 1  # in meters
    cols, rows = 5, 10  # number of columns and rows

    checkerboard_points = checkerboard(width, height, cols, rows)
    print(checkerboard_points)
    plot_checkerboard(checkerboard_points,width,height)




if __name__ == "__main__":
    main()
