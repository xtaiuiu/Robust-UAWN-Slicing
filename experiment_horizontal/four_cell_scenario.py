# generate a four-cell UAWN scenario
import logging
import pickle
from pathlib import Path

import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from algorithms.MP_Relax_algorithms.main_algorithm.Bayesian_opt.BO_horizontal import optimize_horizontal_Bayesian
from scenarios.scenario_creators import create_scenario
from network_classes.scenario import Scenario
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from matplotlib.colors import LogNorm
from matplotlib import cm
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
BASE_DIR = Path(__file__).resolve().parent

class FourCellScenario:
    def __init__(self, cell_size=100, n_slices_per_cell=2, min_users_per_slice=10, max_users_per_slice=25):
        """
        Initialize a four-cell scenario.

        Args:
            cell_size (int): Size of each cell in meters (creates cell_size x cell_size area)
            n_slices_per_cell (int): Number of slices per cell
            min_users_per_slice (int): Minimum number of users per slice
            max_users_per_slice (int): Maximum number of users per slice
        """
        self.cell_size = cell_size
        self.n_slices_per_cell = n_slices_per_cell
        self.min_users = min_users_per_slice
        self.max_users = max_users_per_slice
        self.scenarios = []

    def generate(self):
        """Generate the four-cell scenario with random user distribution."""
        # Define the center points of each cell
        cell_centers = [
            (-self.cell_size, -self.cell_size),  # Bottom-left cell
            (self.cell_size, -self.cell_size),  # Bottom-right cell
            (-self.cell_size, self.cell_size),  # Top-left cell
            (self.cell_size, self.cell_size)  # Top-right cell
        ]

        # Create a scenario for each cell
        for i, (center_x, center_y) in enumerate(cell_centers):
            # Randomly determine number of users for each slice
            n_users_per_slice = [
                random.randint(self.min_users, self.max_users)
                for _ in range(self.n_slices_per_cell)
            ]

            # Calculate network radius to cover the cell (diagonal distance from center to corner)
            network_radius = self.cell_size

            # Create scenario with the calculated parameters
            scenario = create_scenario(
                n_slices=self.n_slices_per_cell,
                network_radius=network_radius,
                n_UEs_per_slice=sum(n_users_per_slice) // self.n_slices_per_cell,
                p_max=100,
                b_tot=100,
            )

            # Computing the UAV position to cover the cell by BO
            f_opt, x_opt = optimize_horizontal_Bayesian(scenario)
            x_opt += cell_centers[i]
            scenario.uav.x, scenario.uav.y = x_opt[0], x_opt[1]

            # Update user positions to be within the cell boundaries
            for sl in scenario.slices:
                for ue in sl.UEs:
                    # Generate random position within the cell
                    ue.x = np.random.uniform(
                        center_x - self.cell_size,
                        center_x + self.cell_size
                    )
                    ue.y = np.random.uniform(
                        center_y - self.cell_size,
                        center_y + self.cell_size
                    )

            self.scenarios.append(scenario)

        return self


def plot_scenario(ax, scenarios, slice_styles):
    plt.rcParams.update({'font.size': 18})

    """Helper function to plot the scenario visualization"""

    # Keep track of labels to avoid duplicates in legend
    embb_label_added = False
    urllc_label_added = False
    cell_size = scenarios[0].pn.radius

    # Define the center points of each cell
    cell_centers = [
        (-cell_size, -cell_size),  # Bottom-left cell
        (cell_size, -cell_size),  # Bottom-right cell
        (-cell_size, cell_size),  # Top-left cell
        (cell_size, cell_size)  # Top-right cell
    ]

    uav_movement = [
        (0, 0),  # UAV 1
        (-15, 10),  # UAV 2
        (0, 0),  # UAV 3
        (-20, -20),  # UAV 4
    ]

    x, y = [], []  # Store user positions for density plot
    r = []  # Store requested rates for density plot

    # Draw each cell
    for i, scenario in enumerate(scenarios):
        # Calculate cell boundaries
        cell_center_x = cell_centers[i][0]
        cell_center_y = cell_centers[i][1]
        cell_radius = scenario.pn.radius

        # Draw cell boundary (circle)
        cell = plt.Circle(
            (cell_center_x, cell_center_y),
            cell_radius,
            fill=False,
            color='black',
            linestyle='--',
            alpha=0.3,
            zorder=10,
        )
        ax.add_patch(cell)

        # Plot UAV
        drone_path = BASE_DIR / "drone.png"
        uav_icon = plt.imread(drone_path)
        imagebox = OffsetImage(uav_icon, zoom=0.035)  # 调整缩放比例
        ab = AnnotationBbox(imagebox, (scenario.uav.x + uav_movement[i][0],
                                       scenario.uav.y + uav_movement[i][1]), frameon=False, zorder=15)
        ax.add_artist(ab)
        # ax.text(scenario.uav.x + 0.1, scenario.uav.y + 0.1, f"UAV {i}", fontsize=10, color="black", weight="bold")

        # Plot users for each slice
        for sl in scenario.slices:
            # Determine slice type based on bandwidth (from slice_creators.py)
            if sl.b_width >= 0.1:  # eMBB has higher bandwidth
                sl_type = 'embb'
                label = 'eMBB UE' if not embb_label_added else ''
                embb_label_added = True
            else:
                sl_type = 'urllc'
                label = 'URLLC UE' if not urllc_label_added else ''
                urllc_label_added = True

            # Get user positions
            x_coords = [ue.x for ue in sl.UEs]
            y_coords = [ue.y for ue in sl.UEs]
            r_requested = [ue.tilde_r * sl.r_sla for ue in sl.UEs]

            # Store user positions and rates for density plot
            x.extend(x_coords)
            y.extend(y_coords)
            r.extend(r_requested)

            # Plot users only if they are within any cell boundary
            if x_coords:  # Only process if there are UEs
                # Create lists to store coordinates of users within any cell
                x_filtered = []
                y_filtered = []

                # Check each user's position against all cell boundaries
                for x_c, y_c in zip(x_coords, y_coords):
                    # Check if user is within any cell
                    user_in_any_cell = False
                    for scenario in scenarios:
                        cell_center_x, cell_center_y = scenario.uav.x, scenario.uav.y
                        cell_radius = scenario.pn.radius
                        # Calculate distance from user to cell center
                        distance = np.sqrt((x_c - cell_center_x) ** 2 + (y_c - cell_center_y) ** 2)
                        if distance <= cell_radius:
                            user_in_any_cell = True
                            break

                    # Only add user if they're in at least one cell
                    if user_in_any_cell:
                        x_filtered.append(x_c)
                        y_filtered.append(y_c)

                # Only plot if there are users within cells
                if x_filtered:
                    ax.scatter(
                        x_filtered, y_filtered,
                        c=slice_styles[sl_type]['color'],
                        marker=slice_styles[sl_type]['marker'],
                        alpha=0.5,
                        edgecolors='w',
                        s=50,
                        label=label,
                        zorder=5
                    )
    # Density plot
    values = np.vstack([x, y])
    kde = gaussian_kde(values, weights=r, bw_method=0.3)

    X, Y = np.meshgrid(np.linspace(-2 * cell_radius, 2 * cell_radius, 200),
                       np.linspace(-2 * cell_radius, 2 * cell_radius, 200))
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    #  Heatmap plot
    contour = ax.contourf(X, Y, Z, levels=50, cmap="coolwarm", zorder=1)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("User Demand (Mbps)")

    # Set axis labels and title
    # ax.set_xlabel('X Coordinate (m)', fontsize=16)
    # ax.set_ylabel('Y Coordinate (m)', fontsize=16)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    # # Create legend
    # handles, labels = ax.get_legend_handles_labels()
    # unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    # if unique:  # Only add legend if there are items to show
    #     ax.legend(*zip(*unique), loc='upper right')

    ax.grid(True, linestyle='--', alpha=0.6)


def create_four_cell_scenario():
    """
    Create a four-cell scenario with default parameters.

    Returns:
        FourCellScenario: The generated four-cell scenario
    """
    scenarios = FourCellScenario(
        cell_size=100,  # 100x100m cells
        n_slices_per_cell=2,  # 2 slices per cell
        min_users_per_slice=20,  # 20-40 users per slice
        max_users_per_slice=40
    ).generate()
    file = BASE_DIR / 'four_cell_scenario.pkl'
    with open(file, 'wb') as f:
        pickle.dump(scenarios, f)

    return scenarios


if __name__ == "__main__":
    logging.disable(logging.INFO)
    # First, create a figure and axis
    fig, ax = plt.subplots()

    # Define your slice styles (see explanation below)
    slice_styles = {
        'embb': {'color': 'blue', 'marker': 'o', 'label': 'eMBB UE'},
        'urllc': {'color': 'red', 'marker': '^', 'label': 'URLLC UE'}
    }

    # Create the four-cell scenario
    # four_cell = create_four_cell_scenario()
    # Load the four-cell scenario
    file = BASE_DIR / 'four_cell_scenario.pkl'
    with open(file, 'rb') as f:
        four_cell = pickle.load(f)

    # Plot the scenario
    plot_scenario(ax, four_cell.scenarios, slice_styles)

    # Show the plot
    plt.tight_layout()
    plt.show()
