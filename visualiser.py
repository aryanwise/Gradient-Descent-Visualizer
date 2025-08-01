import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd


# --- Data Scaling Utilities ---
# We scale data to the 0-1 range to help gradient descent converge better.
def scale_data(data):
    """Scales data to a 0-1 range."""
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val), min_val, max_val


def unscale_parameter(val, min_val, max_val):
    """Helper to unscale a single value."""
    return val * (max_val - min_val) + min_val


# --- Main Application Class ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Gradient Descent Visualizer (Salary Data)")

        # --- Load and Prepare Data ---
        try:
            df = pd.read_csv("Salary_dataset.csv")
            # Drop any missing values to be safe
            df = df.dropna(subset=["YearsExperience", "Salary"])

            self.X_orig = df["YearsExperience"].values.reshape(-1, 1)
            self.y_orig = df["Salary"].values.reshape(-1, 1)

            # Scale the data for better training performance
            self.X, self.X_min, self.X_max = scale_data(self.X_orig)
            self.y, self.y_min, self.y_max = scale_data(self.y_orig)

        except FileNotFoundError:
            messagebox.showerror(
                "Error",
                "Salary_dataset.csv not found! Make sure it's in the same directory.",
            )
            root.destroy()
            return

        # --- Control Frame ---
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Hyperparameters
        self.learning_rate = 0.5
        self.n_epochs = 100

        self.start_button = tk.Button(
            control_frame, text="Start Visualization", command=self.start_visualization
        )
        self.start_button.pack(side=tk.LEFT)

        # --- Matplotlib Plot Setup ---
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.reset_plot()

    def reset_plot(self):
        """Clears the plot and shows the initial data."""
        self.ax.clear()
        self.ax.scatter(self.X_orig, self.y_orig)  # Plot the original data points
        self.ax.set_title("Salary vs. Experience")
        self.ax.set_xlabel("Years of Experience")
        self.ax.set_ylabel("Salary")
        self.canvas.draw()

    def gradient_descent(self):
        """Generator function that yields the state at each epoch on scaled data."""
        m = np.random.randn()
        b = np.random.randn()
        N = len(self.X)

        for i in range(self.n_epochs):
            y_predicted = m * self.X + b

            dm = (-2 / N) * sum(self.X * (self.y - y_predicted))
            db = (-2 / N) * sum(self.y - y_predicted)

            m = m - self.learning_rate * dm
            b = b - self.learning_rate * db

            yield m, b

    def start_visualization(self):
        """Runs the visualization step-by-step."""
        self.reset_plot()

        (line,) = self.ax.plot([], [], "r-", lw=2)

        gd_generator = self.gradient_descent()

        def update_frame(frame_num):
            try:
                m_scaled, b_scaled = next(gd_generator)

                # Plotting line needs to be on the original data's scale
                x_line_orig = np.array([self.X_orig.min(), self.X_orig.max()])
                # Convert original x to scaled x to use with scaled m and b
                x_line_scaled = (x_line_orig - self.X_min) / (self.X_max - self.X_min)
                y_line_scaled = m_scaled * x_line_scaled + b_scaled
                # Convert predicted y back to original scale
                y_line_orig = unscale_parameter(y_line_scaled, self.y_min, self.y_max)

                line.set_data(x_line_orig, y_line_orig)
                self.ax.set_title(f"Epoch: {frame_num+1}/{self.n_epochs}")
                self.canvas.draw()

                if frame_num < self.n_epochs - 1:
                    self.root.after(50, update_frame, frame_num + 1)
                else:
                    # Final title with unscaled parameters for interpretability
                    final_slope = (y_line_orig[1] - y_line_orig[0]) / (
                        x_line_orig[1] - x_line_orig[0]
                    )
                    final_intercept = y_line_orig[0] - final_slope * x_line_orig[0]
                    self.ax.set_title(
                        f"Finished! Salary = {final_slope:.2f} * Experience + {final_intercept:.2f}"
                    )
                    self.canvas.draw()

            except StopIteration:
                return

        update_frame(0)


# --- Main execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
