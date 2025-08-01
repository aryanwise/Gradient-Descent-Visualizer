# Goal: Analysing the cost function and gradient descent for Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D


class LinearRegressionAnalysis:
    """
    A class to encapsulate the process of training, evaluating, and visualizing
    a simple linear regression model using both Scikit-learn and a custom
    gradient descent implementation.
    """

    def __init__(self, dataset_path):
        """
        Initializes the class, loads and preprocesses the data.
        """
        self.dataset_path = dataset_path
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """
        Loads the dataset, splits it, and applies feature scaling.
        """
        # --- 1. Load Data ---
        # Use index_col=0 to treat the first column of the CSV as the index, not data.
        dataset = pd.read_csv(self.dataset_path, index_col=0)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values.reshape(
            -1, 1
        )  # Use -1 to always get the last column (Salary)

        # --- 2. Split Data ---
        self.X_train_orig, self.X_test_orig, self.y_train_orig, self.y_test_orig = (
            train_test_split(X, y, test_size=1 / 3, random_state=0)
        )

        # --- 3. Feature Scaling ---
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()

        self.X_train = self.sc_X.fit_transform(self.X_train_orig)
        self.X_test = self.sc_X.transform(self.X_test_orig)
        # Flatten y_train and y_test to be 1D arrays (vectors) for calculations
        self.y_train = self.sc_y.fit_transform(self.y_train_orig).flatten()
        self.y_test = self.sc_y.transform(self.y_test_orig).flatten()

        # Add a bias (intercept) term to the training data for matrix calculations
        self.X_train_b = np.c_[np.ones((self.X_train.shape[0], 1)), self.X_train]

    def train_sklearn_model(self):
        """
        Fits a Simple Linear Regression model using Scikit-learn.
        """
        self.regressor = LinearRegression()
        # regressor.fit can handle a 1D y_train
        self.regressor.fit(self.X_train, self.y_train)

        # Store the optimal parameters found by sklearn
        # intercept_ is a scalar, coef_ is an array e.g., [slope]
        self.theta_optimal_sklearn = np.array(
            [self.regressor.intercept_, self.regressor.coef_[0]]
        )

        print("--- Scikit-learn Model ---")
        print(f"Optimal Theta (intercept, slope): {self.theta_optimal_sklearn}")

        # Evaluate the model
        y_pred = self.regressor.predict(self.X_test)
        # Use self.y_test which is already scaled and flattened
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"Mean Squared Error (on test set): {mse:.4f}")
        print(f"R2 Score (on test set): {r2:.4f}\n")

    @staticmethod
    def _cost_function(X_b, y, theta):
        """
        Calculates the cost (Mean Squared Error).
        X_b: Input features with bias term.
        """
        m = len(y)
        predictions = X_b.dot(theta)
        cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
        return cost

    def run_gradient_descent(self, alpha=0.01, iterations=1000):
        """
        Performs gradient descent to find the optimal theta values.
        Returns the final theta, cost history, and the path of theta values.
        """
        print("--- Custom Gradient Descent ---")
        m = len(self.y_train)
        theta = np.zeros(self.X_train_b.shape[1])  # Initialize theta to zeros

        self.cost_history = []
        self.theta_path = []

        for i in range(iterations):
            # All arrays are now correctly shaped for matrix multiplication
            predictions = self.X_train_b.dot(theta)
            errors = predictions - self.y_train

            # Update theta using the gradient descent formula
            theta -= (alpha / m) * (self.X_train_b.T.dot(errors))

            # Store history for plotting
            self.theta_path.append(theta.copy())
            cost = self._cost_function(self.X_train_b, self.y_train, theta)
            self.cost_history.append(cost)

        self.theta_gd = theta
        print(f"Theta found by Gradient Descent: {self.theta_gd}")
        return self.theta_gd, self.cost_history, self.theta_path

    def visualize_all(self):
        """
        Generates all visualizations for the analysis.
        """
        self._visualize_regression_fit()
        self._visualize_cost_function_surface()
        self._visualize_gd_path()

    def _visualize_regression_fit(self):
        """
        Visualizes the training and test set results with the regression line.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plotting logic for both train and test sets
        def plot_subset(ax, X_orig, y_orig, title):
            ax.scatter(X_orig, y_orig, color="red", label="Actual Data")

            # Create a continuous line for the prediction
            x_range = np.linspace(X_orig.min(), X_orig.max(), 100).reshape(-1, 1)
            x_range_scaled = self.sc_X.transform(x_range)
            y_pred_scaled = self.regressor.predict(x_range_scaled)

            # Reshape y_pred_scaled to 2D for inverse_transform
            y_pred_orig = self.sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

            ax.plot(x_range, y_pred_orig, color="blue", label="Regression Line")
            ax.set_title(title)
            ax.set_xlabel("Years of Experience")
            ax.set_ylabel("Salary")
            ax.legend()

        # Training set plot
        plot_subset(
            ax1,
            self.X_train_orig,
            self.y_train_orig,
            "Salary vs Experience (Training set)",
        )

        # Test set plot
        plot_subset(
            ax2, self.X_test_orig, self.y_test_orig, "Salary vs Experience (Test set)"
        )

        plt.tight_layout()
        plt.show()

    def _visualize_cost_function_surface(self):
        """
        Plots the cost function as a 3D surface and a 2D contour plot.
        """
        theta0_vals = np.linspace(
            self.theta_optimal_sklearn[0] - 1, self.theta_optimal_sklearn[0] + 1, 100
        )
        theta1_vals = np.linspace(
            self.theta_optimal_sklearn[1] - 1, self.theta_optimal_sklearn[1] + 1, 100
        )

        J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

        for i, theta0 in enumerate(theta0_vals):
            for j, theta1 in enumerate(theta1_vals):
                t = np.array([theta0, theta1])
                J_vals[i, j] = self._cost_function(self.X_train_b, self.y_train, t)

        theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)

        fig = plt.figure(figsize=(18, 7))

        # 3D Surface Plot
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot_surface(theta0_mesh, theta1_mesh, J_vals.T, cmap="viridis", alpha=0.8)
        ax1.set_xlabel("Theta 0 (Intercept)")
        ax1.set_ylabel("Theta 1 (Slope)")
        ax1.set_zlabel("Cost J")
        ax1.set_title("Cost Function Surface Plot")

        # 2D Contour Plot
        ax2 = fig.add_subplot(122)
        ax2.contour(
            theta0_mesh,
            theta1_mesh,
            J_vals.T,
            levels=np.logspace(-2, 3, 20),
            cmap="viridis",
        )
        ax2.set_xlabel("Theta 0 (Intercept)")
        ax2.set_ylabel("Theta 1 (Slope)")
        ax2.set_title("Cost Function Contour Plot")

        # Mark the optimal point
        ax2.plot(
            self.theta_optimal_sklearn[0],
            self.theta_optimal_sklearn[1],
            "rx",
            markersize=10,
            label="Optimal Theta (Sklearn)",
        )
        ax2.legend()

        plt.show()

    def _visualize_gd_path(self):
        """
        Plots the cost history and the path of gradient descent on the contour plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Cost History Plot
        ax1.plot(self.cost_history)
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Cost J")
        ax1.set_title("Cost History during Gradient Descent")

        # Contour Plot with GD Path
        theta0_vals = np.linspace(
            self.theta_optimal_sklearn[0] - 1, self.theta_optimal_sklearn[0] + 1, 100
        )
        theta1_vals = np.linspace(
            self.theta_optimal_sklearn[1] - 1, self.theta_optimal_sklearn[1] + 1, 100
        )
        theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)
        J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
        for i, theta0 in enumerate(theta0_vals):
            for j, theta1 in enumerate(theta1_vals):
                t = np.array([theta0, theta1])
                J_vals[i, j] = self._cost_function(self.X_train_b, self.y_train, t)

        ax2.contour(
            theta0_mesh,
            theta1_mesh,
            J_vals.T,
            levels=np.logspace(-2, 3, 20),
            cmap="viridis",
        )

        # Plot the path
        path = np.array(self.theta_path)
        ax2.plot(path[:, 0], path[:, 1], "r-o", markersize=3, label="GD Path")
        ax2.plot(
            self.theta_optimal_sklearn[0],
            self.theta_optimal_sklearn[1],
            "gx",
            markersize=15,
            mew=2,
            label="Optimal Theta",
        )
        ax2.set_xlabel("Theta 0 (Intercept)")
        ax2.set_ylabel("Theta 1 (Slope)")
        ax2.set_title("Gradient Descent Path")
        ax2.legend()

        plt.tight_layout()
        plt.show()


# --- Main execution ---
if __name__ == "__main__":
    # Create an instance of the analysis class
    analysis = LinearRegressionAnalysis(dataset_path="Salary_dataset.csv")

    # Train the sklearn model and print results
    analysis.train_sklearn_model()

    # Run the custom gradient descent
    analysis.run_gradient_descent(alpha=0.1, iterations=1000)

    # Generate all visualizations
    analysis.visualize_all()
