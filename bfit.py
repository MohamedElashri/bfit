import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.integrate import quad


# -------------------------- Fit Functions -------------------------- #
class FitFunctions:
    def __init__(self, bin_width, x_min, x_max):
        self.bin_width = bin_width
        self.x_min = x_min
        self.x_max = x_max

    def gaussian(self, x, mu, sigma):
        return self.bin_width * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)

    def expA(self, x, A, b):
        integral = (A / b) * (1.0 - np.exp(-b * (self.x_max - self.x_min)))
        norm = 1. / integral
        return self.bin_width * norm * A * np.exp(-b * (x - self.x_min))

    def DoubleGaussian_plus_ExpA(self, x_vals, n_s, f, n_b, mu1, mu2, sigma1, sigma2, A, b):
        n_s1 = n_s * f
        n_s2 = n_s * (1 - f)
        return n_s1 * self.gaussian(x_vals, mu1, sigma1) + n_s2 * self.gaussian(x_vals, mu2, sigma2) + n_b * self.expA(x_vals, A, b)

    def DoubleGaussian_plus_Parabola(self, x_vals, n_s, f, n_b, mu1, mu2, sigma1, sigma2, a, b, c):
        n_s1 = n_s * f
        n_s2 = n_s * (1 - f)
        return n_s1 * self.gaussian(x_vals, mu1, sigma1) + n_s2 * self.gaussian(x_vals, mu2, sigma2) + n_b * self.parabola(x_vals, a, b, c)

    def parabola(self, x, a, b, c):
        return self.bin_width * a * (x ** 2) + b * x + c

    def DoubleGaussian_plus_Linear(self, x_vals, n_s, f, n_b, mu1, mu2, sigma1, sigma2, m, b):
        n_s1 = n_s * f
        n_s2 = n_s * (1 - f)
        return n_s1 * self.gaussian(x_vals, mu1, sigma1) + n_s2 * self.gaussian(x_vals, mu2, sigma2) + n_b * self.Linear(x_vals, m, b)

    def Linear(self, x, m, b):
        integral = m * (self.x_max - self.x_min) + 0.5 * b * (self.x_max ** 2 - self.x_min ** 2)
        norm = 1. / integral
        return self.bin_width * norm * (m * x + b)

    def argus_bg_integral(self, m0, c, p):
        def integrand(x):
            z = 1 - (x / m0) ** 2
            return x * np.sqrt(z) * np.exp(c * z ** p) if z > 0 else 0

        integral, _ = quad(integrand, 0, m0, limit=10000)
        return integral

    def argus_bg(self, x, m0, c, p):
        normalization = self.argus_bg_integral(m0, c, p)
        z = 1 - (x / m0) ** 2
        return np.where(z > 0, (x * np.sqrt(z) * np.exp(c * z ** p)) / normalization, 0)

    def Gaussian_plus_Argus(self, x_vals, n_s, n_b, mu, sigma, m0, c, p):
        return n_s * self.gaussian(x_vals, mu, sigma) + n_b * self.argus_bg(x_vals, m0, c, p)

    def Gaussian_plus_ExpA(self, x_vals, n_s, n_b, mu, sigma, A, b):

        return n_s * self.gaussian(x_vals, mu, sigma) + n_b * self.expA(x_vals, A, b)

    def Gaussian_plus_Linear(self, x_vals, n_s, n_b, mu, sigma, m, b):
        return n_s * self.gaussian(x_vals, mu, sigma) + n_b * self.Linear(x_vals, m, b)

    def breit_wigner(self, x, M, Gamma):
        return (1 / np.pi) * (0.5 * Gamma) / ((x - M) ** 2 + (0.5 * Gamma) ** 2)

    def BreitWigner_plus_ExpA(self, x_vals, n_s, n_b, M, Gamma, A, b):
        return n_s * self.breit_wigner(x_vals, M, Gamma) + n_b * self.expA(x_vals, A, b)

    def BreitWigner_plus_Linear(self, x_vals, n_s, n_b, M, Gamma, m, b):
        return n_s * self.breit_wigner(x_vals, M, Gamma) + n_b * self.Linear(x_vals, m, b)

    def crystal_ball(self, x, alpha, n, mu, sigma):
        A = (n / np.abs(alpha)) ** n * np.exp(-alpha ** 2 / 2)
        B = n / np.abs(alpha) - np.abs(alpha)
        return np.where((x - mu) / sigma > -alpha, np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)),
                        A * (B - (x - mu) / sigma) ** (-n))

    def CrystalBall_plus_ExpA(self, x_vals, n_s, n_b, alpha, n, mu, sigma, A, b):
        return n_s * self.crystal_ball(x_vals, alpha, n, mu, sigma) + n_b * self.expA(x_vals, A, b)


# -------------------------- Base Fit Class -------------------------- #
class FitBase:
    def __init__(self, bins, nC, minuit_limits=None):
        self.bins = bins  # Add this line to store the bins as an attribute
        self.x_min, self.x_max, self.bin_width, self.x_vals, self.y_vals, self.y_errs = self.fit_setup(nC, bins)
        self.minuit_limits = minuit_limits or {}
        self.fit_functions = FitFunctions(self.bin_width, self.x_min, self.x_max)
        self.fit_result = None  # Will store Minuit fit result
    def fit_setup(self, nC, bins):
        x_min = bins[0]
        x_max = bins[-1]
        bin_width = bins[1] - bins[0]

        # Calculate x values at bin centers
        x_vals = 0.5 * (bins[:-1] + bins[1:])

        # Assign counts and errors
        y_vals = nC
        y_errs = np.sqrt(nC)  # Assuming Poisson statistics for errors

        return x_min, x_max, bin_width, x_vals, y_vals, y_errs
    def chi_squared(self, *params):
        mask = self.y_errs > 0
        # Apply the mask to filter x values and y values
        x_vals_masked = self.x_vals[mask]
        y_vals_masked = self.y_vals[mask]
        y_errs_masked = self.y_errs[mask]
        # Use the appropriate fit function and calculate the prediction for the masked x values
        prediction = self.fit_function(x_vals_masked, *params)

        # Calculate the residuals squared, using the masked y values and errors
        residuals_squared = ((y_vals_masked - prediction) / y_errs_masked) ** 2

        # Sum of the squared residuals gives the chi-squared value
        chi_sq_value = np.sum(residuals_squared)
        return chi_sq_value

    def fit(self, init_pars, param_names=None):
        # Check if parameter names are provided, otherwise default to generic names
        if param_names is None:
            param_names = ['x{}'.format(i) for i in range(len(init_pars))]

        # Ensure the number of parameter names matches the number of initial parameters
        assert len(param_names) == len(init_pars), "Number of parameter names must match the number of initial parameters."

        # Initialize Minuit with parameter names
        m = Minuit(self.chi_squared, name=param_names, *init_pars)

        for key, value in self.minuit_limits.items():
            m.limits[key] = value

        m.migrad()  # Perform the minimization
        self.fit_result = m
        return m

    def plot(self, title='Fit plot', xlabel='X', ylabel='Y', vlines=None, range=None,
            data_color='b', data_label='Data', fit_color='r', fit_label='Fit',
            show_plot=True):

        fig, ax = plt.subplots()
        ax.set_title(title)

        # Plotting histogram with all data within the specified visual range
        visual_bins = self.bins
        if range is not None:
            visual_bins = visual_bins[(visual_bins >= range[0]) & (visual_bins <= range[1] + self.bin_width)]
            ax.set_xlim(range)

        ax.hist(self.x_vals, bins=visual_bins, weights=self.y_vals, color=data_color, label=data_label)

        # Plotting the fit function if results exist, over the original full range
        if self.fit_result:
            # Generating a dense x-values array for smooth plotting of the fit curve over the original range
            dense_x_vals = np.linspace(self.x_min, self.x_max, 1000)
            predictions = self.fit_function(dense_x_vals, *self.fit_result.values)
            ax.plot(dense_x_vals, predictions, color=fit_color, label=fit_label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Adjusting y-axis limit to better fit the visualized data and prediction
        ax.set_ylim(0, 1.15 * max(np.append(self.y_vals, predictions)))

        ax.grid(True)
        if vlines:
            for vl in vlines:
                ax.vlines(vl, 0., 0.6 * max(self.y_vals), colors='yellow')
        ax.legend()

        if show_plot:
            plt.show()

        return fig, ax

    def summary(self):
        if self.fit_result is not None:
            # Print the Minuit summary table
            print(self.fit_result)

            '''
            ## This part is for debugging only

            # Optionally, for more detailed analysis:
            # self.fit_result.minos()  # For detailed error analysis, if neede
            # Accessing specific fit parameters and errors
            print("Fit Parameters:")
            for name in self.fit_result.parameters:
                print(f"{name}: {self.fit_result.values[name]} ± {self.fit_result.errors[name]}")
            '''
        else:
            print("Fit has not been performed yet.")


    @property
    def fit_params(self):
        if self.fit_result:
            return self.fit_result.values
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def fit_errors(self):
        if self.fit_result:
            return self.fit_result.errors
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def chi_squared_value(self):
        if self.fit_result:
            return self.fit_result.fval
        else:
            raise RuntimeError("Fit has not been performed yet.")


# -------------------------- Derived Fit Classes -------------------------- #
class Gaussian_plus_Exp(FitBase):
    def __init__(self, bins, nC, minuit_limits=None):
        super().__init__(bins, nC, minuit_limits)

    def fit_function(self, x_vals, *params):
        return self.fit_functions.Gaussian_plus_ExpA(x_vals, *params)


class DoubleGaussian_plus_Exp(FitBase):
    def __init__(self, bins, nC, minuit_limits=None):
        super().__init__(bins, nC, minuit_limits)

    def fit_function(self, x_vals, *params):
        return self.fit_functions.DoubleGaussian_plus_ExpA(x_vals, *params)


class DoubleGaussian_plus_Parabola(FitBase):
    def __init__(self, bins, nC, minuit_limits=None):
        super().__init__(bins, nC, minuit_limits)

    def fit_function(self, x_vals, *params):
        return self.fit_functions.DoubleGaussian_plus_Parabola(x_vals, *params)


class DoubleGaussian_plus_Linear(FitBase):
    def __init__(self, bins, nC, minuit_limits=None):
        super().__init__(bins, nC, minuit_limits)

    def fit_function(self, x_vals, *params):
        return self.fit_functions.DoubleGaussian_plus_Linear(x_vals, *params)


class Gaussian_plus_Argus(FitBase):
    def __init__(self, bins, nC, minuit_limits=None):
        super().__init__(bins, nC, minuit_limits)

    def fit_function(self, x_vals, *params):
        return self.fit_functions.Gaussian_plus_Argus(x_vals, *params)

class Gaussian_plus_Linear(FitBase):
    def __init__(self, bins, nC, minuit_limits=None):
        super().__init__(bins, nC, minuit_limits)

    def fit_function(self, x_vals, *params):
        return self.fit_functions.Gaussian_plus_Linear(x_vals, *params)

class BreitWigner_plus_Exp(FitBase):
    def __init__(self, bins, nC, minuit_limits=None):
        super().__init__(bins, nC, minuit_limits)

    def fit_function(self, x_vals, *params):
        return self.fit_functions.BreitWigner_plus_ExpA(x_vals, *params)


class BreitWigner_plus_Linear(FitBase):
    def __init__(self, bins, nC, minuit_limits=None):
        super().__init__(bins, nC, minuit_limits)

    def fit_function(self, x_vals, *params):
        return self.fit_functions.BreitWigner_plus_Linear(x_vals, *params)


class CrystalBall_plus_Exp(FitBase):
    def __init__(self, bins, nC, minuit_limits=None):
        super().__init__(bins, nC, minuit_limits)

    def fit_function(self, x_vals, *params):
        return self.fit_functions.CrystalBall_plus_ExpA(x_vals, *params)
