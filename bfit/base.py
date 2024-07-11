import numpy as np
from scipy.stats import rv_continuous
from scipy.integrate import quad
from iminuit import Minuit
import matplotlib.pyplot as plt
from .utils import PlotConfig

class Component(rv_continuous):
    """
    Base class for fit components.

    This class extends scipy's rv_continuous to create custom probability
    distributions that can be used as components in composite fit models.

    Attributes:
        function (callable): The function defining the component.
        _pdf (callable): The probability density function of the component.
        n_params (int): The number of parameters in the component.
        normalization (float): The normalization factor for the component.
    """

    def __init__(self, function, pdf, n_params):
        """
        Initialize the Component instance.

        Args:
            function (callable): The function defining the component.
            pdf (callable): The probability density function of the component.
            n_params (int): The number of parameters in the component.
        """
        super().__init__()
        self.function = function
        self._pdf = pdf
        self.n_params = n_params
        self.normalization = 1.0

    def __call__(self, x, *params):
        """
        Call the component's function.

        Args:
            x (array-like): The x-values.
            *params: The parameters for the function.

        Returns:
            array-like: The y-values of the function.
        """
        return self.function(x, *params)

    def pdf(self, x, *params):
        """
        Probability density function of the component.

        Args:
            x (array-like): The x-values.
            *params: The parameters for the PDF.

        Returns:
            array-like: The y-values of the PDF.
        """
        return self._pdf(x, *params) / self.normalization

    def normalize(self, x_min, x_max, *params):
        """
        Normalize the component over a given range.

        Args:
            x_min (float): The minimum x-value of the range.
            x_max (float): The maximum x-value of the range.
            *params: The parameters for the PDF.

        Returns:
            float: The normalization factor.
        """
        self.normalization, _ = quad(self._pdf, x_min, x_max, args=params)
        return self.normalization

class FitBase:
    """
    Base class for all binned fit models.

    This class provides the fundamental structure and methods for performing
    binned fits on histogram data.

    Attributes:
        bins (array-like): The bin edges of the histogram.
        x_min (float): The minimum x-value of the fit range.
        x_max (float): The maximum x-value of the fit range.
        bin_width (float): The width of each bin.
        x_vals (array-like): The center of each bin.
        y_vals (array-like): The count in each bin.
        y_errs (array-like): The error on the count in each bin.
        param_limits (dict): Optional limits on the fit parameters.
        fit_result (Minuit): The result of the fit, once performed.
    """

    def __init__(self, bins, counts, param_limits=None):
        """
        Initialize the FitBase instance.

        Args:
            bins (array-like): The bin edges of the histogram.
            counts (array-like): The counts in each bin.
            param_limits (dict, optional): Limits on the fit parameters.
        """
        self.bins = bins
        self.x_min, self.x_max, self.bin_width, self.x_vals, self.y_vals, self.y_errs = self._setup_fit(counts, bins)
        self.param_limits = param_limits or {}
        self.fit_result = None

    def _setup_fit(self, counts, bins):
        """
        Set up the fit by calculating necessary values from the input data.

        Args:
            counts (array-like): The counts in each bin.
            bins (array-like): The bin edges of the histogram.

        Returns:
            tuple: x_min, x_max, bin_width, x_vals, y_vals, y_errs
        """
        x_min, x_max = bins[0], bins[-1]
        bin_width = bins[1] - bins[0]
        x_vals = 0.5 * (bins[:-1] + bins[1:])
        y_vals, y_errs = counts, np.sqrt(counts)
        return x_min, x_max, bin_width, x_vals, y_vals, y_errs

    def chi_squared(self, *params):
        """
        Calculate chi-squared value for the current fit parameters.

        Args:
            *params: The current fit parameters.

        Returns:
            float: The chi-squared value.
        """
        mask = self.y_errs > 0
        x_vals_masked = self.x_vals[mask]
        y_vals_masked = self.y_vals[mask]
        y_errs_masked = self.y_errs[mask]
        prediction = self.fit_function(x_vals_masked, *params)
        residuals_squared = ((y_vals_masked - prediction) / y_errs_masked) ** 2
        return np.sum(residuals_squared)

    def fit(self, initial_params, param_names=None):
        """
        Perform the fit using Minuit.

        Args:
            initial_params (list): Initial values for the fit parameters.
            param_names (list, optional): Names of the fit parameters.

        Returns:
            Minuit: The Minuit object containing the fit results.
        """
        if param_names is None:
            param_names = [f'param_{i}' for i in range(len(initial_params))]
        assert len(param_names) == len(initial_params), "Number of parameter names must match the number of initial parameters."
        
        minuit = Minuit(self.chi_squared, name=param_names, *initial_params)
        for key, value in self.param_limits.items():
            minuit.limits[key] = value
        minuit.migrad()
        self.fit_result = minuit
        return minuit

    def plot(self, config=None):
        """
        Plot the fit results.

        Args:
            config (PlotConfig, optional): Configuration for the plot.

        Returns:
            tuple: Figure and Axes objects of the plot.
        """
        if config is None:
            config = PlotConfig()

        fig, ax = plt.subplots()
        ax.set_title(config.title)

        visual_bins = self.bins
        if config.plot_range is not None:
            visual_bins = visual_bins[(visual_bins >= config.plot_range[0]) & (visual_bins <= config.plot_range[1] + self.bin_width)]
            ax.set_xlim(config.plot_range)

        ax.hist(self.x_vals, bins=visual_bins, weights=self.y_vals, color=config.data_color, label=config.data_label)

        if self.fit_result:
            dense_x_vals = np.linspace(self.x_min, self.x_max, 1000)
            predictions = self.fit_function(dense_x_vals, *self.fit_result.values)
            ax.plot(dense_x_vals, predictions, color=config.fit_color, label=config.fit_label)

        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.set_ylim(0, 1.15 * max(np.append(self.y_vals, predictions)))
        ax.grid(True)
        
        if config.vlines:
            for vl in config.vlines:
                ax.vlines(vl, 0., 0.6 * max(self.y_vals), colors='yellow')
        
        ax.legend()

        if config.show_plot:
            plt.show()

        return fig, ax

    def summary(self):
        """
        Print a summary of the fit results.
        """
        if self.fit_result is not None:
            print(self.fit_result)
        else:
            print("Fit has not been performed yet.")

    @property
    def fit_params(self):
        """
        Get the fit parameters.

        Returns:
            dict: The fit parameters.

        Raises:
            RuntimeError: If the fit has not been performed.
        """
        if self.fit_result:
            return self.fit_result.values
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def fit_errors(self):
        """
        Get the errors on the fit parameters.

        Returns:
            dict: The errors on the fit parameters.

        Raises:
            RuntimeError: If the fit has not been performed.
        """
        if self.fit_result:
            return self.fit_result.errors
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def chi_squared_value(self):
        """
        Get the chi-squared value of the fit.

        Returns:
            float: The chi-squared value.

        Raises:
            RuntimeError: If the fit has not been performed.
        """
        if self.fit_result:
            return self.fit_result.fval
        else:
            raise RuntimeError("Fit has not been performed yet.")

class UnbinnedFitBase:
    """
    Base class for all unbinned fit models.

    This class provides the fundamental structure and methods for performing
    unbinned fits on raw data points.

    Attributes:
        data (array-like): The raw data points.
        x_min (float): The minimum x-value of the fit range.
        x_max (float): The maximum x-value of the fit range.
        param_limits (dict): Optional limits on the fit parameters.
        fit_result (Minuit): The result of the fit, once performed.
    """

    def __init__(self, data, param_limits=None):
        """
        Initialize the UnbinnedFitBase instance.

        Args:
            data (array-like): The raw data points.
            param_limits (dict, optional): Limits on the fit parameters.
        """
        self.data = data
        self.x_min, self.x_max = self._setup_fit(data)
        self.param_limits = param_limits or {}
        self.fit_result = None

    def _setup_fit(self, data):
        """
        Set up the fit by calculating the range of the data.

        Args:
            data (array-like): The raw data points.

        Returns:
            tuple: x_min, x_max
        """
        return np.min(data), np.max(data)

    def log_likelihood(self, *params):
        """
        Calculate log-likelihood value.

        This method should be implemented by subclasses.

        Args:
            *params: The current fit parameters.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclass must implement abstract method")
        
    def fit(self, initial_params, param_names=None):
        """
        Perform the fit using Minuit.

        Args:
            initial_params (list): Initial values for the fit parameters.
            param_names (list, optional): Names of the fit parameters.

        Returns:
            Minuit: The Minuit object containing the fit results.
        """
        if param_names is None:
            param_names = [f'param_{i}' for i in range(len(initial_params))]
        assert len(param_names) == len(initial_params), "Number of parameter names must match the number of initial parameters."
        
        minuit = Minuit(self.log_likelihood, name=param_names, *initial_params)
        for key, value in self.param_limits.items():
            minuit.limits[key] = value
        minuit.migrad()
        self.fit_result = minuit
        return minuit

    def plot(self, config=None):
        """
        Plot the fit results.

        Args:
            config (PlotConfig, optional): Configuration for the plot.

        Returns:
            tuple: Figure and Axes objects of the plot.
        """
        if config is None:
            config = PlotConfig()

        fig, ax = plt.subplots()
        ax.set_title(config.title)

        if config.plot_range is not None:
            ax.set_xlim(config.plot_range)
        else:
            ax.set_xlim(self.x_min, self.x_max)

        ax.hist(self.data, bins=100, density=True, color=config.data_color, label=config.data_label)

        if self.fit_result:
            dense_x_vals = np.linspace(self.x_min, self.x_max, 1000)
            predictions = self.pdf(dense_x_vals, *self.fit_result.values)
            ax.plot(dense_x_vals, predictions, color=config.fit_color, label=config.fit_label)

        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.grid(True)
        
        if config.vlines:
            for vl in config.vlines:
                ax.vlines(vl, 0., 0.6 * ax.get_ylim()[1], colors='yellow')
        
        ax.legend()

        if config.show_plot:
            plt.show()

        return fig, ax

    def summary(self):
        """
        Print a summary of the fit results.
        """
        if self.fit_result is not None:
            print(self.fit_result)
        else:
            print("Fit has not been performed yet.")

    @property
    def fit_params(self):
        """
        Get the fit parameters.

        Returns:
            dict: The fit parameters.

        Raises:
            RuntimeError: If the fit has not been performed.
        """
        if self.fit_result:
            return self.fit_result.values
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def fit_errors(self):
        """
        Get the errors on the fit parameters.

        Returns:
            dict: The errors on the fit parameters.

        Raises:
            RuntimeError: If the fit has not been performed.
        """
        if self.fit_result:
            return self.fit_result.errors
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def log_likelihood_value(self):
        """
        Get the log-likelihood value of the fit.

        Returns:
            float: The log-likelihood value.

        Raises:
            RuntimeError: If the fit has not been performed.
        """
        if self.fit_result:
            return -self.fit_result.fval  # Negative because we minimize -log(L)
        else:
            raise RuntimeError("Fit has not been performed yet.")