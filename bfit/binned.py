from .base import FitBase
import numpy as np

class CompositeModel(FitBase):
    """
    Composite model class for combining multiple fit components.

    This class allows the creation of complex models by combining multiple
    simpler components.

    Attributes:
        components (list): List of (component, weight) tuples.
    """

    def __init__(self, bins, counts, param_limits=None):
        """
        Initialize the CompositeModel instance.

        Args:
            bins (array-like): The bin edges of the histogram.
            counts (array-like): The counts in each bin.
            param_limits (dict, optional): Limits on the fit parameters.
        """
        super().__init__(bins, counts, param_limits)
        self.components = []

    def add_component(self, component, weight=1.0):
        """
        Add a component to the composite model.

        Args:
            component (Component): The component to add.
            weight (float, optional): The weight of the component in the model.
        """
        self.components.append((component, weight))

    def fit_function(self, x_vals, *params):
        """
        Composite fit function combining all components.

        Args:
            x_vals (array-like): The x-values to evaluate the function at.
            *params: The parameters for all components.

        Returns:
            array-like: The y-values of the composite function.
        """
        result = np.zeros_like(x_vals)
        param_index = 0
        for component, weight in self.components:
            n_params = component.n_params
            component_params = params[param_index:param_index+n_params]
            component.normalize(self.x_min, self.x_max, *component_params)
            result += weight * component(x_vals, *component_params)
            param_index += n_params
        return result

    def chi_squared(self, *params):
        """
        Calculate chi-squared value for the composite model.

        This method overrides the base class method to include scaling.

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
        
        # Scale the prediction to match the total number of events
        scale_factor = np.sum(y_vals_masked) / np.sum(prediction)
        scaled_prediction = prediction * scale_factor
        
        residuals_squared = ((y_vals_masked - scaled_prediction) / y_errs_masked) ** 2
        return np.sum(residuals_squared)
