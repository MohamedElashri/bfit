from .base import UnbinnedFitBase
import numpy as np


class UnbinnedCompositeModel(UnbinnedFitBase):
    """
    Composite model class for combining multiple unbinned fit components.

    This class allows the creation of complex unbinned models by combining
    multiple simpler components.

    Attributes:
        components (list): List of (component, weight) tuples.
    """

    def __init__(self, data, param_limits=None):
        """
        Initialize the UnbinnedCompositeModel instance.

        Args:
            data (array-like): The raw data points.
            param_limits (dict, optional): Limits on the fit parameters.
        """
        super().__init__(data, param_limits)
        self.components = []

    def add_component(self, component, weight=1.0):
        """
        Add a component to the composite model.

        Args:
            component (Component): The component to add.
            weight (float, optional): The weight of the component in the model.
        """
        self.components.append((component, weight))

    def pdf(self, x_vals, *params):
        """
        Composite probability density function combining all components.

        Args:
            x_vals (array-like): The x-values to evaluate the PDF at.
            *params: The parameters for all components.

        Returns:
            array-like: The y-values of the composite PDF.
        """
        result = np.zeros_like(x_vals)
        param_index = 0
        for component, weight in self.components:
            n_params = component.n_params
            component_params = params[param_index:param_index+n_params]
            component.normalize(self.x_min, self.x_max, *component_params)
            result += weight * component.pdf(x_vals, *component_params)
            param_index += n_params
        return result

    def log_likelihood(self, *params):
        """
        Calculate log-likelihood value for the composite model.

        Args:
            *params: The current fit parameters.

        Returns:
            float: The negative log-likelihood value.
        """
        epsilon = 1e-10  # Small value to prevent log(0)
        return -np.sum(np.log(self.pdf(self.data, *params) + epsilon))