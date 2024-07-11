import numpy as np

class FitModels:
    """
    Contains various probability distribution functions and background models.

    This class provides a collection of common statistical distributions and
    background models used in fitting procedures.

    Attributes:
        bin_width (float): The width of histogram bins.
        x_min (float): The minimum x-value of the fit range.
        x_max (float): The maximum x-value of the fit range.
    """
    
    def __init__(self, bin_width=1, x_min=0, x_max=1):
        """
        Initialize the FitModels instance.

        Args:
            bin_width (float): The width of histogram bins.
            x_min (float): The minimum x-value of the fit range.
            x_max (float): The maximum x-value of the fit range.
        """
        self.bin_width = bin_width
        self.x_min = x_min
        self.x_max = x_max

    def gaussian(self, x, mean, std_dev):
        """
        Gaussian distribution function.

        Args:
            x (array-like): The x-values.
            mean (float): The mean of the distribution.
            std_dev (float): The standard deviation of the distribution.

        Returns:
            array-like: The y-values of the Gaussian distribution.
        """
        return np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2)) / np.sqrt(2 * np.pi * std_dev ** 2)

    def exponential(self, x, amplitude, decay):
        """
        Exponential distribution function.

        Args:
            x (array-like): The x-values.
            amplitude (float): The amplitude of the exponential.
            decay (float): The decay rate of the exponential.

        Returns:
            array-like: The y-values of the exponential distribution.
        """
        return amplitude * np.exp(-decay * (x - self.x_min))

    def linear(self, x, slope, intercept):
        """
        Linear function.

        Args:
            x (array-like): The x-values.
            slope (float): The slope of the line.
            intercept (float): The y-intercept of the line.

        Returns:
            array-like: The y-values of the linear function.
        """
        return slope * x + intercept

    def parabola(self, x, a, b, c):
        """
        Parabolic function.

        Args:
            x (array-like): The x-values.
            a (float): The coefficient of x^2.
            b (float): The coefficient of x.
            c (float): The constant term.

        Returns:
            array-like: The y-values of the parabolic function.
        """
        return a * (x ** 2) + b * x + c

    def breit_wigner(self, x, mass, width):
        """
        Breit-Wigner distribution function.

        Args:
            x (array-like): The x-values.
            mass (float): The mass parameter.
            width (float): The width parameter.

        Returns:
            array-like: The y-values of the Breit-Wigner distribution.
        """
        return (1 / np.pi) * (0.5 * width) / ((x - mass) ** 2 + (0.5 * width) ** 2)

    def crystal_ball(self, x, alpha, n, mean, std_dev):
        """
        Crystal Ball function.

        Args:
            x (array-like): The x-values.
            alpha (float): The alpha parameter.
            n (float): The n parameter.
            mean (float): The mean of the Gaussian core.
            std_dev (float): The standard deviation of the Gaussian core.

        Returns:
            array-like: The y-values of the Crystal Ball function.
        """
        A = (n / np.abs(alpha)) ** n * np.exp(-alpha ** 2 / 2)
        B = n / np.abs(alpha) - np.abs(alpha)
        return np.where((x - mean) / std_dev > -alpha,
                        np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2)),
                        A * (B - (x - mean) / std_dev) ** (-n))

    def argus_bg(self, x, m0, c, p):
        """
        ARGUS background function.

        Args:
            x (array-like): The x-values.
            m0 (float): The kinematic limit.
            c (float): The curvature parameter.
            p (float): The power parameter.

        Returns:
            array-like: The y-values of the ARGUS background function.
        """
        z = 1 - (x / m0) ** 2
        return np.where(z > 0, x * np.sqrt(z) * np.exp(c * z ** p), 0)


