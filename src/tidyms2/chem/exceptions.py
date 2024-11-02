"""Chemistry related exceptions."""


class IsotopeError(ValueError):
    """Exception raised when an invalid isotope string is passed."""


class InvalidFormula(ValueError):
    """Exception raised when an invalid formula string is passed."""
