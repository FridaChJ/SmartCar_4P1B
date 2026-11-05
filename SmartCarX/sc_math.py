"""Small numeric helpers shared by the SmartCar modules."""


def sc_clamp(value, low, high):
    """Clamp a numeric value into a closed range."""
    return max(low, min(high, value))
