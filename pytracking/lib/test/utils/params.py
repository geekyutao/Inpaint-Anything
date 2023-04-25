from lib.utils import TensorList
import random


class TrackerParams:
    """Class for tracker parameters."""
    def set_default_values(self, default_vals: dict):
        for name, val in default_vals.items():
            if not hasattr(self, name):
                setattr(self, name, val)

    def get(self, name: str, *default):
        """Get a parameter value with the given name. If it does not exists, it return the default value given as a
        second argument or returns an error if no default value is given."""
        if len(default) > 1:
            raise ValueError('Can only give one default value.')

        if not default:
            return getattr(self, name)

        return getattr(self, name, default[0])

    def has(self, name: str):
        """Check if there exist a parameter with the given name."""
        return hasattr(self, name)


class FeatureParams:
    """Class for feature specific parameters"""
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise ValueError

        for name, val in kwargs.items():
            if isinstance(val, list):
                setattr(self, name, TensorList(val))
            else:
                setattr(self, name, val)


def Choice(*args):
    """Can be used to sample random parameter values."""
    return random.choice(args)
