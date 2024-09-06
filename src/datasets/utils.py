"""Utility functions for generating datasets."""

import hashlib


def generate_unique_id_for_parameter_set(*args) -> str:
    """Generate an ID that identifies the set of parameters used for generation."""
    param_string = ",".join(map(str, args))
    return hashlib.md5(param_string.encode(), usedforsecurity=False).hexdigest()[:7]
