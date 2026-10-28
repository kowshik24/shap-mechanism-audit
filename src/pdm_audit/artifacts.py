"""Compare reproduced scientific records while retaining explicit runtime changes."""
import math


def scientific_records_match(original, reproduced):
    """Only known runtime leaves may differ; scientific values remain binding."""
    if isinstance(original, dict):
        if not isinstance(reproduced, dict) or original.keys() != reproduced.keys():
            return False
        return all((key in {'seconds', 'seconds_per_run'} and
                    type(value) in (int, float) and type(reproduced[key]) in (int, float)) or
                   scientific_records_match(value, reproduced[key])
                   for key, value in original.items())
    if isinstance(original, list):
        return (isinstance(reproduced, list) and len(original) == len(reproduced) and
                all(scientific_records_match(a, b) for a, b in zip(original, reproduced)))
    if type(original) in (int, float) and type(reproduced) in (int, float):
        return math.isclose(original, reproduced, rel_tol=1e-10, abs_tol=1e-12)
    return type(original) is type(reproduced) and original == reproduced
