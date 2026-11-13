"""Select a saved checkpoint using VALID only, within a fixed epoch budget."""
import math


def select_validation_best(records, max_epoch=5):
    if not 1 <= max_epoch <= 5:
        raise ValueError('The comparison budget must be between one and five epochs.')
    eligible=[row for row in records if 1 <= row['epoch'] <= max_epoch]
    if not eligible:
        raise ValueError('No completed VALID evaluation within the epoch budget.')
    if len({row['epoch'] for row in eligible}) != len(eligible):
        raise ValueError('Duplicate epoch records.')
    if not all(math.isfinite(row['validation_ndcg10']) for row in eligible):
        raise ValueError('Nonfinite VALID metric.')
    return max(eligible,key=lambda row:(row['validation_ndcg10'],-row['epoch']))
