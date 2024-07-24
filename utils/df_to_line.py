import numpy as np


def df_to_linemap(df, threshold=0.5):
    """
    Convert a distance field to a line map.

    Args:
    df (numpy array): The distance field as a 2D numpy array.
    threshold (float): Threshold value to detect lines.

    Returns:
    linemap (numpy array): A binary mask where lines are marked.
    """
    linemap = (df < threshold).astype(np.uint8)
    return linemap