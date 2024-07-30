import numpy as np
from utils.df_loss import df_in_neighbor_loss, l1_loss_fn, denormalize_df
from utils.regression_loss import reverse_log_transform

def df_to_linemap(df, df_neighborhood=10, threshold=0.5):
    """
    Convert a distance field to a line map.

    Args:
    df (numpy array): The distance field as a 2D numpy array.
    threshold (float): Threshold value to detect lines.

    Returns:
    linemap (numpy array): A binary mask where lines are marked.
    """
    df = denormalize_df(df, df_neighborhood=df_neighborhood)
    df = df.squeeze().cpu().numpy()
    linemap = np.zeros_like(df)
    linemap[df < threshold] = 1

    return linemap


def df_wf_to_linemap(df, wf, 
                     df_neighborhood=10, threshold=0.5):
    
    # fisrt, get binary mask from distance field
    bin_mask = df_to_linemap(df, df_neighborhood, threshold)
    wf_pred = reverse_log_transform(wf)
    wf_pred = wf_pred.squeeze().cpu().numpy()
    linemap = np.zeros_like(wf_pred)
    linemap[bin_mask > 0] = wf_pred[bin_mask > 0]
    print("linemap max: ", linemap.max())
    print("linemap min: ", linemap.min())

    return linemap