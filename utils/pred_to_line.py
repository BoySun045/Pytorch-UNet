import numpy as np
from utils.df_loss import df_in_neighbor_loss, l1_loss_fn, denormalize_df
from utils.regression_loss import reverse_log_transform
from utils.dataset_statistics import reverse_label_mask

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



def df_cls_to_line_weight(df, cls_mask, 
                          df_neighborhood=10, 
                          threshold=0.5,
                          bin_edges=None):
    
    # fisrt, get binary mask from distance field
    bin_mask = df_to_linemap(df, df_neighborhood, threshold)
    cls_mask = cls_mask.argmax(dim=1, keepdim=False)
    cls_mask = cls_mask.squeeze().cpu().numpy()
    reverse_mask = reverse_label_mask(cls_mask, bin_edges)
    print("reverse mask shape: ", reverse_mask.shape)
    print("reverse mask max: ", reverse_mask.max())
    print("reverse mask min: ", reverse_mask.min())
    weight = np.zeros_like(reverse_mask)
    weight[bin_mask > 0] = reverse_mask[bin_mask > 0]

    return bin_mask, weight