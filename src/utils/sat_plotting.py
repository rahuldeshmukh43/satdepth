import bisect
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
import numpy as np
from PIL import Image

from skimage import exposure

from satdepth.src.utils.sat_metrics import _compute_conf_thresh, compute_precision_recall

# --- VISUALIZATION --- #
def draw_matches(data, config, mode='evaluation'):
    figures = make_matching_figures(data, config, mode=mode)

    # convert figures to images
    imgs = [convert_figure_to_img(f) for f in figures[mode]]
    imgs = np.concatenate(imgs, axis=0) #[H,W,3]
    imgs = torch.from_numpy(imgs)
    # stack the images together
    # img = torchvision.utils.make_grid(imgs, nrow=1)
    return imgs #[H,W,3]

def convert_figure_to_img(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    figure.savefig(buf, format='jpg')
    buf.seek(0)
    image = np.array(Image.open(buf), dtype=np.uint8) #[H,W,3]
    # image = alpha_to_color(image) #[H,W,3]
    buf.close()
    plt.close('all')
    return image

def make_matching_figure(img0, 
                         img1, 
                         mkpts0, 
                         mkpts1, 
                         color, 
                         epi_err, 
                         conf_thr, 
                         kpts0=None, 
                         kpts1=None, 
                         text=[], 
                         dpi=75, 
                         path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                             (fkpts0[i, 1], fkpts1[i, 1]),
                                             transform=fig.transFigure, c=color[i], linewidth=1)
                     for i in range(len(mkpts0)) if epi_err[i] < conf_thr]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    # txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    txt_color = 'r'
    # fig.text(
    #     0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
    #     fontsize=15, va='top', ha='left', color=txt_color)
    fig.suptitle( '\n'.join(text),
        fontsize=15, color=txt_color)
    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close('all')
    else:
        plt.close('all')
        return fig

def _make_evaluation_figure(data, b_id, alpha='dynamic', conf_thr=None):
    b_mask = data['m_bids'] == b_id
    if not conf_thr:
        conf_thr = _compute_conf_thresh(data)

    img0 = (data["image0"][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    img1 = (data["image1"][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)

    img0 = exposure.equalize_hist(img0)
    img1 = exposure.equalize_hist(img1)

    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    precision, recall, n_correct, n_gt_matches, len_correct_mask, epi_errs = compute_precision_recall(data, b_id,epi_err_thr=conf_thr)

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len_correct_mask)
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)

    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)} Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]

    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, epi_errs, conf_thr, text=text)
    return figure

# def _make_confidence_figure(data, b_id):
#     raise NotImplementedError()

def make_matching_figures(data, config, mode='evaluation'):
    """ 
    Make matching figures for a batch.
    Args:
        data (Dict): a batch updated by yhe lightning module
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence']  # 'confidence'
    figures = {mode: []}
    for b_id in range(min(data["image0"].size(0), config["trainer"]["max_pairs_to_plot"])):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config["trainer"]["plot_matches_alpha"],
                conf_thr=config["trainer"]["epipolar_thr"])
        # elif mode == 'confidence':
        #     fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    """
    this creates a straight line between red and green, the smaller the error the more the x and hence more green
    """
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)

# def error_colormap(err, thr, alpha=1.0):
#     x = np.clip(err / (thr * 2), 0, 1)
#     cmap = matplotlib.cm.get_cmap("jet")
#     return cmap(x)
