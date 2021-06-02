from scipy import signal
import numpy as np
import pywt
import matplotlib.pyplot as plt

from .mytracing import skeletor

PYWT_FIXED = True
if PYWT_FIXED:
    # if pywt library is changed to have trimmed wavelet functions, their support is from -1 to 1 exactly
    GAUS1_SUPPORT = 1
    GAUS2_SUPPORT = 1
else:
    # otherwise, the support is different for various wavelet base functions and is approximated by following number
    # NOTE: these are not exact and probably wrong.
    GAUS1_SUPPORT = 2.45
    GAUS2_SUPPORT = 2.68

MAX_SUPPORT = max(GAUS1_SUPPORT, GAUS2_SUPPORT)


def get_max_scale(signal_length):
    """
    Returns the max possible scale for the given signal length
    :param signal_length:
    :return:
    """
    if signal_length < 1:
        raise AssertionError(signal_length)
    return int(np.floor(signal_length/(2*MAX_SUPPORT)))


def create_w_coef_mask(w_coefs, order, epsilon=0.1, remove_inf=False):
    """
    Create a new matrix, the same shape as the wavelet coefficient one, but with zeros everywhere except for local
    maxima's. Epsilon here is used for ranking the strength of the local maxima.

    Assumes that the coefficient matrix coming in is already in absolute terms

    :param w_coefs: wavelet coefficient matrix
    :param epsilon: divided against the maxima, used for transparent ranking
    :param order: how many neighboors on a given row to look at to determine maxima
    :return: same shape array, see above
    """
    if remove_inf:
        w_coefs[w_coefs == np.inf] = 0.0

    mask = np.zeros_like(w_coefs, dtype=int)
    for n, row in enumerate(w_coefs):
        maxs = signal.argrelmax(row, order=order)[0]
        mask[n, maxs] = row[maxs] / epsilon

    return mask


def wtmm(sig, scales=None, wavelet=None, remove_inf=False, epsilon=0.1,
         order=1, proximity=9, plot=False):
    """
    Just a fast path to run perform_cwt and skeletor together

    :param sig: 1 dimensional array -- the signal to be hit with the wavelet
    :param scales: List of scales to run WT on
    :param wavelet: what wavelet to use as the mother
    :param epsilon: how to score the maxima's intensity (e.g. intensity / epsilon )
    :param order: how many neighbors to look at when finding the local maxima
    :param smallest_scale: the smallest scale to look at in search of skeletons
    :param proximity: how close to look for the next scale during skeleton construction
    :param plot: whether to plot the original CWT coefficient matrix as a heatmap
    :param corona_prox: proximity used to test for matched coronal loops
    :param top_threshold: percent distance from max-row to use for escaping cutoff

    :return: wtmm matrix (wt values, masked with zeros), full wt values matrix (w/o mask) and ridge lines
    """
    if proximity > len(sig) / 3:
        proximity = int(len(sig) / 3)
        print('proximity was too high, so it is reduced to ', proximity)

    scales = np.array(scales)
    max_scale = get_max_scale(len(sig))
    orig_max_scale = np.max(scales)
    #if max_scale != orig_max_scale:
    #    print('Warning in WTMM: max scale reduced from {} to {}'.format(orig_max_scale, max_scale))
    scales = scales[scales <= max_scale]

    mask, w_coef = perform_cwt(sig, scales=scales, wavelet=wavelet, epsilon=epsilon, order=order, plot=plot,
                               remove_inf=remove_inf)

    bifurcations = skeletor(mask, proximity=proximity, plot=plot, scales=scales)

    wtmm_matr = np.zeros_like(w_coef)

    for bif in bifurcations:
        pts = bif.get_points()
        rows, cols = zip(*pts)
        line_coefs = w_coef[rows, cols]
        # supremum algorithm
        # line_coefs = [max(line_coefs[:ii+1]) for ii, t in enumerate(line_coefs)]
        wtmm_matr[rows, cols] = line_coefs

    # wtmm_matr = np.abs(wtmm_matr)

    # normalize the rows to sum up to 1. Needed ONLY for some specific algorithms.
    # normalize_rows = False
    # if normalize_rows:
    #     row_sums = wtmm_matr.sum(axis=1)
    #     assert len(row_sums) == mask.shape[0]
    #     wtmm_matr = wtmm_matr / row_sums[:, np.newaxis]
    #
    #     if 0 in row_sums:
    #         max_scale_found = np.where(row_sums == 0)[0][0]
    #         print('max_scale_found: ', max_scale_found-1)
    #         wtmm_matr = wtmm_matr[:max_scale_found]

    return wtmm_matr, w_coef, bifurcations


def perform_cwt(sig, scales, wavelet, epsilon=0.1, order=1, plot=False, remove_inf=False):
    """
    Perform the continuous wavelet transform against the incoming signal. This function will normalize the signal
    (to 0-1 in the y axis) for you, as well as taking the -1 * abs( log( ) ) of the matrix that is found. Literature
    suggests that len/4 is a good balance for finding the bifurcations vs execution time

    This will automatically create the maxima only mask of the wavelet coef matrix for you. To see the original, use
    plot=True
    :param sig: 1 dimensional array -- the signal to be hit with the wavelet
    :param scales: List of scales for WT.
    :param wavelet: what wavelet to use as the mother
    :param epsilon: how to score the maxima's intensity (e.g. intensity / epsilon )
    :param order: how many neighbors to look at when finding the local maxima
    :param plot: whether to plot the original CWT coefficient matrix as a heatmap
    :return: the mask, see above
    """
    if np.isscalar(scales):
        scales = np.array([scales])
    else:
        if not is_ascending(scales):
            raise ValueError('Scales are not in ascending order: ', scales)

    if not isinstance(wavelet, pywt.ContinuousWavelet):
        raise ValueError('Please pass pywt.ContinuousWavelet object...')

    # normalize the signal to fit in the wavelet
    sig = normalize_signal(sig)

    # Run the transform
    w_coefs, freqs = pywt.cwt(sig, scales, wavelet)

    # Create the mask, keeping only the maxima
    # Here we use log values because for the very small scales, WT values are all so small that
    # it is impossible to distinguish local maxima
    mask = create_w_coef_mask(np.abs(w_coefs)*1000.0, order=order, epsilon=epsilon, remove_inf=remove_inf)
    # mask = create_w_coef_mask(np.abs(np.log(np.abs(w_coefs))), order=order, epsilon=epsilon, remove_inf=remove_inf)

    # set non-valid WT values to 0
    mask = clear_edges(mask, scales=scales, wavelet=wavelet)
    mask[mask > 0] = 1

    if plot:
        to_plot = np.ma.masked_array(mask, mask=(mask == 0))
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(to_plot)
        #plt.title('WTMM mask')
        if not np.all(np.diff(scales) == 1):
            scales_str = ['%.2f' % sc for sc in scales]
            plt.yticks(range(mask.shape[0]), scales_str)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.xaxis.tick_top()
        plt.xlabel('Dilation b')
        plt.ylabel('Scale a')
        plt.title('WTMM mask')
        plt.show()

    return mask, w_coefs


def clear_edges(cwtmatr, scales, wavelet):
    """
    It removes values from the sides of the skeleton that are not valid
    :param cwtmatr: matrix with WT coeffiecients
    :param scales: scales used
    :param wavelet: wavelet object, needed to know the right support
    :return: new WT matrix with invalid values equal to 0
    """
    if cwtmatr.shape[0] != len(scales):
        raise AssertionError('{} != {}'.format(cwtmatr.shape[0], len(scales)))
    if not isinstance(wavelet, pywt.ContinuousWavelet):
        raise ValueError('Wavelet must be given.')
    if wavelet.name == 'gaus1':
        support = GAUS1_SUPPORT
    elif wavelet.name == 'gaus2':
        support = GAUS2_SUPPORT
    else:
        raise NotImplementedError('Only gaus1 and gaus2 have known support values, but {} wavelet given.'.format(wavelet.name))

    for row, scale in enumerate(scales):
        cutoff = int(np.ceil(support * scale))
        cwtmatr[row, :cutoff] = 0
        cwtmatr[row, -cutoff:] = 0

    return cwtmatr


def normalize_signal(sig):
    """
    Normalizes signal by subtracting the mean.
    """
    return sig / np.mean(sig)


def is_ascending(lst):
    """
    Returns True if list is in ascending order.
    :param lst: list
    :return:
    """
    lst = np.array(lst)
    is_asc = (np.diff(lst) > 0).all()
    return is_asc
