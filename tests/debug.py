# 3rd party
import pathlib
import traceback
from typing import List, Optional
import numpy as np
from skimage.feature import peak_local_max

# Our own imports
from multilineage_organoid import signals as sg
from multilineage_organoid.io import DataReader
from multilineage_organoid.utils import *

SIGNAL_TYPE = 'F/F0'  # F-F0, F/F0, F-F0/F0

LINEAR_MODEL = 'ransac'  # 'least_squares' or 'ransac' or 'exp_ransac'

DATA_TYPE = 'ephys'  # one of 'ca', 'ephys'

FILTER_ORDER = 1  # Order of the butterworth filter
FILTER_CUTOFF = 4  # Hz cutoff for the filter
MIN_STATS_SCORE = 0.2  # Cutoff for peaks to be sufficiently wavy

FIGSIZE = 8  # inches - the width of a (square) figure panel

PLOT_SUFFIX = '.png'

SAMPLES_AROUND_PEAK = 10  # Minimum number of samples around a peak before the next peak can start

TIME_SCALE = 1000  # milliseconds / second

DEBUG_OPTIMIZER = True  # If True, print optimizer debug messages

def debug_calc_signal_stats(time: np.ndarray,
                      signals: np.ndarray,
                      time_scale: float = TIME_SCALE,
                      samples_around_peak: int = SAMPLES_AROUND_PEAK,
                      skip_model_fit: bool = False) -> Tuple[List]:
    """ Calculate useful stats for a set of signals

    :param ndarray time:
        The n x 1 time array
    :param ndarray signals:
        The n x m signal array
    :param float time_scale:
        The conversion of time to seconds
    :param int samples_around_peak:
        Minimum number of samples around a peak before the next peak
    :returns:
        The list of valid signal indicies and the list of stats for each peak found in the signal array
    """

    all_peaks = []
    valid_signal_indicies = []

    for i in range(signals.shape[1]):
        signal = signals[:, i]

        # Mask out invalid values
        sigmask = np.isfinite(signal)
        if not np.any(sigmask):
            print(f'Empty signal in sample: {i}')
            continue

        # Mask out signals with discontinuities
        offset_st, offset_ed = np.nonzero(sigmask)[0][[0, -1]]
        if not np.all(sigmask[offset_st:offset_ed+1]):
            invalid_indices = np.nonzero(~sigmask[offset_st:offset_ed+1])
            print(f'Non-contiguous signal in sample {i}: {invalid_indices}')
            continue

        signal_finite = signal[sigmask]
        time_finite = time[sigmask]

        peak_indicies = peak_local_max(signal_finite,
                                       min_distance=samples_around_peak,
                                       threshold_rel=0.8)

        peaks = sg.refine_signal_peaks(time_finite, signal_finite, peak_indicies,
                                    offset=offset_st,
                                    time_scale=time_scale,
                                    skip_model_fit=skip_model_fit)
        
        all_peaks.append(peaks)
        valid_signal_indicies.append(i)
    for trace in all_peaks:
        print("TRACE")
        for p in trace:
            print(str(p['peak_index']) + " " + str(p["total_wave_time"]))
    return valid_signal_indicies, all_peaks
 
def debug_filter_datafile(
                    infile: pathlib.Path = pathlib.Path("debug_2_cols.csv"),
                    data_type: str = DATA_TYPE,
                    signal_type: str = SIGNAL_TYPE,
                    filter_cutoff: float = FILTER_CUTOFF,
                    filter_order: float = FILTER_ORDER,
                    plot_types: Optional[List[str]] = None,
                    time_scale: float = TIME_SCALE,
                    skip_detrend: bool = False,
                    skip_lowpass: bool = False,
                    skip_model_fit: bool = False,
                    samples_around_peak: int = SAMPLES_AROUND_PEAK,
                    flip_signal: bool = False,
                    level_shift: Optional[str] = None,
                    plotfile: Optional[pathlib.Path] = None,
                    linear_model: str = LINEAR_MODEL):
    try:
        time, raw_signals = DataReader(data_type).read_infile(
            infile, flip_signal=flip_signal, level_shift=level_shift)
    except OSError as err:
        print(f'Error {err} reading "{infile}"')
        if DEBUG_OPTIMIZER:
            traceback.print_exc()
        return None

    dt = np.nanmedian(time[1:] - time[:-1]) / time_scale  # Seconds
    sample_rate = 1.0 / dt

    signals, trend_lines = sg.remove_trend(raw_signals,
                                        signal_type=signal_type,
                                        skip_detrend=skip_detrend,
                                        linear_model=linear_model)
    if not skip_lowpass:
        signals = lowpass_filter(signals,
                                 sample_rate=sample_rate,
                                 cutoff=filter_cutoff,
                                 order=filter_order)
        
    signal_indicies, stats = debug_calc_signal_stats(time, signals,
                                               time_scale=time_scale,
                                               samples_around_peak=samples_around_peak,
                                               skip_model_fit=skip_model_fit)
    print(signal_indicies)

    raw_signals = raw_signals[:, signal_indicies]
    signals = signals[:, signal_indicies]

    stats = sg.select_top_stats(stats)
    stats = sg.add_summary_stats(stats, time_scale=time_scale)

    print([x["signal_stats"][1:-1] for x in stats])

debug_filter_datafile(skip_detrend = True, skip_lowpass=True)
