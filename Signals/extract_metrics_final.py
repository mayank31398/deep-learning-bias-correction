import os
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, medfilt, savgol_filter, filtfilt, butter

from peakdetect import peakdetect, peakdetect_parabola, peakdetect_spline



def Mark(diff_x):
    found = False
    start = -1
    l = []
    for i in range(diff_x.shape[0]):
        if(diff_x[i] == 0 and not found):
            found = True
            start = i
        elif(diff_x[i] == 1 and found):
            found = False
            end = i

            if(end - start > 50):
                l.append([start, end])
    return l


def detrend(e):
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    order = 4
    fs = 15
    cutoff = 1
    e = butter_lowpass_filter(e, cutoff, fs, order)

    return e


def detectgci(data):
    degg = np.insert(np.diff(data), 0, 0)
    # degg_max = np.max(degg)
    # degg /= degg_max
    # degg *= 1000
    # out = np.array(peakdetect(degg, lookahead=32))
    out = np.array(peakdetect_spline(degg, range(len(degg))))
    out = out[1, :, 0].astype(np.int)  # Extracted peaks

    threshold = -1 / 9 * \
        pd.Series(np.abs(degg)).nlargest(100).mean()  # Soft threshold
    dec = degg[out] <= threshold  # Filter out spurious peaks
    fin = out[
        np.nonzero(1 * dec)
    ]  # Extract pertinent peak positions from original extracted peaks in out

    return fin


def smooth(x, window_len=100, window='hanning'):
    #     ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    if window_len < 3:
        return x

    # s = np.r_[x[window_len - 1: 0: -1], x, x[-2: -window_len - 1: -1]]
    s = x
    if window == "median":
        y = medfilt(s, kernel_size=window_len)
    elif window == "savgol":
        y = savgol_filter(s, window_len, 0)
    else:
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='same')

    return y


def detectscipygci(data):
    degg = np.insert(np.diff(data), 0, 0)
    out, _ = find_peaks(-degg, distance=10, prominence=(None, None))
    return out


def corrected_naylor_metrics(ref_signal, est_signal):
    # Settings
    # TODO: precise values to be decided later

    try:
        assert np.squeeze(ref_signal).ndim == 1
        assert np.squeeze(est_signal).ndim == 1
    except:
        return

    ref_signal = np.squeeze(ref_signal)
    est_signal = np.squeeze(est_signal)

    min_f0 = 50
    max_f0 = 500
    min_glottal_cycle = 1 / max_f0
    max_glottal_cycle = 1 / min_f0

    nHit = 0
    nMiss = 0
    nFalse = 0
    nCycles = 0
    highNumCycles = 100000
    estimation_distance = np.full(highNumCycles, np.nan)

    ref_fwdiffs = np.zeros_like(ref_signal)
    ref_bwdiffs = np.zeros_like(ref_signal)

    ref_fwdiffs[:-1] = np.diff(ref_signal)
    ref_fwdiffs[-1] = max_glottal_cycle
    ref_bwdiffs[1:] = np.diff(ref_signal)
    ref_bwdiffs[0] = max_glottal_cycle

    for i in range(len(ref_fwdiffs)):
        # m in original file
        ref_cur_sample = ref_signal[i]
        ref_dist_fw = ref_fwdiffs[i]
        ref_dist_bw = ref_bwdiffs[i]

        # Condition to check for valid larynx cycle
        # TODO: Check parity of differences, neg peak <-> gci, pos peak <-> goi
        # TODO: Check applicability of strict inequality
        dist_in_allowed_range = (
            0 <= ref_dist_fw <= np.inf
            and 0 <= ref_dist_bw <= np.inf
        )

        if dist_in_allowed_range:

            cycle_start = ref_cur_sample - ref_dist_bw / 2
            cycle_stop = ref_cur_sample + ref_dist_fw / 2

            est_GCIs_in_cycle = est_signal[
                np.logical_and(est_signal > cycle_start,
                               est_signal < cycle_stop)
            ]
            n_est_in_cycle = np.count_nonzero(est_GCIs_in_cycle)

            nCycles += 1

            if n_est_in_cycle == 1:
                nHit += 1
                estimation_distance[nHit] = est_GCIs_in_cycle[0] - \
                    ref_cur_sample
            elif n_est_in_cycle < 1:
                nMiss += 1
            else:
                nFalse += 1

    estimation_distance = estimation_distance[np.invert(
        np.isnan(estimation_distance))]

    identification_rate = nHit / nCycles
    miss_rate = nMiss / nCycles
    false_alarm_rate = nFalse / nCycles
    identification_accuracy = (
        0 if np.size(estimation_distance) == 0 else np.std(estimation_distance)
    )

    return {
        "identification_rate": identification_rate,
        "miss_rate": miss_rate,
        "false_alarm_rate": false_alarm_rate,
        "identification_accuracy": identification_accuracy,
    }


def extract_metrics(true_egg, estimated_egg, fs=15):
    # true_gci = np.nonzero(true_egg)
    # estimated_gci = np.nonzero(estimated_egg)

    # metrics = corrected_naylor_metrics(true_gci, estimated_gci)
    # return metrics

    # estimated_egg = smooth(estimated_egg, window_len=20, window="hanning")

    true_gci = true_egg.astype(int)
    estimated_gci = estimated_egg.astype(int)

    true_gci_ = np.zeros(true_egg.shape[0])
    estimated_gci_ = np.zeros(estimated_egg.shape[0])

    true_egg = true_egg / np.max(np.abs(true_egg))
    estimated_egg = estimated_egg / np.max(np.abs(estimated_egg))

    true_gci_[true_gci] = 1
    estimated_gci_[estimated_gci] = 1

    plt.figure()
    plt.plot(true_gci_, "r")
    plt.plot(estimated_gci_, "b")

    plt.figure()
    plt.plot(true_egg, "r")
    plt.plot(estimated_egg, "b")

    plt.show()


def main():
    files = os.listdir("Data/Speech")

    mean_idr = 0
    mean_msr = 0
    mean_far = 0
    mean_ida = 0
    count = 0
    for i in files:
        speech = np.load("Data/Speech/" + i)
        peak = np.load("Data/Peaks/" + i)
        e = detrend(speech)
        x = np.mean(np.abs(speech))
        x = np.abs(speech) < 0.25 * np.max(np.abs(speech))
        diff_x = np.diff(x)
        l = Mark(diff_x)
        y = np.zeros(x.shape[0])
        for i in l:
            y[i[0]:i[1]] = 1
        y = np.abs(np.diff(y))

        result = extract_metrics(peak, y)

        if(result != None):
            print(i, result)

            mean_idr += result["identification_rate"]
            mean_msr += result["miss_rate"]
            mean_far += result["false_alarm_rate"]
            mean_ida += result["identification_accuracy"]
            count += 1

    mean = {
        "identification_rate": mean_idr / count,
        "miss_rate": mean_msr / count,
        "false_alarm_rate": mean_far / count,
        "identification_accuracy": mean_ida / count,
    }

    print()
    print("Mean", mean)


if(__name__ == "__main__"):
    main()
