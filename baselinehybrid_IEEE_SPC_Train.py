import os
from scipy import signal
import numpy as np
from scipy.signal import hilbert
import scipy.io
from scipy.signal import butter, lfilter
from mne.preprocessing.ecg import qrs_detector
from numpy import savetxt
from numpy import std
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def ppg_preprocessing(data, fs, view=False):
    m_avg = lambda t, x, w: (np.asarray([t[i] for i in range(w, len(x) - w)]),
                             np.convolve(x, np.ones((2 * w + 1,)) / (2 * w + 1),
                                         mode='valid'))

    time = np.linspace(0, len(data), len(data))

    # moving average
    w_size = int(fs * .5)
    mt, ms = m_avg(time, data, w_size)

    # remove global modulation
    sign = data[w_size: -w_size] - ms

    # compute signal envelope
    analytical_signal = np.abs(hilbert(sign))

    fs = len(sign) / (max(mt) - min(mt))
    w_size = int(fs)

    # moving average of envelope
    mt_new, mov_avg = m_avg(mt, analytical_signal, w_size)

    # remove envelope
    signal_pure = sign[w_size: -w_size] / mov_avg

    if view:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)
        ax1.plot(time, data, "b-", label="Original")
        ax1.legend(loc='best')

        ax2.plot(mt, sign, 'r-', label="Pure signal")
        ax2.plot(mt_new, mov_avg, 'b-', label='Modulation', alpha=.5)
        ax2.legend(loc='best')
        ax2.set_title("Raw -> filtered", fontsize=14)  # , fontweight="bold")

        ax3.plot(mt_new, signal_pure, "g-", label="Demodulated")
        ax3.set_xlim(0, mt[-1])
        ax3.set_title("Raw -> filtered -> demodulated", fontsize=14)  # , fontweight="bold")

        ax3.set_xlabel("Samples", fontsize=14)  # common axis label
        ax3.legend(loc='best')

        plt.show()
        plt.clf()

    return signal_pure


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def linearinterpolaion(t0, t1, t, y0, y1):
    return (y0 * (t1 - t) + y1 * (t - t0)) / (t1 - t0)


def sampling(listPPG, listHR, aug):
    a = []
    b = []
    increment1 = 250
    increment2 = 50
    for i in range(len(listHR)):
        idx1 = 0
        idx2 = 1000
        for j in range(len(listHR[i])):
            if aug is True:
                if j <= len(listHR[i]) - 2:
                    val = 0  # iterator over values
                    for k in range(int(increment1 / increment2)):
                        d = listHR[i][j].item()
                        e = listHR[i][j + 1].item()
                        hr = np.array([linearinterpolaion(idx1, idx1 + increment1, idx1 + increment2 * val, d, e)])
                        ppg = listPPG[i][idx1 + increment2 * val:idx2 + increment2 * val]
                        if len(ppg) == 1000:
                            b.append(ppg)
                            a.append(hr)
                        val += 1
                    idx1 += increment1
                    idx2 += increment1
            else:
                c = listPPG[i][idx1:idx2]
                if len(c) == 1000:
                    b.append(c)
                    a.append(listHR[i][j])
                idx1 += increment1
                idx2 += increment1
    return b, a


def estimate_average_heartrate(s, sampling_frequency):
    peaks = find_peaks(s, distance=33)[0]

    instantaneous_rates = (sampling_frequency * 60) / np.diff(peaks)

    # remove instantaneous rates which are lower than 40, higher than 180
    selector = (instantaneous_rates > 40) & (instantaneous_rates < 180)
    return float(np.nan_to_num(instantaneous_rates[selector].mean())), peaks


def plot_signal(s, sampling_frequency):

    avg, peaks = estimate_average_heartrate(s, sampling_frequency)

    ax = plt.gca()
    ax.plot(np.arange(0, len(s) / sampling_frequency, 1 / sampling_frequency),
            s, label='Raw signal');
    xmin, xmax, ymin, ymax = plt.axis()
    ax.vlines(peaks / sampling_frequency, ymin, ymax, colors='r', label='P-T QRS detector')
    plt.xlim(0, len(s) / sampling_frequency)
    plt.ylabel('uV')
    plt.xlabel('Samples')
    ax.grid(True)
    ax.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.show()


if __name__ == '__main__':

    samplingfreqPPG = 125

    samplingperiodPPG = 1 / samplingfreqPPG

    PATH_train = 'C:/Users/Usuario/Downloads/SPC_2015_dataset/Training_data'

    person = ['DATA_01', 'DATA_02', 'DATA_03', 'DATA_04', 'DATA_05', 'DATA_06', 'DATA_07',
              'DATA_08', 'DATA_09', 'DATA_10', 'DATA_11', 'DATA_12']

    report = []
    means = []

    hr1 = 'BPM'

    os.chdir(PATH_train)

    for item in person:

        lecturaPPG = []
        lecturaHR = []

        for file in os.listdir():
            file_path = f"{PATH_train}/{file}"
            filename = os.path.basename(file_path)
            if item in filename:
                mat = scipy.io.loadmat(file_path)
                mat_names = [name for name in mat.keys() if not name.startswith('__')]
                mat_names = mat_names[0]
                if hr1 not in filename:
                    b = mat[mat_names][1]
                    lecturaPPG.append(b)
                else:
                    c = mat[mat_names]
                    lecturaHR.append(c)

        lecturaPPG[0] = butter_bandpass_filter(lecturaPPG[0], 1, 3.4, samplingfreqPPG, 2)
        # lecturaPPG[0] = ppg_preprocessing(lecturaPPG[0], samplingfreqPPG, view=False)
        arrayppg, arrayhr = sampling(lecturaPPG, lecturaHR, False)

        error = []
        arraypredhr = []

        print('We are on subject: ', item)

        for i in range(len(arrayppg)):
            f, pxx = signal.periodogram(arrayppg[i], samplingfreqPPG, nfft=2 ** (len(arrayppg[i]) - 1).bit_length(),
                                        detrend=False)
            idx = np.argwhere((f >= 1) & (f <= 3))

            mask_ppg = np.take(f, idx)
            mask_pxx = np.take(pxx, idx)
            fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60

            hr_t, _ = estimate_average_heartrate(arrayppg[i], samplingfreqPPG)

            if i == 0:
                hr = fft_hr
                arraypredhr.append(hr)
            else:
                if abs(fft_hr - arraypredhr[-1]) < abs(hr_t - arraypredhr[-1]):
                    hr = fft_hr
                    arraypredhr.append(hr)
                else:
                    hr = hr_t
                    arraypredhr.append(hr)

            error.append((abs(hr - arrayhr[i])).item())
            print('Predicted hr: ', hr, hr / 60, 'Real: ', arrayhr[i], arrayhr[i] / 60, 'Error: ',
                  abs(hr - arrayhr[i]))

        meanexp = sum(error) / len(error)
        std_f = std(error)
        print('Mean error of subject is :', meanexp)

        report.append(['Subject: ', item, 'Mean error: ', meanexp, 'Std error: ', std_f])

        means.append(meanexp)

        lecturaPPG = lecturaPPG.clear()
        lecturaHR = lecturaHR.clear()

    print('The total mean is:', sum(means) / len(means))
    savetxt('C:/Users/Usuario/Downloads/reportIEEESPCtrain.csv', report, delimiter=',', comments='', fmt='%s')
