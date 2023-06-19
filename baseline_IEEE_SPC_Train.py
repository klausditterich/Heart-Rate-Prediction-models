import os
from scipy import signal
import numpy as np
from scipy.signal import hilbert
import scipy.io
from scipy.signal import butter, lfilter
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
    print(listHR)
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
                c = listPPG[idx1:idx2]
                if len(c) == 1000:
                    b.append(c)
                    a.append(listHR[i][j])
                idx1 += increment1
                idx2 += increment1
    return b, a


def estimate_average_heartrate(s, sampling_frequency):
    peaks = find_peaks(s, height=0.87, distance=26, width=1, wlen=5)[0]

    instantaneous_rates = (sampling_frequency * 60) / np.diff(peaks)

    # remove instantaneous rates which are lower than 40, higher than 180
    selector = (instantaneous_rates > 40) & (instantaneous_rates < 180)
    return float(np.nan_to_num(instantaneous_rates[selector].mean())), peaks


def estimate_average_heartrate1(s, sampling_frequency):
    peaks = find_peaks(s, height=6, distance=35, width=1, wlen=5)[0]

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
    plt.xlabel('time (s)')
    ax.grid(True)
    ax.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.show()


def plot_signal1(s, sampling_frequency):
    avg, peaks = estimate_average_heartrate1(s, sampling_frequency)

    ax = plt.gca()
    ax.plot(np.arange(0, len(s) / sampling_frequency, 1 / sampling_frequency),
            s, label='Raw signal');
    xmin, xmax, ymin, ymax = plt.axis()
    ax.vlines(peaks / sampling_frequency, ymin, ymax, colors='r', label='P-T QRS detector')
    plt.xlim(0, len(s) / sampling_frequency)
    plt.ylabel('uV')
    plt.xlabel('time (s)')
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
    meansf = []
    meanst = []
    meansf1 = []
    meanst1 = []

    arraybetter = [0, 0, 0, 0]

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

        lecturaPPG[0] = butter_bandpass_filter(lecturaPPG[0], 1, 3.5, samplingfreqPPG, 2)
        prova = np.pad(lecturaPPG[0], (63, 63), 'symmetric')
        lecturafin = ppg_preprocessing(prova, samplingfreqPPG, view=False)
        arrayppg, arrayhr = sampling(lecturafin, lecturaHR, False)

        error_t = []
        error_f = []
        error_t1 = []
        error_f1 = []

        print('We are on subject: ', item)

        for i in range(len(arrayppg)):

            print('limpia')
            f, pxx = signal.periodogram(arrayppg[i], samplingfreqPPG, nfft=2 ** (len(arrayppg[i]) - 1).bit_length(),
                                        detrend=False)
            idx = np.argwhere((f >= 1) & (f <= 3))
            mask_ppg = np.take(f, idx)
            mask_pxx = np.take(pxx, idx)
            fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60

            error_f.append((abs(fft_hr - arrayhr[i])).item())
            print('Predicted freq: ', fft_hr, fft_hr / 60, 'Real: ', arrayhr[i], arrayhr[i] / 60, 'Error: ',
                  abs(fft_hr - arrayhr[i]))

            print('no limpia')
            f1, pxx1 = signal.periodogram(lecturaPPG[0][i * 250:1000 + i * 250], samplingfreqPPG,
                                          nfft=2 ** (len(arrayppg[i]) - 1).bit_length(),
                                          detrend=False)
            idx1 = np.argwhere((f1 >= 1) & (f1 <= 3))
            mask_ppg1 = np.take(f1, idx1)
            mask_pxx1 = np.take(pxx1, idx1)
            fft_hr1 = np.take(mask_ppg1, np.argmax(mask_pxx1, 0))[0] * 60

            error_f1.append((abs(fft_hr1 - arrayhr[i])).item())
            print('Predicted freq: ', fft_hr1, fft_hr1 / 60, 'Real: ', arrayhr[i], arrayhr[i] / 60, 'Error: ',
                  abs(fft_hr1 - arrayhr[i]))

            print('limpia')
            hr_t, _ = estimate_average_heartrate(arrayppg[i], samplingfreqPPG)
            error_t.append((abs(hr_t - arrayhr[i]).item()))
            print('Predicted temp: ', hr_t, 'Real: ', arrayhr[i], 'Error: ', abs(hr_t - arrayhr[i]))

            print('no limpia')
            hr_t1, _ = estimate_average_heartrate1(lecturaPPG[0][i * 250:1000 + i * 250], samplingfreqPPG)
            error_t1.append((abs(hr_t1 - arrayhr[i]).item()))
            print('Predicted temp: ', hr_t1, 'Real: ', arrayhr[i], 'Error: ', abs(hr_t1 - arrayhr[i]))
            # 0: freq neta, 1: temp neta, 2: freq bruta, 3: temp bruta
            if (error_t1[-1] < error_f[-1]) & (error_t1[-1] < error_t[-1]) & (error_t1[-1] < error_f1[-1]) & (error_t1[-1] < 10):
                arraybetter[3] += 1
            if (error_t[-1] < error_f[-1]) & (error_t[-1] < error_t1[-1]) & (error_t[-1] < error_f1[-1]) & (error_t[-1] < 10):
                arraybetter[1] += 1
            if (error_f[-1] < error_t1[-1]) & (error_f[-1] < error_t[-1]) & (error_f[-1] < error_f1[-1]) & (error_f[-1] < 10):
                arraybetter[0] += 1
            if (error_f1[-1] < error_f[-1]) & (error_f1[-1] < error_t[-1]) & (error_f1[-1] < error_f[-1]) & (error_f1[-1] < 10):
                arraybetter[2] += 1

        mean_f = sum(error_f) / len(error_f)
        std_f = std(error_f)
        print('Mean error freq :', mean_f)

        mean_t = sum(error_t) / len(error_t)
        std_t = std(error_t)
        print('Mean error temp :', mean_t)

        mean_f1 = sum(error_f1) / len(error_f1)
        std_f1 = std(error_f1)
        print('Mean error freq no limpia:', mean_f1)

        mean_t1 = sum(error_t1) / len(error_t1)
        std_t1 = std(error_t1)
        print('Mean error temp no limpia:', mean_t1)

        report.append(['Subject: ', item, 'Mean error freq: ', mean_f, 'Std error freq: ', std_f,
                       'Mean error temp: ', mean_t, 'Std error temp: ', std_t])
        meansf.append(mean_f)
        meanst.append(mean_t)

        meansf1.append(mean_f1)
        meanst1.append(mean_t1)

        lecturaPPG = lecturaPPG.clear()
        lecturaHR = lecturaHR.clear()

    print('The total mean freq is:', sum(meansf) / len(meansf), 'std: ', std(meansf))
    print('The total mean temp is:', sum(meanst) / len(meanst), 'std: ', std(meanst))

    print('The total mean freq bruta is:', sum(meansf1) / len(meansf1), 'std: ', std(meansf1))
    print('The total mean temp bruta is:', sum(meanst1) / len(meanst1), 'std: ', std(meanst1))

    print(arraybetter)

    savetxt('C:/Users/Usuario/Downloads/reportbaseline2train.csv', report, delimiter=',', comments='', fmt='%s')
