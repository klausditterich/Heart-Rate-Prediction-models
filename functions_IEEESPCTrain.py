import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from scipy.signal import hilbert
import scipy.io
from scipy.signal import butter, lfilter
import matplotlib
matplotlib.use('TkAgg')

# RECALL: traintest variable: 0 is for train, 2 is for train with random segments, 1 for validation and 3 for test


# FUNCIÓ PREPROCESSING

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

        ax3.set_xlabel("Time (sec)", fontsize=14)  # common axis label
        ax3.legend(loc='best')

        plt.show()
        plt.clf()

    return signal_pure


# BAND PASS FILTER

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# SOLVE BATCH PROBLEM WHEN THE BATCH SIZE IS NOT MULTIPLE OF NUMBER OF SAMPLES

def batchprob(ppglist, hrlist, batchsize, traintest):
    seqsize = len(ppglist[0])
    rest = len(ppglist) % batchsize
    if rest != 0:
        if traintest in (0, 2):
            ppglist = ppglist[:len(ppglist)-rest]
            hrlist = hrlist[:len(hrlist)-rest]
        else:
            sobrant = batchsize - rest
            for i in range(sobrant):
                a = np.zeros(seqsize).tolist()
                ppglist.append(np.asarray(a))
                hrlist.append(np.array([0]))
    return ppglist, hrlist


# LINEAR INTERPOLATION

def linearinterpolaion(t0, t1, t, y0, y1):
    return (y0*(t1-t) + y1*(t-t0))/(t1-t0)


# SEGMENTATION AND DATA AUGMENTATION

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
                    for k in range(int(increment1/increment2)):
                        d = listHR[i][j].item()
                        e = listHR[i][j+1].item()
                        hr = np.array([linearinterpolaion(idx1, idx1+increment1, idx1+increment2*val, d, e)])
                        ppg = listPPG[i][idx1+increment2*val:idx2+increment2*val]
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


# DATALOADER

class DATALOADER(torch.utils.data.Dataset):
    # Initialization method for the dataset
    def __init__(self, samplingperiodPPG_, samplingfreqPPG_, window, segsize, repsize, traintest, PATH_train,
                 times, batchsize, person, aug):

        # read data
        lecturaPPG = []
        lecturaHR = []

        hr1 = 'BPM'

        os.chdir(PATH_train)

        for file in os.listdir():
            file_path = f"{PATH_train}/{file}"
            filename = os.path.basename(file_path)
            for item in person:
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

        # band pass filter
        for i in range(len(lecturaPPG)):
            lecturaPPG[i] = butter_bandpass_filter(lecturaPPG[i], 1, 3.4, samplingfreqPPG_, 2)

        # preprocessing
        '''
        for i in range(len(lecturaPPG)):
            view_ = False
            if i == len(lecturaPPG) - 1:
                view_ = True
            lecturaPPG[i] = ppg_preprocessing(lecturaPPG[i], samplingfreqPPG_, view=view_)
        '''

        # reestructure
        listPPG, listHR = sampling(lecturaPPG, lecturaHR, aug)

        # save data and solve batch problem
        listPPG, listHR = batchprob(listPPG, listHR, batchsize, traintest)
        self.data = np.asarray(listPPG)
        self.data = torch.from_numpy(self.data).type(torch.float32)
        self.labels = np.asarray(listHR)
        self.labels = torch.from_numpy(self.labels).type(torch.float32)

    # What to do to load a single item in the dataset (read PPG, HR)
    def __getitem__(self, index):
        data = self.data[index]
        lbl = self.labels[index]

        # return the PPG, HR(label) and name of the experiment
        return data, lbl

    # Return the number of PPGs
    def __len__(self):
        return self.data.shape[0]
