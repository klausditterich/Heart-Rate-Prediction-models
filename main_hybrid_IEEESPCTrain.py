import traintest
import torch
import random


if __name__ == '__main__':
    # define some variables

    samplingfreqPPG = 125

    samplingperiodPPG = 1 / samplingfreqPPG

    segmentsize = 1000
    repetition = 10
    wind = 20
    times = 25

    PATH_train = 'C:/Users/Usuario/Downloads/SPC_2015_dataset/Training_data'

    input_size = 1
    num_epochs = 200
    hidden_size = 1000
    batchsize = 20

    persons = ['DATA_01', 'DATA_02', 'DATA_03', 'DATA_04', 'DATA_05',
               'DATA_06', 'DATA_07', 'DATA_08', 'DATA_09', 'DATA_10',
               'DATA_11', 'DATA_12']

    random.seed(20)
    random.shuffle(persons)

    listpers = []

    for i in range(4):
        a = []
        for j in range(3):
            a.append(persons[(i*3)+j])
        listpers.append(a)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    traintest.traintest_model(input_size, hidden_size, num_epochs, PATH_train,
                              samplingperiodPPG, samplingfreqPPG, wind, segmentsize,
                              repetition, device, times, batchsize, listpers)
