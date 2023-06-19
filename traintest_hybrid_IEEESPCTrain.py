import torch
import functions
import models
import os
from datetime import datetime
import matplotlib.pyplot as plt
from numpy import savetxt
from numpy import std
import torch.nn as nn

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def traintest_model(input_size_, hidden_size_, num_epochs_, PATH_train,
                    samplingperiodPPG, samplingfreqPPG, wind, segmentsize, repetition, device, times, batchsize,
                    persons):
    report = []
    losses = []
    for i in range(4):
        listvaltest = persons[i]
        listtrain = [x for x in persons if x != listvaltest]
        listtrain = [item for sublist in listtrain for item in sublist]
        for j in range(3):
            test = []
            test.append(listvaltest[j])
            val = [x for x in listvaltest if x not in test]
            print('Val:', val)
            print('Test:', test)

            if (i == 0) & (j == 0):
                now = datetime.now()
                dt_string = now.strftime('%d-%m-%Y_%H%M')

                folderpath = 'C:/Users/Usuario/Downloads/' + dt_string
                if not os.path.exists(folderpath):
                    os.makedirs(folderpath)

            hr_est_F = models.Fourier(input_size_, hidden_size_, device=device).to(device)

            hr_est_T = models.CORNET2(input_size_, hidden_size_).to(device)

            criterion_F = nn.CrossEntropyLoss()

            criterion_T = nn.L1Loss()

            optimizer_F = torch.optim.Adam(list(hr_est_F.parameters()), lr=0.0008)

            optimizer_T = torch.optim.Adam(list(hr_est_T.parameters()), lr=0.00009)

            # Load Data Train and Test

            DATALOADER_Train = functions.DATALOADER(samplingperiodPPG, samplingfreqPPG, wind, segmentsize, repetition,
                                                    2, PATH_train, times, batchsize,
                                                    listtrain, True)
            train_loader = torch.utils.data.DataLoader(dataset=DATALOADER_Train,  # indicate the used dataset
                                                       batch_size=batchsize,
                                                       # Number of samples that will be loaded for iteration
                                                       shuffle=True)

            DATALOADER_val = functions.DATALOADER(samplingperiodPPG, samplingfreqPPG, wind, segmentsize, repetition, 1,
                                                  PATH_train, times, batchsize,
                                                  val, False)
            val_loader = torch.utils.data.DataLoader(dataset=DATALOADER_val,  # indicate the used dataset
                                                     batch_size=batchsize,
                                                     # Number of samples that will be loaded for iteration
                                                     shuffle=False)

            DATALOADER_test = functions.DATALOADER(samplingperiodPPG, samplingfreqPPG, wind, segmentsize, repetition, 3,
                                                   PATH_train, times, batchsize,
                                                   test, False)
            test_loader = torch.utils.data.DataLoader(dataset=DATALOADER_test,  # indicate the used dataset
                                                      batch_size=batchsize,
                                                      # Number of samples that will be loaded for iteration
                                                      shuffle=False)

            t_train = []
            t_val = []
            t_test = []
            f_train = []
            f_val = []
            f_test = []
            f_train_mae = []
            f_val_mae = []
            f_test_mae = []
            h_test = []

            minimumh = 1000
            minimumt = 1000
            minimumf = 1000

            # iterate over epochs
            for epoch in range(num_epochs_):

                hr_est_F.train()
                hr_est_T.train()

                loss_epoch_train_f = 0
                loss_epoch_train_mae_f = 0
                loss_epoch_val_f = 0
                loss_epoch_val_mae_f = 0
                loss_epoch_test_f = 0
                loss_epoch_test_mae_f = 0
                loss_epoch_train_t = 0
                loss_epoch_val_t = 0
                loss_epoch_test_t = 0
                loss_epoch_test_h = 0

                for k, (inputs, labels) in enumerate(train_loader):

                    inputs_t = inputs.unsqueeze(-1).to(device)
                    labels_t = labels.unsqueeze(-1).to(device) - 40

                    inputs_f = inputs.to(device)
                    labels_f = (labels.to(device) - 40)
                    labels_f = torch.round(labels_f)
                    labels_f = torch.squeeze(labels_f.to(torch.long))

                    outputs_t = hr_est_T(inputs_t)

                    outputs_f = hr_est_F(inputs_f)

                    idx_f = torch.argmax(outputs_f, dim=1)
                    hr_f = (idx_f + 40).detach().cpu().numpy()

                    optimizer_T.zero_grad()
                    optimizer_F.zero_grad()

                    labels_t = labels_t.view(20, 1)

                    loss_t = criterion_T(outputs_t, labels_t)

                    loss_f = criterion_F(outputs_f, labels_f)

                    loss_epoch_train_t += loss_t

                    mae = 0
                    for h in range(len(hr_f)):
                        mae += torch.abs(labels[h][0] - hr_f[h])

                    mae = mae/len(hr_f)

                    loss_epoch_train_f += loss_f
                    loss_epoch_train_mae_f += mae

                    loss_t.backward()
                    loss_f.backward()

                    optimizer_T.step()
                    optimizer_F.step()

                    print('loss temp: ', loss_t.item(), ' Iteració:', k + 1, '/', len(train_loader), 'Epoch:',
                          epoch + 1)
                    print('loss CE freq: ', loss_f.item(), ' Iteració:', k + 1, '/', len(train_loader), 'Epoch:',
                          epoch + 1)
                    print('MAE freq: ', mae.item(), ' Iteració:', k + 1, '/', len(train_loader), 'Epoch:', epoch + 1)

                hr_est_T.eval()
                hr_est_F.eval()

                with torch.no_grad():
                    for g, (inputs, labels) in enumerate(val_loader):
                        inputs_t = inputs.unsqueeze(-1).to(device)
                        labels_t = labels.unsqueeze(-1).to(device) - 40

                        inputs_f = inputs.to(device)
                        labels_f = (labels.to(device) - 40)
                        labels_f = torch.round(labels_f)
                        labels_f = torch.squeeze(labels_f.to(torch.long))

                        outputs_t = hr_est_T(inputs_t)

                        outputs_f = hr_est_F(inputs_f)

                        if g == len(val_loader) - 1:
                            idx = 0
                            for f in range(len(labels)):
                                if torch.mean(labels[f]) != 0:
                                    idx += 1
                            if idx == 0:
                                idx = len(labels)
                            labels = labels[:idx]
                            labels_t = labels_t[:idx]
                            labels_f = labels_f[:idx]
                            outputs_t = outputs_t[:idx]
                            outputs_f = outputs_f[:idx]

                        idx_f = torch.argmax(outputs_f, dim=1)
                        hr_f = (idx_f + 40).detach().cpu().numpy()

                        dim = labels_t.shape[0]
                        labels_t = labels_t.view(dim, 1)

                        loss_t = criterion_T(outputs_t, labels_t)

                        loss_f = criterion_F(outputs_f, labels_f)

                        loss_epoch_val_t += loss_t

                        mae = 0
                        for r in range(len(hr_f)):
                            mae += torch.abs(labels[r][0] - hr_f[r])

                        mae = mae/len(hr_f)

                        loss_epoch_val_f += loss_f
                        loss_epoch_val_mae_f += mae

                        print('loss temp: ', loss_t.item(), ' Iteració:', g + 1, '/', len(val_loader), 'Epoch:',
                              epoch + 1)
                        print('loss CE freq: ', loss_f.item(), ' Iteració:', g + 1, '/', len(val_loader), 'Epoch:',
                              epoch + 1)
                        print('MAE freq: ', mae.item(), ' Iteració:', g + 1, '/', len(val_loader), 'Epoch:', epoch + 1)

                    for d, (inputs, labels) in enumerate(test_loader):

                        inputs_t = inputs.unsqueeze(-1).to(device)
                        labels_t = labels.unsqueeze(-1).to(device) - 40

                        inputs_f = inputs.to(device)
                        labels_f = (labels.to(device) - 40)
                        labels_f = torch.round(labels_f)
                        labels_f = torch.squeeze(labels_f.to(torch.long))

                        outputs_t = hr_est_T(inputs_t)

                        outputs_f = hr_est_F(inputs_f)

                        if d == len(test_loader) - 1:
                            idx = 0
                            for f in range(len(labels)):
                                if torch.mean(labels[f]) != 0:
                                    idx += 1
                            if idx == 0:
                                idx = len(labels)
                            labels = labels[:idx]
                            labels_t = labels_t[:idx]
                            labels_f = labels_f[:idx]
                            outputs_t = outputs_t[:idx]
                            outputs_f = outputs_f[:idx]

                        idx_f = torch.argmax(outputs_f, dim=1)
                        hr_f = (idx_f + 40).detach().cpu().numpy()

                        hr_h = torch.empty_like(outputs_t)

                        hr_h[0] = outputs_t[0] + 40

                        for z in range(1, len(outputs_t)):
                            if abs((outputs_t[z] + 40) - hr_h[z-1]) <= abs(hr_f[z] - hr_h[z-1]):
                                hr_h[z] = outputs_t[z] + 40
                            else:
                                hr_h[z] = hr_f[z]

                        hr_h = hr_h.squeeze()

                        mae_h = 0
                        for t in range(len(hr_h)):
                            mae_h += torch.abs(labels[t][0] - hr_h[t])

                        mae_h = mae_h / len(hr_h)

                        loss_epoch_test_h += mae_h

                        dim = labels_t.shape[0]
                        labels_t = labels_t.view(dim, 1)

                        loss_t = criterion_T(outputs_t, labels_t)

                        loss_f = criterion_F(outputs_f, labels_f)

                        loss_epoch_test_t += loss_t

                        mae = 0
                        for s in range(len(hr_f)):
                            mae += torch.abs(labels[s][0] - hr_f[s])

                        mae = mae / len(hr_f)

                        loss_epoch_test_f += loss_f
                        loss_epoch_test_mae_f += mae

                        print('loss temp: ', loss_t.item(), ' Iteració:', d + 1, '/', len(test_loader), 'Epoch:',
                              epoch + 1)
                        print('loss CE freq: ', loss_f.item(), ' Iteració:', d + 1, '/', len(test_loader), 'Epoch:',
                              epoch + 1)
                        print('MAE freq: ', mae.item(), ' Iteració:', d + 1, '/', len(test_loader), 'Epoch:', epoch + 1)
                        print('MAE hybrid: ', mae_h.item(), ' Iteració:', d + 1, '/', len(test_loader), 'Epoch:',
                              epoch + 1)

                loss_epoch_test_f = (loss_epoch_test_f / len(test_loader)).detach().cpu().numpy().item()
                f_test.append(loss_epoch_test_f)

                loss_epoch_test_mae_f = (loss_epoch_test_mae_f / len(test_loader))
                f_test_mae.append(loss_epoch_test_mae_f.item())

                loss_epoch_test_t = (loss_epoch_test_t / len(test_loader)).detach().cpu().numpy().item()
                t_test.append(loss_epoch_test_t)

                loss_epoch_test_h = (loss_epoch_test_h / len(test_loader)).detach().cpu().numpy().item()
                h_test.append(loss_epoch_test_h)


                if loss_epoch_test_h <= minimumh:
                    minimumh = loss_epoch_test_h
                    # torch.save(hr_est_F.fc3.state_dict(), folderpath + '/fc3_weights' + test[0] + str(epoch))

                if loss_epoch_test_mae_f < minimumf:
                    minimumf = loss_epoch_test_mae_f

                if loss_epoch_test_t < minimumt:
                    minimumt = loss_epoch_test_t

                loss_epoch_train_f = (loss_epoch_train_f / len(train_loader)).detach().cpu().numpy().item()
                f_train.append(loss_epoch_train_f)

                loss_epoch_train_mae_f = (loss_epoch_train_mae_f / len(train_loader))
                f_train_mae.append(loss_epoch_train_mae_f.item())

                loss_epoch_train_t = (loss_epoch_train_t / len(train_loader)).detach().cpu().numpy().item()
                t_train.append(loss_epoch_train_t)

                loss_epoch_val_f = (loss_epoch_val_f / len(val_loader)).detach().cpu().numpy().item()
                f_val.append(loss_epoch_val_f)

                loss_epoch_val_mae_f = (loss_epoch_val_mae_f / len(val_loader))
                f_val_mae.append(loss_epoch_val_mae_f.item())

                loss_epoch_val_t = (loss_epoch_val_t / len(val_loader)).detach().cpu().numpy().item()
                t_val.append(loss_epoch_val_t)

                # Print losses
                print('Frequencial')
                print(f_train_mae, f_val_mae, f_test_mae)
                print(f_train, f_val, f_test)
                print('Temporal')
                print(t_train, t_val, t_test)
                print('Hybrid')
                print(h_test)

            print('The minimum temp mae is: ', minimumt)
            print('The minimum freq mae is: ', minimumf.item())
            print('The minimum hybrid mae is: ', minimumh)

            losses.append((['Test:', test[0], 'Val:', val[0], val[1], 'Loss temp:', minimumt, 'Loss freq:',
                            minimumf.item(), 'Loss hybrid:', minimumh]))

            plt.plot(f_train_mae)
            plt.plot(f_val_mae)
            plt.plot(f_test_mae)
            plt.legend(labels=['freq mae train', 'freq mae val', 'freq mae test'])
            string = '/' + test[0] + '_plot_fmaelosses.png'
            plt.savefig(folderpath + string)
            plt.clf()

            plt.plot(f_train)
            plt.plot(f_val)
            plt.plot(f_test)
            plt.legend(labels=['CE train', 'CE val', 'CE test'])
            string = '/' + test[0] + '_plot_CElosses.png'
            plt.savefig(folderpath + string)
            plt.clf()

            plt.plot(t_train)
            plt.plot(t_val)
            plt.plot(t_test)
            plt.legend(labels=['temp mae train', 'temp mae val', 'temp mae test'])
            string = '/' + test[0] + '_plot_tmaelosses.png'
            plt.savefig(folderpath + string)
            plt.clf()

            plt.plot(h_test)
            plt.legend(labels=['hybrid mae test'])
            string = '/' + test[0] + '_plot_hmaelosses.png'
            plt.savefig(folderpath + string)
            plt.clf()

    listlosses = [item[6] for item in losses]
    mean1t = 0
    std1t = 0
    for i in range(4):
        lossfold = []
        for j in range(3):
            lossfold.append(listlosses[3 * i + j])
        mean1t += sum(lossfold) / len(lossfold)
        std1t += std(lossfold)
    mean1t = mean1t / 4
    std1t = std1t / 4
    mean2t = sum(listlosses) / len(listlosses)
    std_t = std(listlosses)

    listlosses = [item[8] for item in losses]
    mean1f = 0
    std1f = 0
    for i in range(4):
        lossfold = []
        for j in range(3):
            lossfold.append(listlosses[3 * i + j])
        mean1f += sum(lossfold) / len(lossfold)
        std1f += std(lossfold)
    mean1f = mean1f / 4
    std1f = std1f / 4
    mean2f = sum(listlosses) / len(listlosses)
    std_f = std(listlosses)

    listlosses = [item[10] for item in losses]
    mean1h = 0
    std1h = 0
    for i in range(4):
        lossfold = []
        for j in range(3):
            lossfold.append(listlosses[3 * i + j])
        mean1h += sum(lossfold) / len(lossfold)
        std1h += std(lossfold)
    mean1h = mean1h / 4
    std1h = std1h / 4
    mean2h = sum(listlosses) / len(listlosses)
    std_h = std(listlosses)

    report.append(['Mean through subjects temp:', mean2t, std_t, 'Mean through folds temp:', mean1t, std1t,
                   'Mean through subjects freq:', mean2f, std_f, 'Mean through folds freq:', mean1f, std1f,
                   'Mean through subjects hybrid:', mean2h, std_h, 'Mean through folds hybrid:', mean1h, std1h])

    savetxt(folderpath + '/lossperperson.csv', losses, delimiter=',', comments='', fmt='%s')
    savetxt(folderpath + '/mean.csv', report, delimiter=',', comments='', fmt='%s')
