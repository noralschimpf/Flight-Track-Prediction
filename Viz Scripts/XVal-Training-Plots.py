import os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

import numpy as np

eps = 500
os.chdir('../Models/ECHO_TOP/')
mdldirs = [x for x in os.listdir() if 'EPOCHS{}'.format(eps) in x and os.path.isdir(x)]
for dir in mdldirs:
    if not dir == 'CONV1.0.0-LSTM1lay-OPTAdam-LOSSMSELoss()-EPOCHS3-BATCH5-RNN6_100_3': continue
    os.chdir(dir)
    shortdir = dir.split('-')[:-2]; shortdir[0] = shortdir[0][:-4]; shortdir[2]=shortdir[2][3:]
    shortdir[3] = shortdir[3][4:]; shortdir = '-'.join(shortdir)

    folddirs = [x for x in os.listdir() if 'fold' in x and os.path.isdir(x)]
    fig, ax = plt.subplots(1,1)
    nda_losses, nda_vallosses = np.zeros((len(folddirs),eps)), np.zeros((len(folddirs),eps))
    for f in range(len(folddirs)):
        df = pd.read_csv('{}/model_epoch_losses.txt'.format(folddirs[f]))
        nda_losses[f] = df['loss']; nda_vallosses[f] = df['valloss']
    #     ax.plot(df['loss'].values, color='blue', alpha=0.1)
    #     ax.plot(df['valloss'].values, color='orange', alpha=0.1)
    # ax.plot(nda_losses.mean(0), color='blue', label='Training Loss')
    # ax.plot(nda_vallosses.mean(0), color='orange', label='Validation Loss')
    # ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (MSE)'); ax.legend()
    # ax.set_title('Cross-Validated Training for {}'.format(dir))
    # fig.savefig('XVAL-Trainloss.png', dpi=300)
    # ax.set_ylim([0,.01]);fig.savefig('LIM XVAL-Trainloss.png', dpi=300)
    # plt.close(); fig.clf(); ax.cla()

    df = pd.DataFrame()
    df['dataset'] = np.repeat(['train','validation'], len(folddirs)*eps)
    df['loss'] = np.hstack((nda_losses.reshape(-1), nda_vallosses.reshape(-1)))
    df['epochs'] = np.tile(np.arange(1,eps+1), len(folddirs)*2)
    sns.lineplot(data=df, x='epochs', y='loss', hue='dataset')
    plt.title("{} {}-fold Cross-Validation".format(shortdir, len(folddirs)))
    plt.savefig("XVAL-Training.png", dpi=300); plt.close()

    os.chdir('..')