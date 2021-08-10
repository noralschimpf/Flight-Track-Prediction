import os, pandas as pd, matplotlib.pyplot as plt

import numpy as np

eps = 300
os.chdir('../Models/ECHO_TOP/')
mdldirs = [x for x in os.listdir() if 'EPOCHS{}'.format(eps) in x and os.path.isdir(x)]
for dir in mdldirs:
    os.chdir(dir)
    folddirs = [x for x in os.listdir() if 'fold' in x and os.path.isdir(x)]
    fig, ax = plt.subplots(1,1)
    nda_losses, nda_vallosses = np.zeros((len(folddirs),eps)), np.zeros((len(folddirs),eps))
    for f in range(len(folddirs)):
        df = pd.read_csv('{}/model_epoch_losses.txt'.format(folddirs[f]))
        nda_losses[f] = df['loss']; nda_vallosses[f] = df['valloss']
        ax.plot(df['loss'].values, color='blue', alpha=0.1)
        ax.plot(df['valloss'].values, color='orange', alpha=0.1)
    ax.plot(nda_losses.mean(0), color='blue', label='Training Loss')
    ax.plot(nda_vallosses.mean(0), color='orange', label='Validation Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (MSE)'); ax.legend()
    ax.set_title('Cross-Validated Training for {}'.format(dir))
    fig.savefig('XVAL-Trainloss.png', dpi=300)
    ax.set_ylim([0,.01]);fig.savefig('LIM XVAL-Trainloss.png', dpi=300)
    plt.close(); fig.clf(); ax.cla(); os.chdir('..')