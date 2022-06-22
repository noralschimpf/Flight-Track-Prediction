import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

EXP = 'ATTN-CONSTINIT'
EXPDIR = '/home/dualboot/Desktop/Tuning-Results-Analysis/{}'.format(EXP)
os.chdir(EXPDIR)

opts = ['adam_','rmsprop_',]
colors=['blue','red','teal','magenta','green','gold','red','orange','fuchsia']
keys = ['loss', 'valloss']
for key in keys:
    fig, ax = plt.subplots(1,1)
    for i in range(len(opts)):
      files = [x for x in os.listdir() if opts[i] in x]
      nda_loss = np.zeros((len(files),300))
      for f in range(len(files)):
        df = pd.read_csv(os.path.join(files[f],'progress.csv'))
        ax.plot(df['valloss'].values,color='gray',alpha=0.1)
        nda_loss[f] = df['valloss'].values
      ax.plot(nda_loss.mean(0), color=colors[i], alpha=1, label=opts[i][:-1])
    ax.set_title('SAR_GRU-OPTFULL'); ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('{trainval} Loss (MSE)'.format(trainval='Training' if key == 'loss' else 'Validation'))
    ylim=.1; ax.set_ylim([0,ylim])
    plt.savefig('{trainval}-{l}-{exp}.png'.format(trainval='TRAIN' if key == 'loss' else 'VAL', l=ylim, exp=EXP), dpi=300)
    #plt.show(block=True)

