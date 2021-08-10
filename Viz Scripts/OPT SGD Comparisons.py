import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

EXPDIR = '/home/dualboot/Desktop/Results-Analysis/SAR_GRU-SGD'
os.chdir(EXPDIR)
df_mom = pd.DataFrame()
dirs = [x for x in os.listdir() if os.path.isdir(x)]
for dir in dirs:
    if 'progress.csv' in os.listdir(dir):
        df = pd.read_csv(os.path.join(dir,'progress.csv'))
        loss, valloss, = df['loss'].values[-1], df['valloss'].values[-1]
        spltdir = dir.split(',')
        lr, mom, nest= float(spltdir[0].split('=')[-1]), float(spltdir[1].split('=')[-1]), spltdir[2].split('=')[-1] == 'True'
        df_mom = df_mom.append({'lr': lr, 'momentum': mom, 'nesterov': nest, 'loss': loss, 'valloss': valloss}, ignore_index=True)

lim = .05
print('{}      {}'.format(df_mom['loss'].min(), df_mom['valloss'].min()))
for n in [True, False]:
    fig, ax = plt.subplots(1,1)
    key = (df_mom['loss'] <= lim) & (df_mom['nesterov'] == n)
    lr = np.array(df_mom['lr'].values)
    mom = np.array(df_mom['momentum'].values)
    trainloss = ax.scatter(lr[key],mom[key],c=df_mom['loss'][key],cmap='coolwarm')
    cbar = plt.colorbar(trainloss)
    cbar.set_label('Training Loss (MSE)')
    ax.set_xlabel('Learning Rate'); ax.set_ylabel('Momentum Weight')
    ax.set_title('CNN_GRU-SGD Training Loss {nest}'.format(nest='W/ Nesterov Momentum' if n else ''))
    fig.tight_layout()
    plt.show(block=False)
    ax.set_xlim([1e-7,10]); ax.set_xscale('log')
    ax.set_ylim([-.05,1.05])
    fig.savefig('LIM{l}-TRAIN_2D-CNN_GRU-SGD{nest}.png'.format(l=lim, nest='NEST' if n else ''),dpi=300)
    plt.close();fig.clf();ax.cla(); del trainloss; del cbar;
    valfig, valax = plt.subplots(1,1)
    valkey = (df_mom['loss'] <= lim) & (df_mom['nesterov'] == n)
    valloss = valax.scatter(lr[valkey],mom[valkey],c=df_mom['valloss'][valkey],cmap='coolwarm')
    cbar2 = plt.colorbar(valloss)
    cbar2.set_label('Validation Loss (MSE)')
    valax.set_xlabel('Learning Rate'); valax.set_ylabel('Momentum')
    valax.set_title('CNN_GRU-SGD Validation Loss {nest}'.format(nest='W/ Nesterov Momentum' if n else ''))
    valfig.tight_layout()
    plt.show(block=False)
    valax.set_xlim([1e-7,10]); valax.set_xscale('log')
    valax.set_ylim([-.05,1.05])
    valfig.savefig('LIM{l}-VAL_2D-CNN_GRU{nest}.png'.format(l=lim, nest='NEST' if n else ''),dpi=300)
    plt.close();valfig.clf();valax.cla(); del valloss; del cbar2
    df_mom.to_csv('SAR_GRU-SGD.csv')
