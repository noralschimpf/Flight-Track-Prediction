import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

os.chdir('/home/dualboot/Desktop/Tuning-Results-Analysis/SAR_GRU-OPTFULL-SCHED')

steps = ['forcestep=10', 'forcestep=30', 'forcestep=50']
gammas = ['forcegamma=0.1', 'forcegamma=0.5', 'forcegamma=0.9']
opts = ['optim=sgd_', 'optim=adam_', 'optim=rmsprop_']
colors = ['magenta', 'green', 'gold']

opt_fig, opt_ax = plt.subplots(1, 1)
opt_valfig, opt_valax = plt.subplots(1, 1)

for o in range(len(opts)):
    fig, ax = plt.subplots(1, 1)
    df_opt = pd.DataFrame()
    for g in range(len(gammas)):
        for s in range(len(steps)):
            files = [x for x in os.listdir() if steps[s] in x and gammas[g] in x and opts[o] in x]
            gma, stp = float(gammas[g][11:]), float(steps[s][10:])
            opt = opts[o][6:-1]
            for file in files:
                df = pd.read_csv('{}/progress.csv'.format(file))
                df_opt = df_opt.append(
                    {'stepsize': stp, 'decay': gma, 'loss': df['loss'].values[-1], 'valloss': df['valloss'].values[-1]},
                    ignore_index=True)

    df_opt.to_csv('CNN_GRU LR Scheduling-{}.csv'.format(opts[o]))
    grp = df_opt.groupby(['stepsize', 'decay'])
    df_scat = pd.DataFrame(
        {'stepsize': [x[0] for x in list(grp.groups.keys())], 'decay': [x[1] for x in list(grp.groups.keys())],
         'loss': grp.mean()['loss'].values, 'valloss': grp.mean()['valloss'].values})

    lim = .005
    key = df_scat['loss'] <= lim
    valkey = df_scat['valloss'] <= lim

    trainloss = ax.scatter(df_scat['stepsize'][key], df_scat['decay'][key], c=df_scat['loss'][key], cmap='coolwarm')
    train_cbar = plt.colorbar(trainloss);
    train_cbar.set_label('Training Loss (MSE)')
    ax.set_title('LR Scheduling for {}'.format(opt))
    ax.set_xlabel('Step Size');
    ax.set_ylabel('LR Decay')
    fig.savefig('TRAIN-LIM{} CNN_GRU LR Scheduling-{}.png'.format(lim, opts[o]), dpi=300)
    plt.close();
    fig.clf();
    ax.cla()

    valfig, valax = plt.subplots(1, 1)
    valloss = valax.scatter(df_scat['stepsize'][valkey], df_scat['decay'][valkey], c=df_scat['valloss'][valkey],
                            cmap='coolwarm')
    val_cbar = plt.colorbar(valloss);
    val_cbar.set_label('Validation Loss (MSE)')
    valax.set_title('LR Scheduling for {}'.format(opt))
    valax.set_xlabel('Step Size');
    valax.set_ylabel('LR Decay')
    valfig.tight_layout();
    valfig.savefig('VAL-LIM{} CNN_GRU LR Scheduling-{}.png'.format(lim, opts[o]), dpi=300)
    plt.close();
    valfig.clf();
    valax.cla()

    row_train = df_scat[df_scat['loss'] == df_scat.min()['loss']]
    besttrain = 'forcegamma={},forcestep={},{}'.format(row_train['decay'].values[0],
                                                       int(row_train['stepsize'].values[0]), opts[o])
    # trainfiles = [x for x in os.listdir() if besttrain in x]
    # nda_trains, nda_vals = np.zeros((len(trainfiles), 301)), np.zeros((len(trainfiles), 301))
    # for f in range(len(trainfiles)):
    #     df = pd.read_csv('{}/progress.csv'.format(trainfiles[f]))
    #     nda_trains[f] = df['loss'].values
    #     opt_ax.plot(df['loss'].values, color='gray', alpha=0.1)
    # opt_ax.plot(nda_trains.mean(0), color=colors[o], label=besttrain.replace('force', '')[:-1])

    row_val = df_scat[df_scat['valloss'] == df_scat.min()['valloss']]
    bestval = 'forcegamma={},forcestep={},{}'.format(row_val['decay'].values[0], int(row_val['stepsize'].values[0]),
                                                     opts[o])
    valfiles = [x for x in os.listdir() if bestval in x]
    nda_trains, nda_vals = np.zeros((len(valfiles), 301)), np.zeros((len(valfiles), 301))
    for f in range(len(valfiles)):
        df = pd.read_csv('{}/progress.csv'.format(valfiles[f]))
        nda_trains[f] = df['loss'].values; opt_ax.plot(df['loss'].values, color='gray', alpha=0.1)
        nda_vals[f] = df['valloss'].values; opt_valax.plot(df['valloss'].values, color='gray', alpha=0.1)
    opt_ax.plot(nda_trains.mean(0), color=colors[o], label=bestval.replace('force','')[:-1])
    opt_valax.plot(nda_vals.mean(0), color=colors[o], label=bestval.replace('force', '')[:-1])

opt_ax.set_xlabel('Epoch'); opt_ax.set_ylabel('Training Loss (MSE)'); opt_ax.set_ylim([0,0.01])
opt_ax.set_title('CNN_GRU OPT Scheduling Comparison'); opt_ax.legend()
opt_fig.savefig('CNN_GRU LR Scheduling TrainLosses.png', dpi=300)
opt_fig.clf(); opt_ax.cla()

opt_valax.set_xlabel('Epoch'); opt_valax.set_ylabel('Validation Loss (MSE)'); opt_valax.set_ylim([0,0.01])
opt_valax.set_title('CNN_GRU OPT Scheduling Comparison'); opt_valax.legend()
opt_valfig.savefig('CNN_GRU LR Scheduling ValLosses.png', dpi=300)
plt.close(); opt_valfig.clf(); opt_valax.cla()