import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, shutil, gc, tqdm, json, random
from custom_dataset import pad_batch
from libmodels.CONV_RECURRENT import CONV_RECURRENT
from libmodels.model import load_model, init_constant
from ray import tune

def fit(config: dict, train_dataset: torch.utils.data.DataLoader, test_dataset: torch.utils.data.Dataset):

    if config["determinist"]:
	    # FORCE DETERMINISTIC INITIALIZATION
	    seed = 1234
	    random.seed(seed)
	    os.environ['PYTHONHASHSEED']=str(seed)
	    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
	    np.random.seed(seed)
	    torch.manual_seed(seed)
	    torch.cuda.manual_seed(seed)
	    torch.cuda.manual_seed_all(seed)
	    #torch.backends.cudnn.deterministic = True
	    torch.backends.cudnn.benchmark = False

	    #torch.backends.cudnn.enabled = False
	    torch.use_deterministic_algorithms(True)

    train_dl = torch.utils.data.DataLoader(train_dataset, collate_fn=pad_batch, batch_size=config['batch_size'], num_workers=8, pin_memory=True,
                          shuffle=False, drop_last=True)
    test_dl = torch.utils.data.DataLoader(test_dataset, collate_fn=pad_batch, batch_size=1, num_workers=8, pin_memory=True,
                         shuffle=False, drop_last=True)

    mdl = CONV_RECURRENT(paradigm=config['paradigm'], cube_height=config['cube_height'], device=config['device'],
                         rnn=config['rnn'], features=config['features'], rnn_layers=config['RNNDepth'],
                         attn=config['attn'], batch_size=config['batch_size'],
                         conv_input=config['ConvCh'][0], conv_hidden=config['ConvCh'][1],
                         conv_output=config['ConvCh'][2], batchnorm=config['batchnorm'],
                         dense_hidden=config['HLs'], rnn_input=config['RNNIn'], rnn_hidden=config['RNNHidden'],
                         rnn_output=3, droprate=config['droprate'])
    if config["const"]: mdl.apply(init_constant)
    if config['optim'] == 'sgd':
        #lr, mom, dec, nest= 0.001, 0.0, config['weight_reg'], False
        lr, mom, dec, nest = 0.01, 0.7, config['weight_reg'], True
        if 'forcelr' in config: lr = config['forcelr']
        if 'forcemom' in config: mom = config['forcemom']
        mdl.optimizer = torch.optim.SGD(mdl.parameters(), lr=lr, weight_decay=dec, momentum=mom, nesterov=nest)
    elif config['optim'] == 'sgd+momentum':
        lr, mom, dec, nest = 0.001, 0.0, config['weight_reg'], False
        if 'forcelr' in config: lr = config['forcelr']
        if 'forcemom' in config: mom = config['forcemom']
        mdl.optimizer = torch.optim.SGD(mdl.parameters(), lr=1e-5, momentum=0.5, nesterov=False, weight_decay=dec)
    elif config['optim'] == 'sgd+nesterov':
        lr, mom, dec, nest = 0.001, 0.0, config['weight_reg'], False
        if 'forcelr' in config: lr = config['forcelr']
        if 'forcemom' in config: mom = config['forcemom']
        mdl.optimizer = torch.optim.SGD(mdl.parameters(), lr=lr, momentum=mom, nesterov=True, weight_decay=dec)
    elif config['optim'] == 'adam':
        lr=0.001
        if 'forcelr' in config: lr = config['forcelr']
        mdl.optimizer = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=config['weight_reg'])
    elif config['optim'] == 'rmsprop':
        lr = 0.001
        if 'forcelr' in config: lr = config['forcelr']
        mdl.optimizer = torch.optim.RMSprop(mdl.parameters(), lr=lr, weight_decay=config['weight_reg'])
    elif config['optim'] == 'adadelta':
        lr = 1.
        if 'forcelr' in config: lr = config['forcelr']
        mdl.optimizer = torch.optim.Adadelta(mdl.parameters(), lr=lr, weight_decay=config['weight_reg'])
    elif config['optim'] == 'adagrad':
        lr = 0.001
        if 'forcelr' in config: lr = config['forcelr']
        mdl.optimizer = torch.optim.Adagrad(mdl.parameters(), lr=lr, weight_decay=config['weight_reg'])
    if 'forcestep' in config and 'forcegamma' in config:
        mdl.sched = torch.optim.lr_scheduler.StepLR(mdl.optimizer, step_size=config['forcestep'],gamma=config['forcegamma'])
    mdl.update_dict()

    if config['checkpoint_dir'] != "None":
        chkpt = os.path.join(config["checkpoint_dir"], 'checkpoint')
        with open(chkpt) as f:
            state = json.loads(f.read())
            start = state['step'] + 1
        mdl = load_model(chkpt)

    epoch_losses = torch.zeros(config['epochs'], device=mdl.device)
    epoch_test_losses = torch.zeros(config['epochs'], device=mdl.device)
    for ep in (tqdm.trange(config['epochs'], desc='epoch', position=0, leave=False) if not config['raytune'] else range(config['epochs'])):
        losses = torch.zeros(len(train_dl), device=mdl.device)

        for batch_idx, (fp, ft, wc) in enumerate((tqdm.tqdm(train_dl, desc='flight', position=1, leave=False) if not config['raytune'] else train_dl)):  # was len(flight_data)
            # Extract flight plan, flight track, and weather cubes
            if mdl.device.__contains__('cuda'):
                fp = fp[:, :, :].cuda(device=mdl.device, non_blocking=True)
                ft = ft[:, :, :].cuda(device=mdl.device, non_blocking=True)
                wc = wc.cuda(device=mdl.device, non_blocking=True)
            else:
                fp, ft = fp[:, :, :], ft[:, :, :]
            if config['scale']:
                # scale lats 24 - 50 -> 0-1
                fp[:, :, 0] = (fp[:, :, 0] - 24.) / (50. - 24.)
                ft[:, :, 0] = (ft[:, :, 0] - 24.) / (50. - 24.)

                # scale lons -126 - -66-> 0-1
                fp[:, :, 1] = (fp[:, :, 1] + 126.) / (-66. + 126.)
                ft[:, :, 1] = (ft[:, :, 1] + 126.) / (-66. + 126.)

                # scale alts/ETs -1000 - 64000 -> 0-1
                fp[:, :, 2] = (fp[:, :, 2] + 1000.) / (64000. + 1000.)
                ft[:, :, 2] = (ft[:, :, 2] + 1000.) / (64000. + 10000)
                wc = (wc + 1000.) / (64000. + 1000.)

            if mdl.paradigm == 'Regression':
                print("\nFlight {}/{}: ".format(batch_idx + 1, len(train_dl)) + str(len(fp)) + " points")
                for pt in tqdm.trange(len(wc)):
                    mdl.optimizer.zero_grad()
                    lat, lon, alt = fp[0][0].clone().detach(), fp[0][1].clone().detach(). fp[0][2].clone().detach()
                    if mdl.rnn_type == torch.nn.LSTM:
                        mdl.hidden_cell = (
                            lat.repeat(1, 1, mdl.rnn_hidden),
                            lon.repeat(1, 1, mdl.rnn_hidden),
                            alt.repeate(1,1,mdl.rnn_hidden))
                    elif mdl.rnn_type == torch.nn.GRU:
                        mdl.hidden_cell = torch.cat(lat.repeat(1, 1, mdl.rnn_hidden / 3),
                                                    lon.repeat(1, 1, mdl.rnn_hidden / 3),
                                                    alt.repeat(1, 1, mdl.rnn_hidden / 3),
                                                    torch.zeros(1, 1, mdl.rnn_hidden - (3*int(mdl.rnn_hidden/3))))
                    y_pred = mdl(wc[:pt + 1], fp[:pt + 1])
                    # print(y_pred)
                    single_loss = mdl.loss_function(y_pred, ft[:pt + 1].view(-1, 2))
                    if batch_idx < len(train_dl) - 1 and pt % 50 == 0:
                        single_loss.backward()
                        mdl.optimizer.step()
                    if batch_idx == len(train_dl) - 1:
                        losses = torch.cat((losses, single_loss.view(-1)))
            elif mdl.paradigm == 'Seq2Seq':
                mdl.optimizer.zero_grad()

                lat, lon, alt = fp[0,:,0], fp[0,:,1], fp[0,:,2]
                coordlen = int(mdl.rnn_hidden/3)
                padlen = mdl.rnn_hidden - 3*coordlen
                tns_coords = torch.vstack((lat.repeat(coordlen).view(-1, train_dl.batch_size),
                                        lon.repeat(coordlen).view(-1, train_dl.batch_size),
                                        alt.repeat(coordlen).view(-1, train_dl.batch_size),
                                        torch.zeros(padlen, len(lat),
                                        device=mdl.device))).T.view(1,-1,mdl.rnn_hidden)
                mdl.init_hidden_cell(tns_coords)

                y_pred = mdl(wc, fp[:])
                single_loss = mdl.loss_function(y_pred, ft)
                losses[batch_idx] = single_loss.view(-1).detach().item()
                single_loss.backward()
                if config['gradclip']: torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=2, norm_type=2)
                mdl.optimizer.step()

            if batch_idx == len(train_dl) - 1:
                epoch_losses[ep] = torch.mean(losses).view(-1)

        if hasattr(mdl, 'sched'): mdl.sched.step()

        mdl.eval()
        test_losses = torch.zeros(len(test_dl), device=mdl.device)
        for test_batch, (fp, ft, wc) in enumerate(test_dl):
            if mdl.device.__contains__('cuda'):
                fp = fp[:, :, :].cuda(device=mdl.device, non_blocking=True)
                ft = ft[:, :, :].cuda(device=mdl.device, non_blocking=True)
                wc = wc.cuda(device=mdl.device, non_blocking=True)
            else:
                fp, ft = fp[:, :, :], ft[:, :, :]

            if config['scale']:
                # scale lats 24 - 50 -> 0-1
                fp[:, :, 0] = (fp[:, :, 0] - 24.) / (50. - 24.)
                ft[:, :, 0] = (ft[:, :, 0] - 24.) / (50. - 24.)

                # scale lons -126 - -66-> 0-1
                fp[:, :, 1] = (fp[:, :, 1] + 126.) / (-66. + 126.)
                ft[:, :, 1] = (ft[:, :, 1] + 126.) / (-66. + 126.)

                # scale alts/ETs -1000 - 64000 -> 0-1
                fp[:, :, 2] = (fp[:, :, 2] + 1000.) / (64000. + 1000.)
                ft[:, :, 2] = (ft[:, :, 2] + 1000.) / (64000. + 10000)
                wc = (wc + 1000.) / (64000. + 1000.)

            if mdl.paradigm == 'Seq2Seq':
                mdl.optimizer.zero_grad()
                lat, lon, alt = fp[0, :, 0], fp[0, :, 1], fp[0, :, 2]
                coordlen = int(mdl.rnn_hidden / 3)
                padlen = mdl.rnn_hidden - 3 * coordlen
                tns_coords = torch.vstack((lat.repeat(coordlen).view(-1, test_dl.batch_size),
                                           lon.repeat(coordlen).view(-1, test_dl.batch_size),
                                           alt.repeat(coordlen).view(-1, test_dl.batch_size),
                                           torch.zeros(padlen, len(lat),
                                                       device=mdl.device))).T.view(1, -1, mdl.rnn_hidden)
                mdl.init_hidden_cell(tns_coords)

                y_pred = mdl(wc, fp[:])
                single_loss = mdl.loss_function(y_pred, ft)
                test_losses[test_batch] = single_loss.view(-1).detach().item()
        epoch_test_losses[ep] = test_losses.mean().view(-1)
        mdl.train()

        if (not config['raytune']) and ep % 10 == 0:
            if mdl.device.__contains__('cuda'):
                losses = losses.cpu()
                test_losses = test_losses.cpu()
            plt.plot(losses.detach().numpy(), label='train data')
            plt.plot(test_losses.detach().numpy(), label='test data')
            plt.legend()
            plt.title('Losses (Epoch {})'.format(ep + 1))
            plt.xlabel('Flight')
            plt.ylabel('Loss (MSE)')
            # plt.savefig('Eval Epoch{}.png'.format(ep+1), dpi=400)
            plt.savefig('Initialized Plots/Eval Epoch{}.png'.format(ep + 1), dpi=400)
            plt.close()
            del losses
            gc.collect()
        if (not config['raytune']) and ep % 50 == 0:
            mdl.save_model(override=True)
        if config['raytune']:
            if torch.isnan(epoch_test_losses[ep]): raise ValueError('Epoch Loss is NaN')
            with tune.checkpoint_dir(step=ep) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'checkpoint')
                with open(path,"w") as f:
                    f.write(json.dumps({'step': ep}))
                mdl.epochs_trained = ep
                if epoch_test_losses[ep] == epoch_test_losses[:ep+1].min():
                    todel = [x for x in os.listdir('.') if mdl.model_name()[:mdl.model_name().index('EPOCHS')] in x]
                    [os.remove(x) for x in todel]
                    mdl.save_model(override=True, override_path='.')
                    #torch.save({'struct_dict': mdl.struct_dict, 'state_dict': mdl.state_dict(),
                    #        'opt_dict': mdl.optimizer.state_dict(), 'epochs_trained': mdl.epochs_trained,
                    #        'batch_size': mdl.batch_size}, path)
                tune.report(loss=epoch_losses[ep].cpu().numpy(), valloss=epoch_test_losses[ep].cpu().numpy())


    if mdl.device.__contains__('cuda'):
        epoch_losses = epoch_losses.cpu()
        epoch_test_losses = epoch_test_losses.cpu()
    e_losses = epoch_losses.detach().numpy()
    e_test_losses = epoch_test_losses.detach().numpy()
    if not config['raytune']:
        plt.plot(e_losses, label='train data')
        plt.plot(e_test_losses, label='test data')
        plt.legend()
        plt.title('Avg Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Loss (MSE)')
        plt.savefig('Initialized Plots/Model Eval.png', dpi=300)
        plt.close()

        plt.plot(e_losses[:], label='train data')
        plt.plot(e_test_losses[:], label='test data')
        plt.legend()
        plt.title('Avg Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Loss (MSE)')
        plt.ylim([0, .01])
        plt.yticks(np.linspace(0, .01, 11))
        plt.savefig('Initialized Plots/Model Eval RangeLimit.png', dpi=300)
        plt.close()

        df_eloss = pd.DataFrame({'loss': e_losses, 'valloss': e_test_losses})
        df_eloss.to_csv('model_epoch_losses.txt')
        return mdl
