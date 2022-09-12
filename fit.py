import warnings

import torch
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os, shutil, gc, tqdm, json, random, inspect
from custom_dataset import pad_batch
from libmodels.CONV_RECURRENT import CONV_RECURRENT
from libmodels.model import load_model, init_constant
from ray import tune


def store_losses(e_losses: np.ndarray, e_test_losses: np.ndarray, prd: str, mdlname: str):
    # e_losses, e_test_losses = e_losses.detach().cpu().to_numpy(), e_test_losses.detach().cpu().to_numpy()
    el, etl = None, None
    for e, e_orig in [(el, e_losses), (etl, e_test_losses)]:
        if (e_orig==0).any(): e = e_orig[:np.where(e_losses==0)[0][0]]
        else: e = e_orig
    # el, etl = e_losses[:np.where(e_losses==0)[0][0]], e_test_losses[:np.where(e_test_losses==0)[0][0]]
    plt.plot(el, label='train data')
    plt.plot(etl, label='test data')
    plt.legend()
    plt.title('Avg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss (MSE)')
    plt.savefig(f'Initialized Plots/{prd}/{mdlname}/Model Eval.png', dpi=300)
    plt.close()

    plt.plot(el, label='train data')
    plt.plot(etl, label='test data')
    plt.legend()
    plt.title('Avg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss (MSE)')
    plt.ylim([0, .01])
    plt.yticks(np.linspace(0, .01, 11))
    plt.savefig(f'Initialized Plots/{prd}/{mdlname}/Model Eval RangeLimit.png', dpi=300)
    plt.close()

    df_eloss = pd.DataFrame({'loss': el, 'valloss': etl})
    dfpath = f'Models/{prd}/{mdlname}'
    if not os.path.isdir(dfpath): os.makedirs(dfpath)
    df_eloss.to_csv(f'{dfpath}/model_epoch_losses.txt')



def fit(config: dict, train_dataset: torch.utils.data.DataLoader, test_dataset: torch.utils.data.Dataset, checkpoint_dir=None,
        raytune: bool = False, determinist: bool = True, const: bool = False, gradclip: bool = False, model_name: str = 'Default',
        scale: bool = False):
    #print(config['mdldir'])

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

    mdl = CONV_RECURRENT(config=config)
    if config["const"]: mdl.apply(init_constant)


    mdl.update_dict(); eps = config['epochs']
    mdlpath = f'Initialized Plots/{"&".join(mdl.features)}/{mdl.model_name().replace("EPOCHS0",f"EPOCHS{eps}")}'
    if not os.path.isdir(mdlpath): os.makedirs(mdlpath)

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
                fpa, fta, wca = fp[:,:,2], ft[:,:,2], wc
                # print(f'FP: {fpa.min()}-{fpa.max()}\nFT:{fta.min()}-{fta.max()}\nWC:{wca.min()}-{wca.max()}')
                fp[:, :, 2] = (fp[:, :, 2] + 1000.) / (64000. + 1000.)
                ft[:, :, 2] = (ft[:, :, 2] + 1000.) / (64000. + 1000.)
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
            plt.savefig(f'{mdlpath}/Eval Epoch{ep+1}.png', dpi=400)
            plt.close()
            del losses
            gc.collect()

        # stop / re-run model training if model diverges
        if torch.isnan(epoch_losses[ep]) or torch.isnan(epoch_test_losses[ep]):
            mdl.epochs_trained = config['epochs']
            mdl.epochs_actual = ep
            try: store_losses(epoch_losses.cpu().detach().numpy(), epoch_test_losses.cpu().detach().numpy(), '&'.join(mdl.features), mdl.model_name())
            except Exception as e:
                warnings.warn("store_losses failed")
            raise ValueError(f'{mdl.model_name()} diverged at {ep} epochs', mdl)

        # regular checkpointing
        elif (not config['raytune']) and ep % config['checkpoint_freq_epochs'] == 0:
                mdl.save_model(override=True, appenddir='fold0')

        # Early-stopping
        if ep>config['stopping_iter_no_change']:
            ls = epoch_losses if len(test_dataset)==0 else epoch_test_losses
            epoch_diffs = ls[ep-config['stopping_iter_no_change']:ep] - \
                               ls[ep-config['stopping_iter_no_change']-1:ep-1]
            if (torch.abs(epoch_diffs) < config['stopping_tol']).all() and not (epoch_diffs < 0).any():
                mdl.epochs_trained = config['epochs']
                mdl.epochs_actual = ep
                mdl.save_model(override=True, appenddir='fold0')
                try: store_losses(epoch_losses.cpu().detach().numpy(), epoch_test_losses.cpu().detach().numpy(),
                             '&'.join(mdl.features), mdl.model_name())
                except Exception as e:
                    warnings.warn("store_losses failed")
                raise OSError(f'{mdl.model_name()}: Early Stopping. {ep} epochs complete', mdl)

        if raytune:
            if torch.isnan(epoch_test_losses[ep]):
                mdl.epochs_trained = config['epochs']
                mdl.epochs_actual = ep
                try: store_losses(epoch_losses.cpu().detach().numpy(), epoch_test_losses.cpu().detach().numpy(),
                             '&'.join(mdl.features), mdl.model_name())
                except Exception as e:
                    warnings.warn("store_losses failed")
                raise ValueError('Epoch Loss is NaN', mdl)
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
        mdl.epochs_trained = config['epochs']
        mdl.epochs_actual = ep
        try: store_losses(e_losses, e_test_losses, '&'.join(mdl.features), mdl.model_name())
        except Exception as e:
            warnings.warn("store_losses failed")
    return mdl
