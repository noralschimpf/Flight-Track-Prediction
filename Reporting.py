import os
if os.name == 'posix':
    #os.environ['PROJ_LIB'] = "/home/lab/anaconda3/pkgs/proj-7.2.0-he47e99f_1/share/proj/"
    os.environ['PROJ_LIB'] = "/home/dualboot/anaconda3/pkgs/proj4-5.2.0-he6710b0_1/share/proj"
elif os.name == 'nt':
    # os.environ['PROJ_LIB'] = 'C:\\Users\\User\\anaconda3\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share'
    os.environ['PROJ_LIB'] = 'C:\\Users\\natha\\anaconda3\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share'
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits import mplot3d
import Utils.ErrorFn as fn

plotting = False

# Plots: Trajectories v. Known, MSE of each Flight in Test Data
valid_products = ['ECHO_TOP','VIL','tmp','uwind','vwind']
mdl_product_dirs = [os.path.join(os.path.abspath('.'), 'Models/{}'.format(x)) for x in os.listdir('Models') if
                        os.path.isdir('Models/{}'.format(x)) and any([y in x for y in valid_products])]
mdl_dirs = [os.path.join(x,y) for x in mdl_product_dirs for y in os.listdir(x) if 'EPOCHS' in y]
mdl_dirs= [x for x in mdl_dirs if not len(os.listdir(x)) == sum([1 for y in os.listdir(x) if 'IGNORE' in y])]
mdls = [x for x in os.listdir('Models') if os.path.isdir('Models/{}'.format(x)) and 'EPOCH' in x]
df_val_summary, df_test_summary = pd.DataFrame(), pd.DataFrame()
for mdl in mdl_dirs:
    print(mdl)
    outdir = mdl.replace('Models','Output')
    if not os.path.isdir('{}/Figs'.format(outdir)):
        os.makedirs('{}/Figs'.format(outdir))


    df_flight_losses = pd.read_csv('{}/total flight losses.txt'.format(mdl))
    evals = [x for x in os.listdir('{}/Denormed'.format(outdir))
               if os.path.isfile('{}/Denormed/{}'.format(outdir, x)) and 'eval' in x]
    tests = [x for x in os.listdir('{}/Denormed'.format(outdir))
               if os.path.isfile('{}/Denormed/{}'.format(outdir, x)) and 'test' in x]

    for files in [evals, tests]:

        l2_preds_2d, l2_fps_2d = np.zeros(len(files)), np.zeros(len(files))
        l2_preds_3d, l2_fps_3d = np.zeros(len(files)), np.zeros(len(files))
        the_nmi_preds, the_m_preds, tve_preds = np.zeros(len(files)), np.zeros(len(files)), np.zeros(len(files))
        phe_nmi_preds, pve_preds = [], []
        the_nmi_fps, the_m_fps, tve_fps = np.zeros(len(files)), np.zeros(len(files)), np.zeros(len(files))
        phe_nmi_fps, pve_fps = [], []
        if len(files)==0: continue
        for f in tqdm.trange(len(files)):
            # query df for normalized loss
            start_idx = 5 if files == evals else 8
            mse_loss = df_flight_losses[df_flight_losses['flight name'] == files[f][start_idx:]]['loss (MSE)'].values
            if len(mse_loss) > 0:
                # calculated L2norm and % improvement for denormed flight
                df_flight = pd.read_csv('{}/Denormed/{}'.format(outdir, files[f]))
                nda_fp = df_flight[['flight plan LAT', 'flight plan LON', 'flight plan ALT']].values
                nda_pred = df_flight[['predicted LAT', 'predicted LON', 'predicted ALT']].values
                if np.isnan(nda_pred).any(): continue
                nda_act =  df_flight[['actual LAT', 'actual LON', 'actual ALT']].values
                '''
                a = torch.tensor(nda_fp).view(-1,2)
                b = torch.tensor(nda_act).view(-1,2)
                loss = torch.nn.MSELoss()
                print('\nnn.MSE:{:.3f}\nL2:{:.3f}\nMSE:{:.3f}'.format(loss(a,b),fn.L2Norm(b,a),fn.MSE(b,a)))
                '''
                fp_phe_nmi, fp_pve = fn.PointwiseError(nda_act, nda_fp, h_units='nmi')
                pred_phe_nmi, pred_pve = fn.PointwiseError(nda_act, nda_pred)
                df_flight['fp PHE (nmi)'] = fp_phe_nmi; df_flight['fp PVE (ft)'] = fp_pve;
                df_flight['pred PHE (nmi)'] = pred_phe_nmi; df_flight['pred PVE (ft)'] = pred_pve;

                the_nmi_preds[f], tve_preds[f] = np.mean(pred_phe_nmi), np.mean(pred_pve)
                the_nmi_fps[f], tve_fps[f] = np.mean(fp_phe_nmi), np.mean(fp_pve)
                phe_nmi_fps.extend(fp_phe_nmi); pve_fps.extend(fp_pve);
                phe_nmi_preds.extend(pred_phe_nmi); pve_preds.extend(pred_pve);
                l2_fps_2d[f], l2_fps_3d[f] = fn.L2Norm(fp_phe_nmi.reshape(-1, 1)), fn.L2Norm(np.vstack((fp_phe_nmi, fp_pve / fn.ft_per_nmi)).T)
                l2_preds_2d[f], l2_preds_3d[f] = fn.L2Norm(pred_phe_nmi.reshape(-1, 1)), fn.L2Norm(np.vstack((pred_phe_nmi, pred_pve / fn.ft_per_nmi)).T)



                # plot trajectory v flight plan v actual, include mse, reduction in title and norms in legend

                if f%500 == 0 and plotting:
                    reduction_2d = fn.reduction(l2_fps_2d[f], l2_preds_2d[f])
                    reduction_3d = fn.reduction(l2_fps_3d[f], l2_preds_3d[f])
                    m = Basemap(width=6000000, height=4000000,
                                area_thresh=10000, projection='lcc',
                                lat_0=38., lon_0=-98., lat_1=25.)
                    parallels = np.arange(0.,80.,10.)
                    meridians = np.arange(10.,351.,20.)
                    m.drawcoastlines()
                    m.drawparallels(parallels, labels=[False, True, True, False])
                    m.drawmeridians(meridians, labels=[True, False, False, True])
                    fig2 = plt.gca()

                    a = m.plot(nda_fp[:,1], nda_fp[:,0], latlon=True, color='blue', alpha=.5, label='flight plan L2-Norm: {:.3f}'.format(l2_fps_2d[f]))
                    b = m.plot(nda_pred[:,1], nda_pred[:,0], latlon=True, color='green', alpha=.5, label='prediction L2-Norm: {:.3f}'.format(l2_preds_2d[f]))
                    c = m.plot(nda_act[:,1], nda_act[:,0], latlon=True, color='red', alpha=.5, label='actual')
                    plt.legend()
                    plt.title("{}\nMSE: {:.4f}   Reduction: {:.3f}".format(files[f][5:-4], mse_loss[0], reduction_2d * 100))
                    plt.savefig('{}/Figs/2D {}.png'.format(outdir, files[f]), dpi=300)
                    plt.close()

                    fig = plt.figure(); ax = plt.axes(projection='3d')
                    ax.plot(nda_fp[:, 1], nda_fp[:, 0],nda_fp[:,2], color='blue',
                            label='flight plan L2-Norm: {:.3f}'.format(l2_fps_2d[f]))
                    ax.plot(nda_pred[:, 1], nda_pred[:, 0], nda_pred[:,2], color='green',
                            label='prediction L2-Norm: {:.3f}'.format(l2_preds_2d[f]))
                    ax.plot(nda_act[:, 1], nda_act[:, 0], nda_act[:,2], color='red', label='actual')
                    ax.set_xlabel('degrees latitude')
                    ax.set_ylabel('degree longitude')
                    ax.set_zlabel('altitude (ft)')
                    plt.legend(); plt.title("{}\nMSE: {:.4f}   Reduction: {:.3f}".format(files[f][:-4], mse_loss[0], reduction_3d * 100))
                    fig.tight_layout()
                    plt.savefig('{}/Figs/3D {}.png'.format(outdir, files[f]), dpi=300)
                    plt.close(); fig.clf(); ax.cla();


            else:
                print('MISSING {}'.format(files[f]))

        #re-form list-of-list errors into ndarrays
        phe_nmi_fps = np.array(phe_nmi_fps)
        phe_nmi_preds = np.array(phe_nmi_preds)
        pve_fps = np.array(pve_fps); pve_preds = np.array(pve_preds);

        # plot norm distributions (fp, pred) w/ % reduction in title
        reduction_2d = fn.reduction(l2_fps_2d[f], l2_preds_2d[f])
        plt.hist(l2_fps_2d, bins=50, alpha=.5, label='Flight Plans', density=True)
        plt.hist(l2_preds_2d, bins=50, alpha=.5, label='Predictions', density=True)
        plt.legend(); plt.xlabel('2D Norm (nmi.)');
        plt.title('{} L2 Norms (2D)\n{} Flights   Reduction: {:.3f} %'.format('Validation' if files==evals else 'Test',
                                                                               len(l2_fps_2d),reduction_2d*100))
        plt.savefig('{}/Figs/{} L2 Norms 2D.png'.format(outdir, 'Validation' if files==evals else 'Test'), dpi=300)
        plt.close()

        reduction_3d = fn.reduction(l2_fps_3d[f], l2_preds_3d[f])
        plt.hist(l2_fps_3d, bins=50, alpha=.5, label='Flight Plans', density=True)
        plt.hist(l2_preds_3d, bins=50, alpha=.5, label='Predictions', density=True)
        plt.legend(); plt.xlabel('3D Norm (nmi)');
        plt.title('{} L2 Norms (3D)\n{} Flights   Reduction: {:.3f} %'.format('Validation' if files==evals else 'Test',
                                                                              len(l2_fps_3d),reduction_3d * 100))
        plt.savefig('{}/Figs/{} L2 Norms 3D.png'.format(outdir, 'Validation' if files==evals else 'Test'), dpi=300)
        plt.close()

        # Plot PHE (nmi) distribution (MAPHE + stdev in title)
        plt.hist(phe_nmi_fps, bins=50, alpha=.5, label='Flight Plans', density=True)
        plt.hist(phe_nmi_preds, bins=50, alpha=.5, label='Predictions', density=True)
        plt.legend(); plt.xlabel('Error (nmi)');
        plt.title('{} Pointwise Horizontal Errors\n{} Points'.format('Validation' if files==evals else 'Test', len(phe_nmi_fps)))
        plt.savefig('{}/Figs/{} PHE Hist.png'.format(outdir, 'Validation' if files==evals else 'Test'), dpi=300)
        plt.close()


        # Plot PVE (m) distribution (MAPVE + stdev in title)
        plt.hist(pve_fps, bins=50, alpha=.5, label='Flight Plans', density=True)
        plt.hist(pve_preds, bins=50, alpha=.5, label='Predictions', density=True)
        plt.legend(); plt.xlabel('Error (ft)');
        plt.title('{} Pointwise Vertical Errors\n{} Points'.format('Validation' if files==evals else 'Test', len(pve_fps)))
        plt.savefig('{}/Figs/{} PVE Hist.png'.format(outdir, 'Validation' if files==evals else 'Test'), dpi=300)
        plt.close()

        # Plot THE (nmi) distribution (MATHE + stdev in title)
        plt.hist(the_nmi_fps, bins=50, alpha=.5, label='Flight Plans', density=True)
        plt.hist(the_nmi_preds, bins=50, alpha=.5, label='Predictions', density=True)
        plt.legend(); plt.xlabel('Error (nmi)');
        plt.title('{} Trajectorywise Horizontal Errors\n{} Flights'.format('Validation' if files==evals else 'Test', len(the_nmi_fps)))
        plt.savefig('{}/Figs/{} THE Hist.png'.format(outdir, 'Validation' if files==evals else 'Test'), dpi=300)
        plt.close()

        # Plot TVE (m) distribution (MATVE + stdev in title)
        plt.hist(tve_fps, bins=50, alpha=.5, label='Flight Plans', density=True)
        plt.hist(tve_preds, bins=50, alpha=.5, label='Predictions', density=True)
        plt.legend(); plt.xlabel('Error (ft)');
        plt.title('{} Trajectorywise Vertical Errors\n{} Flights'.format('Validation' if files==evals else 'Test', len(tve_fps)))
        plt.savefig('{}/Figs/{} TVE Hist.png'.format(outdir, 'Validation' if files==evals else 'Test'), dpi=300)
        plt.close()

        #Print MAPHE, MAPVE, MATHE, MATVE + stdev's
        # and save as CSV
        pred_maphe, fp_maphe = np.mean(np.abs(phe_nmi_preds)), np.mean(np.abs(phe_nmi_fps))
        pred_stdphe, fp_stdphe = np.std(phe_nmi_preds), np.std(phe_nmi_fps)
        pred_mapve, fp_mapve = np.mean(np.abs(pve_preds)), np.mean(np.abs(pve_fps))
        pred_stdpve, fp_stdpve = np.std(pve_preds), np.std(pve_fps)
        pred_mathe, fp_mathe = np.mean(np.abs(the_nmi_preds)), np.mean(np.abs(the_nmi_fps))
        pred_stdthe, fp_stdthe = np.std(the_nmi_preds), np.std(the_nmi_fps)
        pred_matve, fp_matve = np.mean(np.abs(tve_preds)), np.mean(np.abs(tve_fps))
        pred_stdtve, fp_stdtve = np.std(tve_preds), np.std(tve_fps)

        print('        \u03BC Horiz (nmi)\t\u03BC Vert (m)\t\u03C3 Horiz (nmi)\t\u03C3 Vert (m)')
        print('FP PW   {:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(fp_maphe, fp_mapve, fp_stdphe, fp_stdpve))
        print('PRED PW {:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(pred_maphe, pred_mapve, pred_stdphe, pred_stdpve))
        print('FP TW   {:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(fp_mathe, fp_matve, fp_stdthe, fp_stdtve))
        print('PRED TW {:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(pred_mathe, pred_matve, pred_stdthe, pred_stdtve))
        d = {'model': mdl.split('/')[-1], 'products': mdl.split('/')[-2], 'fp_maphe': fp_maphe, 'fp_mapve': fp_mapve, 'fp_stdphe': fp_stdphe, 'fp_stdpve': fp_stdpve,
             'pred_maphe': pred_maphe, 'pred_mapve': pred_mapve, 'pred_stdphe': pred_stdphe, 'pred_stdpve': pred_stdpve,
             'fp_mathe': fp_mathe, 'fp_matve': fp_matve, 'fp_stdthe': fp_stdthe, 'fp_stdtve': fp_stdtve,
             'pred_mathe': pred_mathe, 'pred_matve': pred_matve, 'pred_stdthe': pred_stdthe, 'pred_stdtve': pred_stdtve}
        if files == evals: df_val_summary = df_val_summary.append(d, ignore_index=True)
        elif files == tests: df_test_summary = df_test_summary.append(d, ignore_index=True)
    df_val_summary.to_csv('Output/validation_summary_statistics.txt')
    df_test_summary.to_csv('Output/test_summary_statistics.txt')
