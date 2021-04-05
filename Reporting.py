import os
if os.name == 'posix':
    os.environ['PROJ_LIB'] = "/home/lab/anaconda3/pkgs/proj-7.2.0-he47e99f_1/share/proj/"
elif os.name == 'nt':
    # os.environ['PROJ_LIB'] = 'C:\\Users\\User\\anaconda3\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share'
    os.environ['PROJ_LIB'] = 'C:\\Users\\natha\\anaconda3\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share'
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import Utils.ErrorFn as fn

# Plots: Trajectories v. Known, MSE of each Flight in Test Data

mdls = [x for x  in os.listdir('Models') if os.path.isdir('Models/{}'.format(x)) and 'EPOCH' in x]
df_summary = pd.DataFrame()
for mdl in mdls:
    if not os.path.isdir('Output/{}/Figs'.format(mdl)):
        os.mkdir('Output/{}/Figs'.format(mdl))


    df_flight_losses = pd.read_csv('Models/{}/flight losses.txt'.format(mdl))
    evals = [x for x in os.listdir('Output/{}/Denormed'.format(mdl))
               if os.path.isfile('Output/{}/Denormed/{}'.format(mdl, x))]

    l2_preds_2d, l2_fps_2d = np.zeros(len(evals)), np.zeros(len(evals))
    l2_preds_3d, l2_fps_3d = np.zeros(len(evals)), np.zeros(len(evals))
    the_nmi_preds, the_m_preds, tve_preds = np.zeros(len(evals)), np.zeros(len(evals)), np.zeros(len(evals))
    phe_nmi_preds, pve_preds = [], []
    the_nmi_fps, the_m_fps, tve_fps = np.zeros(len(evals)), np.zeros(len(evals)), np.zeros(len(evals))
    phe_nmi_fps, pve_fps = [], []

    for e in tqdm.trange(len(evals)):
        # query df for normalized loss
        mse_loss = df_flight_losses[df_flight_losses['flight name'] == evals[e][5:]]['loss (MSE)'].values
        if len(mse_loss) > 0:
            # calculated L2norm and % improvement for denormed flight
            df_flight = pd.read_csv('Output/{}/Denormed/{}'.format(mdl,evals[e]))
            nda_fp = df_flight[['flight plan LAT', 'flight plan LON', 'flight plan ALT']].values
            nda_pred = df_flight[['predicted LAT', 'predicted LON', 'predicted ALT']].values
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

            the_nmi_preds[e], tve_preds[e] = np.mean(pred_phe_nmi), np.mean(pred_pve)
            the_nmi_fps[e], tve_fps[e] = np.mean(fp_phe_nmi), np.mean(fp_pve)
            phe_nmi_fps.extend(fp_phe_nmi); pve_fps.extend(fp_pve);
            phe_nmi_preds.extend(pred_phe_nmi); pve_preds.extend(pred_pve);
            l2_fps_2d[e], l2_fps_3d[e] = fn.L2Norm(fp_phe_nmi.reshape(-1,1)), fn.L2Norm(np.vstack((fp_phe_nmi, fp_pve/fn.ft_per_nmi)).T)
            l2_preds_2d[e], l2_preds_3d[e] = fn.L2Norm(pred_phe_nmi.reshape(-1,1)), fn.L2Norm(np.vstack((pred_phe_nmi, pred_pve/fn.ft_per_nmi)).T)



            # plot trajectory v flight plan v actual, include mse, reduction in title and norms in legend

            '''
            reduction_2d = fn.reduction(l2_fps_2d[e], l2_preds_2d[e])
            reduction_3d = fn.reduction(l2_fps_3d[e], l2_preds_3d[e])
            m = Basemap(width=6000000, height=4000000,
                        area_thresh=10000, projection='lcc',
                        lat_0=38., lon_0=-98., lat_1=25.)
            parallels = np.arange(0.,80.,10.)
            meridians = np.arange(10.,351.,20.)
            m.drawcoastlines()
            m.drawparallels(parallels, labels=[False, True, True, False])
            m.drawmeridians(meridians, labels=[True, False, False, True])
            fig2 = plt.gca()

            a = m.plot(nda_fp[:,1], nda_fp[:,0], latlon=True, color='blue', alpha=.5, label='flight plan L2-Norm: {:.3f}'.format(l2_fps[e]))
            b = m.plot(nda_pred[:,1], nda_pred[:,0], latlon=True, color='green', alpha=.5, label='prediction L2-Norm: {:.3f}'.format(l2_preds[e]))
            c = m.plot(nda_act[:,1], nda_act[:,0], latlon=True, color='red', alpha=.5, label='actual')
            plt.legend()
            plt.title("{}\nMSE: {:.4f}   Reduction: {:.3f}".format(evals[e][5:-4], mse_loss[0], reduction * 100))
            plt.savefig('Output/{}/Figs/{}.png'.format(mdl,evals[e]), dpi=300)
            plt.close()'''

            #TODO： 4D Visuals

        else:
            print('MISSING {}'.format(evals[e]))

    #re-form list-of-list errors into ndarrays
    phe_nmi_fps = np.array(phe_nmi_fps)
    phe_nmi_preds = np.array(phe_nmi_preds)
    pve_fps = np.array(pve_fps); pve_preds = np.array(pve_preds);


    # plot norm distributions (fp, pred) w/ % reduction in title
    reduction_2d = fn.reduction(l2_fps_2d[e], l2_preds_2d[e])
    plt.hist(l2_fps_2d, bins=50, alpha=.5, label='Flight Plans', density=True)
    plt.hist(l2_preds_2d, bins=50, alpha=.5, label='Predictions', density=True)
    plt.legend(); plt.xlabel('2D Norm (nmi.)');
    plt.title('L2 Norms (2D) Comparison\n{} Flights   Reduction: {:.3f} %'.format(len(l2_fps_2d),reduction_2d*100))
    plt.savefig('Output/{}/Figs/L2 Norms 2D.png'.format(mdl), dpi=300)
    plt.close()

    reduction_3d = fn.reduction(l2_fps_3d[e], l2_preds_3d[e])
    plt.hist(l2_fps_3d, bins=50, alpha=.5, label='Flight Plans', density=True)
    plt.hist(l2_preds_3d, bins=50, alpha=.5, label='Predictions', density=True)
    plt.legend(); plt.xlabel('3D Norm (nmi)');
    plt.title('L2 Norms (3D) Comparison\n{} Flights   Reduction: {:.3f} %'.format(len(l2_fps_3d),reduction_3d * 100))
    plt.savefig('Output/{}/Figs/L2 Norms 3D.png'.format(mdl), dpi=300)
    plt.close()

    # Plot PHE (nmi) distribution (MAPHE + stdev in title)
    plt.hist(phe_nmi_fps, bins=50, alpha=.5, label='Flight Plans', density=True)
    plt.hist(phe_nmi_preds, bins=50, alpha=.5, label='Predictions', density=True)
    plt.legend(); plt.xlabel('Error (nmi)');
    plt.title('Pointwise Horizontal Errors\n:{} Points'.format(len(phe_nmi_fps)))
    plt.savefig('Output/{}/Figs/PHE Hist.png'.format(mdl), dpi=300)
    plt.close()


    # Plot PVE (m) distribution (MAPVE + stdev in title)
    plt.hist(pve_fps, bins=50, alpha=.5, label='Flight Plans', density=True)
    plt.hist(pve_preds, bins=50, alpha=.5, label='Predictions', density=True)
    plt.legend(); plt.xlabel('Error (ft)');
    plt.title('Pointwise Vertical Errors\n{} Points'.format(len(pve_fps)))
    plt.savefig('Output/{}/Figs/PVE Hist.png'.format(mdl), dpi=300)
    plt.close()

    # Plot THE (nmi) distribution (MATHE + stdev in title)
    plt.hist(the_nmi_fps, bins=50, alpha=.5, label='Flight Plans', density=True)
    plt.hist(the_nmi_preds, bins=50, alpha=.5, label='Predictions', density=True)
    plt.legend(); plt.xlabel('Error (nmi)');
    plt.title('Trajectorywise Horizontal Errors\n{} Flights'.format(len(the_nmi_fps)))
    plt.savefig('Output/{}/Figs/THE Hist.png'.format(mdl), dpi=300)
    plt.close()

    # Plot TVE (m) distribution (MATVE + stdev in title)
    plt.hist(tve_fps, bins=50, alpha=.5, label='Flight Plans', density=True)
    plt.hist(tve_preds, bins=50, alpha=.5, label='Predictions', density=True)
    plt.legend(); plt.xlabel('Error (ft)');
    plt.title('Trajectorywise Vertical Errors\n{} Flights'.format(len(tve_fps)))
    plt.savefig('Output/{}/Figs/TVE Hist.png'.format(mdl), dpi=300)
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

    print('\u03BC Horiz (nmi)\t\u03BC Vert (m)\t\u03C3 Horiz (nmi)\t\u03C3 Vert (m)')
    print('FP PW{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(fp_maphe, fp_mapve, fp_stdphe, fp_stdpve))
    print('PRED PW{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(pred_maphe, pred_mapve, pred_stdphe, pred_stdpve))
    print('FP TW{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(fp_mathe, fp_matve, fp_stdthe, fp_stdtve))
    print('PRED TW{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(pred_mathe, pred_matve, pred_stdthe, pred_stdtve))
    d = {'model': mdl,'fp_maphe': fp_maphe, 'fp_mapve': fp_mapve, 'fp_stdphe': fp_stdphe, 'fp_stdpve': fp_stdpve,
         'pred_maphe': pred_maphe, 'pred_mapve': pred_mapve, 'pred_stdphe': pred_stdphe, 'pred_stdpve': pred_stdpve,
         'fp_mathe': fp_mathe, 'fp_matve': fp_matve, 'fp_stdthe': fp_stdthe, 'fp_stdtve': fp_stdtve,
         'pred_mathe': pred_mathe, 'pred_matve': pred_matve, 'pred_stdthe': pred_stdthe, 'pred_stdtve': pred_stdtve}
    df_summary = df_summary.append(d, ignore_index=True)
df_summary.to_csv('Output/summary_statistics.txt')