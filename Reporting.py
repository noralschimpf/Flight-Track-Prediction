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
for mdl in mdls:
    df_flight_losses = pd.read_csv('Models/{}/flight losses.txt'.format(mdl))
    evals = [x for x in os.listdir('Output/{}/Denormed'.format(mdl))
               if os.path.isfile('Output/{}/Denormed/{}'.format(mdl, x))]

    l2_preds, l2_fps = np.zeros(len(evals)), np.zeros(len(evals))

    for e in tqdm.trange(len(evals)):
        # query df for normalized loss
        mse_loss = df_flight_losses[df_flight_losses['flight name'] == evals[e][5:]]['loss (MSE)'].values
        if len(mse_loss) > 0:
            # calculated L2norm and % improvement for denormed flight
            df_flight = pd.read_csv('Output/{}/Denormed/{}'.format(mdl,evals[e]))
            nda_fp = df_flight[['flight plan LAT', 'flight plan LON']].values
            nda_pred = df_flight[['predicted LAT', 'predicted LON']].values
            nda_act =  df_flight[['actual LAT', 'actual LON']].values
            '''
            a = torch.tensor(nda_fp).view(-1,2)
            b = torch.tensor(nda_act).view(-1,2)
            loss = torch.nn.MSELoss()
            print('\nnn.MSE:{:.3f}\nL2:{:.3f}\nMSE:{:.3f}'.format(loss(a,b),fn.L2Norm(b,a),fn.MSE(b,a)))
            '''
            l2_fps[e] = fn.L2Norm(nda_act, nda_fp)
            l2_preds[e] = fn.L2Norm(nda_act, nda_pred)
            reduction = fn.reduction(l2_fps[e], l2_preds[e])

            # plot trajectory v flight plan v actual, include mse, reduction in title and norms in legend

            m = Basemap(width=6000000, height=4000000,
                        area_thresh=10000, projection='lcc',
                        lat_0=38., lon_0=-98., lat_1=25.)
            '''m = Basemap(area_thresh=1000, projection='merc',
                        llcrnrlon=-130, urcrnrlon=-60, llcrnrlat=20, urcrnrlat=45)'''
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
            if not os.path.isdir('Output/{}/Figs'.format(mdl)):
                os.mkdir('Output/{}/Figs'.format(mdl))
            plt.savefig('Output/{}/Figs/{}.png'.format(mdl,evals[e]), dpi=300)
            plt.close()
        else:
            print('MISSING {}'.format(evals[e]))

    # plot norm distributions (fp, pred) w/ % reduction in title
    reduction = fn.reduction(l2_fps, l2_preds)
    plt.hist(l2_fps, bins=50, alpha=.5)
    plt.hist(l2_preds, bins=50, alpha=.5)
    plt.legend(['Flight Plans','Predictions'])
    plt.title('L2 Norms Comparison\nReduction: {:.3f} %'.format(reduction*100))
    plt.savefig('Output/{}/Figs/L2 Norms.png'.format(mdl), dpi=300)