import csv, os, shutil, pandas as pd, numpy as np, matplotlib.pyplot as plt
os.chdir('/home/dualboot/Desktop/Trajectory Generated Tracks')
SORTED = True
if not SORTED:
    file = open('/home/dualboot/Desktop/to rafael.csv','r',newline='')
    rd = csv.reader(file,delimiter=',')
    next(rd)
    for row in rd:
        flt, alt = row[1], row[3]
        oldfl = [x for x in os.listdir() if flt in x and '.csv' in x][0]
        if not os.path.isdir(alt): os.mkdir(alt)
        os.rename(oldfl, '{}/{}_{}_Tracks.csv'.format(alt,oldfl[:-10],alt))
    file.close()
    SORTED = True
if SORTED: alts = [x for x in os.listdir() if os.path.isdir(x)]
for alt in alts:
    if not SORTED: files = [x for x in os.listdir() if alt in x and '.csv' in x]
    else: os.chdir(alt); files = [x for x in os.listdir() if '.csv' in x]
    altfig, altax = plt.subplots(1,1)
    spdfig, spdax = plt.subplots(1,1)
    diffig, difax = plt.subplots(1,1)
    if not SORTED: dr = alt[1:-1]
    else: dr = alt
    for file in files:
        if not SORTED:
            if not os.path.isdir(dr): os.mkdir(dr)
            os.rename(file, os.path.join(dr, file)); os.chdir(dr)
        df = pd.read_csv(file)
        # generate altitude, airspeed plots
        t, alts = df['SIM_TIMESTAMP'].values[:10*60], df['ALTITUDE'].values[:10*60]
        t, alts = t[0:len(t) + 1:60], alts[0:len(alts) + 1:60]
        altdif = [alts[i] - alts[i - 1] for i in range(1, len(alts))]
        altax.plot(alts, color='b', alpha=0.1)
        spdax.plot(df['TRUE_AIRSPEED'].values, color='b', alpha=0.1)
        difax.plot(altdif, color='b', alpha=0.1)

    os.chdir('..')

    altax.set_title('Altitudes (CA = {})'.format(dr))
    altax.set_ylabel('Altitude (x100 ft)');
    altax.set_xlabel('timestamp')
    altfig.savefig('Flight Alts CA={}.png'.format(dr), dpi=300)

    spdax.set_title('Airspeeds (CA = {})'.format(dr))
    spdax.set_ylabel('Airspeed (knots)'); spdax.set_xlabel('timestamp')
    spdfig.savefig('Flight Spds CA={}.png'.format(dr), dpi=300)

    difax.set_title('Climb Rates (CA = {})'.format(dr))
    difax.set_ylabel('Rate (x100 ft / min)'); difax.set_xlabel('timestamp')
    diffig.savefig('Flight Climb CA={}.png'.format(dr), dpi=300)

    plt.close(); altfig.clf(); spdfig.clf(); diffig.clf(); altax.cla(); spdax.cla(); difax.cla()