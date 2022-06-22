flight_mins = {'KSEA-KDEN': 170,'KATL-KMCO': 95,'KATL-KORD': 135,'KJFK-KLAX': 380,'KIAH-KBOS': 240,'KLAX-KSFO': 90,
               'KLAS-KLAX': 80, 'KSEA-KPDX': 60, 'KSEA-KLAX': 170, 'KSFO-KSEA': 130, 'KBOS-KLGA': 83, 'KATL-KLGA': 140,
               'KSJC-KLAX': 90, 'KPHX-KLAX': 90, 'KFLL-KATL': 115, 'KHOU-KDAL': 75, 'KORD-KLAX': 260, 'KSFO-KLAS': 110,
               'KSEA-KGEG': 65, 'KDFW-KLAX': 200, 'KDEN-KPHX': 115, 'KDEN-KLAX': 150, 'KORD-KLGA': 130,
               'KATL-KDFW': 140, 'KSFO-KSAN': 100, 'KSLC-KDEN': 90, 'KORD-KDCA': 120, 'KBOS-KPHL': 90, 'KJFK-KSFO': 390,
               'KLAX-KSMF': 90, 'KLAS-KSEA': 170, 'KDEN-KLAS': 120, 'KSFO-KSEA': 120}
flight_min_tol = {key: flight_mins[key]-30 for key in flight_mins}