# FEATURE SETS
import numpy as np
FS1 =  ['Ux', 'Uy', 'Uz', 'k', 'omega', 'p', 'wallDistance',
                       'nut', 'epsilon', 'S_0,0', 'S_1,1', 'S_0,1', 'S_1,0',
                       'R_0,0', 'R_1,1', 'R_0,1', 'R_1,0', 'dk_0', 'dk_1',
                       'dp_0','dp_1']

FS2 =  ['tr_Sij',
                       'tr_Sij_2', 'tr_Sij_3', 'tr_Rij_2', 'tr_Rij_2_Sij_2',
                       'tr_Rij_2_Sij', 'tr_Rij_2_Sij_Rij_Sij_2']

FS3 =  ['Q_Crit', 'k_epsilon_norm', 'k_norm_u', 'wall_Re',
                       'P_epsilon', 'dimless_shear', 'marker_shear']

FS5 =  ['tr_Ak_2',
                       'tr_Ak_2_Sij', 'tr_Ak_2_Sij_2', 'tr_Ak_2_Sij_Ak_Sij_2',
                       'tr_Rij_Ak', 'tr_Rij_Ak_Sij', 'tr_Rij_Ak_Sij_2',
                       'tr_Rij_2_Ak_Sij', 'tr_Ak_2_Rij_Sij', 'tr_Rij_2_Ak_Sij_2',
                       'tr_Ak_2_Rij_Sij_2', 'tr_Rij_2_Sij_Ak_Sij_2',
                       'tr_Ak_2_Sij_Rij_Sij_2', 'tr_Ap_2', 'tr_Ap_2_Sij',
                       'tr_Ap_2_Sij_2', 'tr_Ap_2_Sij_Ap_Sij_2', 'tr_Rij_Ap',
                       'tr_Rij_Ap_Sij', 'tr_Rij_Ap_Sij_2', 'tr_Rij_2_Ap_Sij',
                       'tr_Ap_2_Rij_Sij', 'tr_Rij_2_Ap_Sij_2', 'tr_Ap_2_Rij_Sij_2',
                       'tr_Rij_2_Sij_Ap_Sij_2', 'tr_Ap_2_Sij_Rij_Sij_2', 'tr_Ap_Ak',
                       'tr_Ap_Ak_Sij', 'tr_Ap_Ak_Sij_2', 'tr_Ap_2_Ak_Sij',
                       'tr_Ak_2_Ap_Sij', 'tr_Ap_2_Ak_Sij_2', 'tr_Ak_2_Ap_Sij_2',
                       'tr_Ap_2_Sij_Ak_Sij_2', 'tr_Ak_2_Sij_Ap_Sij_2', 'tr_Rij_Ap_Ak',
                       'tr_Rij_Ap_Ak_Sij', 'tr_Rij_Ak_Ap_Sij', 'tr_Rij_Ap_Ak_Sij_2',
                       'tr_Rij_Ak_Ap_Sij_2', 'tr_Rij_Ap_Sij_Ap_Sij_2']

feature_set = np.hstack([FS1,FS2,FS3,FS5])

['Ux', 'Uy', 'k', 'nut', 'epsilon', 'p', 'wallDistance']