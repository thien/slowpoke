import numpy as np

def subsquares(x):
    # This is the fastest way (albeit also very ugly) i can think of to approach this problem
    return np.array([((x[0] + x[4] + x[5] + x[8])/4),
    ((x[0] + x[1] + x[5] + x[8] + x[9])/5),
    ((x[1] + x[5] + x[6] + x[9])/4),
    ((x[1] + x[2] + x[6] + x[9] + x[10])/5),
    ((x[2] + x[6] + x[7] + x[10])/4),
    ((x[2] + x[3] + x[7] + x[10] + x[11])/5),
    ((x[4] + x[5] + x[8] + x[12] + x[13])/5),
    ((x[5] + x[8] + x[9] + x[13])/4),
    ((x[5] + x[6] + x[9] + x[13] + x[14])/5),
    ((x[6] + x[9] + x[10] + x[14])/4),
    ((x[6] + x[7] + x[10] + x[14] + x[15])/5),
    ((x[7] + x[10] + x[11] + x[15])/4),
    ((x[8] + x[12] + x[13] + x[16])/4),
    ((x[8] + x[9] + x[13] + x[16] + x[17])/5),
    ((x[9] + x[13] + x[14] + x[17])/4),
    ((x[9] + x[10] + x[14] + x[17] + x[18])/5),
    ((x[10] + x[14] + x[15] + x[18])/4),
    ((x[10] + x[11] + x[15] + x[18] + x[19])/5),
    ((x[12] + x[13] + x[16] + x[20] + x[21])/5),
    ((x[13] + x[16] + x[17] + x[21])/4),
    ((x[13] + x[14] + x[17] + x[21] + x[22])/5),
    ((x[14] + x[17] + x[18] + x[22])/4),
    ((x[14] + x[15] + x[18] + x[22] + x[23])/5),
    ((x[15] + x[18] + x[19] + x[23])/4),
    ((x[16] + x[20] + x[21] + x[24])/4),
    ((x[16] + x[17] + x[21] + x[24] + x[25])/5),
    ((x[17] + x[21] + x[22] + x[25])/4),
    ((x[17] + x[18] + x[22] + x[25] + x[26])/5),
    ((x[18] + x[22] + x[23] + x[26])/4),
    ((x[18] + x[19] + x[23] + x[26] + x[27])/5),
    ((x[20] + x[21] + x[24] + x[28] + x[29])/5),
    ((x[21] + x[24] + x[25] + x[29])/4),
    ((x[21] + x[22] + x[25] + x[29] + x[30])/5),
    ((x[22] + x[25] + x[26] + x[30])/4),
    ((x[22] + x[23] + x[26] + x[30] + x[31])/5),
    ((x[23] + x[26] + x[27] + x[31])/4),
    ((x[0] + x[1] + x[4] + x[5] + x[8] + x[9] + x[12] + x[13])/8),
    ((x[0] + x[1] + x[5] + x[6] + x[8] + x[9] + x[13] + x[14])/8),
    ((x[1] + x[2] + x[5] + x[6] + x[9] + x[10] + x[13] + x[14])/8),
    ((x[1] + x[2] + x[6] + x[7] + x[9] + x[10] + x[14] + x[15])/8),
    ((x[2] + x[3] + x[6] + x[7] + x[10] + x[11] + x[14] + x[15])/8),
    ((x[4] + x[5] + x[8] + x[9] + x[12] + x[13] + x[16] + x[17])/8),
    ((x[5] + x[6] + x[8] + x[9] + x[13] + x[14] + x[16] + x[17])/8),
    ((x[5] + x[6] + x[9] + x[10] + x[13] + x[14] + x[17] + x[18])/8),
    ((x[6] + x[7] + x[9] + x[10] + x[14] + x[15] + x[17] + x[18])/8),
    ((x[6] + x[7] + x[10] + x[11] + x[14] + x[15] + x[18] + x[19])/8),
    ((x[8] + x[9] + x[12] + x[13] + x[16] + x[17] + x[20] + x[21])/8),
    ((x[8] + x[9] + x[13] + x[14] + x[16] + x[17] + x[21] + x[22])/8),
    ((x[9] + x[10] + x[13] + x[14] + x[17] + x[18] + x[21] + x[22])/8),
    ((x[9] + x[10] + x[14] + x[15] + x[17] + x[18] + x[22] + x[23])/8),
    ((x[10] + x[11] + x[14] + x[15] + x[18] + x[19] + x[22] + x[23])/8),
    ((x[12] + x[13] + x[16] + x[17] + x[20] + x[21] + x[24] + x[25])/8),
    ((x[13] + x[14] + x[16] + x[17] + x[21] + x[22] + x[24] + x[25])/8),
    ((x[13] + x[14] + x[17] + x[18] + x[21] + x[22] + x[25] + x[26])/8),
    ((x[14] + x[15] + x[17] + x[18] + x[22] + x[23] + x[25] + x[26])/8),
    ((x[14] + x[15] + x[18] + x[19] + x[22] + x[23] + x[26] + x[27])/8),
    ((x[16] + x[17] + x[20] + x[21] + x[24] + x[25] + x[28] + x[29])/8),
    ((x[16] + x[17] + x[21] + x[22] + x[24] + x[25] + x[29] + x[30])/8),
    ((x[17] + x[18] + x[21] + x[22] + x[25] + x[26] + x[29] + x[30])/8),
    ((x[17] + x[18] + x[22] + x[23] + x[25] + x[26] + x[30] + x[31])/8),
    ((x[18] + x[19] + x[22] + x[23] + x[26] + x[27] + x[30] + x[31])/8),
    ((x[0] + x[1] + x[4] + x[5] + x[6] + x[8] + x[9] + x[12] + x[13] + x[14] + x[16] + x[17])/12),
    ((x[0] + x[1] + x[2] + x[5] + x[6] + x[8] + x[9] + x[10] + x[13] + x[14] + x[16] + x[17] + x[18])/13),
    ((x[1] + x[2] + x[5] + x[6] + x[7] + x[9] + x[10] + x[13] + x[14] + x[15] + x[17] + x[18])/12),
    ((x[1] + x[2] + x[3] + x[6] + x[7] + x[9] + x[10] + x[11] + x[14] + x[15] + x[17] + x[18] + x[19])/13),
    ((x[4] + x[5] + x[6] + x[8] + x[9] + x[12] + x[13] + x[14] + x[16] + x[17] + x[20] + x[21] + x[22])/13),
    ((x[5] + x[6] + x[8] + x[9] + x[10] + x[13] + x[14] + x[16] + x[17] + x[18] + x[21] + x[22])/12),
    ((x[5] + x[6] + x[7] + x[9] + x[10] + x[13] + x[14] + x[15] + x[17] + x[18] + x[21] + x[22] + x[23])/13),
    ((x[6] + x[7] + x[9] + x[10] + x[11] + x[14] + x[15] + x[17] + x[18] + x[19] + x[22] + x[23])/12),
    ((x[8] + x[9] + x[12] + x[13] + x[14] + x[16] + x[17] + x[20] + x[21] + x[22] + x[24] + x[25])/12),
    ((x[8] + x[9] + x[10] + x[13] + x[14] + x[16] + x[17] + x[18] + x[21] + x[22] + x[24] + x[25] + x[26])/13),
    ((x[9] + x[10] + x[13] + x[14] + x[15] + x[17] + x[18] + x[21] + x[22] + x[23] + x[25] + x[26])/12),
    ((x[9] + x[10] + x[11] + x[14] + x[15] + x[17] + x[18] + x[19] + x[22] + x[23] + x[25] + x[26] + x[27])/13),
    ((x[12] + x[13] + x[14] + x[16] + x[17] + x[20] + x[21] + x[22] + x[24] + x[25] + x[28] + x[29] + x[30])/13),
    ((x[13] + x[14] + x[16] + x[17] + x[18] + x[21] + x[22] + x[24] + x[25] + x[26] + x[29] + x[30])/12),
    ((x[13] + x[14] + x[15] + x[17] + x[18] + x[21] + x[22] + x[23] + x[25] + x[26] + x[29] + x[30] + x[31])/13),
    ((x[14] + x[15] + x[17] + x[18] + x[19] + x[22] + x[23] + x[25] + x[26] + x[27] + x[30] + x[31])/12),
    ((x[0] + x[1] + x[2] + x[4] + x[5] + x[6] + x[8] + x[9] + x[10] + x[12] + x[13] + x[14] + x[16] + x[17] + x[18] + x[20] + x[21] + x[22])/18),
    ((x[0] + x[1] + x[2] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[21] + x[22] + x[23])/18),
    ((x[1] + x[2] + x[3] + x[5] + x[6] + x[7] + x[9] + x[10] + x[11] + x[13] + x[14] + x[15] + x[17] + x[18] + x[19] + x[21] + x[22] + x[23])/18),
    ((x[4] + x[5] + x[6] + x[8] + x[9] + x[10] + x[12] + x[13] + x[14] + x[16] + x[17] + x[18] + x[20] + x[21] + x[22] + x[24] + x[25] + x[26])/18),
    ((x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26])/18),
    ((x[5] + x[6] + x[7] + x[9] + x[10] + x[11] + x[13] + x[14] + x[15] + x[17] + x[18] + x[19] + x[21] + x[22] + x[23] + x[25] + x[26] + x[27])/18),
    ((x[8] + x[9] + x[10] + x[12] + x[13] + x[14] + x[16] + x[17] + x[18] + x[20] + x[21] + x[22] + x[24] + x[25] + x[26] + x[28] + x[29] + x[30])/18),
    ((x[8] + x[9] + x[10] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26] + x[29] + x[30] + x[31])/18),
    ((x[9] + x[10] + x[11] + x[13] + x[14] + x[15] + x[17] + x[18] + x[19] + x[21] + x[22] + x[23] + x[25] + x[26] + x[27] + x[29] + x[30] + x[31])/18),
    ((x[0] + x[1] + x[2] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[12] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[20] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26])/24),
    ((x[0] + x[1] + x[2] + x[3] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26] + x[27])/25),
    ((x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[12] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[20] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26] + x[28] + x[29] + x[30] + x[31])/25),
    ((x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26] + x[27] + x[29] + x[30] + x[31])/24),
    ((x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19] + x[20] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26] + x[27] + x[28] + x[29] + x[30] + x[31])/32)])