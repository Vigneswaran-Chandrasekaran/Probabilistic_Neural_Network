import numpy as np
import pandas as pd
import pywt 

def bonn_dataset(wavelet_family = 'db10', level = 4):

    data = np.load("../Healthcare_signal_processing/Datasets/Bonn/data_all.npz")
    db = pywt.Wavelet(wavelet_family)
    a4 = []; d4 = []; d3 = []; d2 = []; d1 = []
    for i in data.keys():
        for samp in data[i]:
            cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(samp, db, level = level)
            a4.append(cA4)
            d4.append(cD4)
            d3.append(cD3)
            d2.append(cD2)
            d1.append(cD1)

    a4 = np.array(a4)
    d4 = np.array(d4)
    d3 = np.array(d3)
    d2 = np.array(d2)
    d1 = np.array(d1)
    print("[INFO] Dataset processing completed")
    return [a4, d4, d3, d2, d1]

def bern_dataset(wavelet_family = 'db10', level = 4):
    
    data = np.load("../Healthcare_signal_processing/Datasets/Barca/Focal_all_32bit.npz")
    data['d'].shape
    focal = np.array( np.split(data['d'], 3750) )
    data = np.load("../Healthcare_signal_processing/Datasets/Barca/NFocal_all_32bit.npz")
    nfocal = np.array( np.split(data['d'], 3750) )
    db = pywt.Wavelet(wavelet_family)
    a4 = []; d4 = []; d3 = []; d2 = []; d1 = []

    for samp in focal:
        cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(samp, db, level = level)
        a4.append(cA4)
        d4.append(cD4)
        d3.append(cD3)
        d2.append(cD2)
        d1.append(cD1)

    for samp in nfocal:
        cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(samp, db, level = level)
        a4.append(cA4)
        d4.append(cD4)
        d3.append(cD3)
        d2.append(cD2)
        d1.append(cD1)

    a4 = np.array(a4)
    d4 = np.array(d4)
    d3 = np.array(d3)
    d2 = np.array(d2)
    d1 = np.array(d1)

    return [a4, d4, d3, d2, d1]
