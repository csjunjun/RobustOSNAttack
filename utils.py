import os   
import numpy as np
from PIL.JpegImagePlugin import convert_dict_qtables
from PIL import Image

def jpeg_qtableinv(stream, tnum=0, force_baseline=None):
    assert tnum == 0 or tnum == 1, 'Table number must be 0 or 1'

    if force_baseline is None:
        th_high = 32767
    elif force_baseline == 0:
        th_high = 32767
    else:
        th_high = 255

    h = np.asarray(convert_dict_qtables(Image.open(stream).quantization)[tnum]).reshape((8, 8))

    if tnum == 0:
        # This is table 0 (the luminance table): #JPEG (ISO 10918)
        t = np.matrix(
             [[16,  11,  10,  16,  24,  40,  51,  61],
              [12,  12,  14,  19,  26,  58,  60,  55],
              [14,  13,  16,  24,  40,  57,  69,  56],
              [14,  17,  22,  29,  51,  87,  80,  62],
              [18,  22,  37,  56,  68, 109, 103,  77],
              [24,  35,  55,  64,  81, 104, 113,  92],
              [49,  64,  78,  87, 103, 121, 120, 101],
              [72,  92,  95,  98, 112, 100, 103,  99]])

    elif tnum == 1:
        # This is table 1 (the chrominance table):
        t = np.matrix(
            [[17,  18,  24,  47,  99,  99,  99,  99],
             [18,  21,  26,  66,  99,  99,  99,  99],
             [24,  26,  56,  99,  99,  99,  99,  99],
             [47,  66,  99,  99,  99,  99,  99,  99],
             [99,  99,  99,  99,  99,  99,  99,  99],
             [99,  99,  99,  99,  99,  99,  99,  99],
             [99,  99,  99,  99,  99,  99,  99,  99],
             [99,  99,  99,  99,  99,  99,  99,  99]])

    else:
        raise ValueError(tnum, 'Table number must be 0 or 1')

    h_down = np.divide((2 * h-1), (2 * t))
    h_up   = np.divide((2 * h+1), (2 * t))
    if np.all(h == 1): return 100
    x_down = (h_down[h > 1]).max() 
    x_up   = (h_up[h < th_high]).min() if (h < th_high).any() else None
    if x_up is None:
        s = 1
    elif x_down > 1 and x_up > 1:
        s = np.ceil(50 / x_up)
    elif x_up < 1:
        s = np.ceil(50*(2 - x_up))
    else:
        s = 50
    return s
  
def process1():
    filepath = './data/FacebookDownloadAEs'
    filelists = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
    ori_qf = []
    down_qf = []
    for filename in filelists:
        img_qf = int(jpeg_qtableinv(filepath+"/"+filename))
        in_qf = int(filename.split('_')[4].split("qf")[1].split('.')[0])
        ori_qf.append(in_qf)
        down_qf.append(img_qf)

    ori_meandown_pairs =[]
    for qf in set(ori_qf):
        down_qf_i = np.asarray(down_qf)[np.where(np.asarray(ori_qf)==qf)]
        meanqf_i = np.mean(down_qf_i)
        ori_meandown_pair = [qf,meanqf_i]
        ori_meandown_pairs.append(ori_meandown_pair)

    np.save('./data/ori_meandown_pairs.npy',np.asarray(ori_meandown_pairs))
    np.save('./data/ori_qf.npy',np.asarray(ori_qf))
    np.save('./data/down_qf.npy',np.asarray(down_qf))
