import tensorflow as tf
from network import CGAN
from handler import DataHandler, DataSplitter
import matplotlib.pyplot as plt
import numpy as np


import os
import sys
import argparse
from datetime import datetime
import time
import csv

from tensorflow.python.client import device_lib

fat_HU = np.array([-140, -30])
muscle_HU = np.array([-29, 150])

noise_mean = 0
noise_std = 20
noise_prop = 0.9

def normalization_function(ct):
    ct = ct.astype("float32")
    fat_mean = (fat_HU[1] - fat_HU[0])/2.0
    ct -= (fat_HU[1] - fat_mean)
    ct /= fat_mean
    return ct

def pproc_noise(ct, pet, mask):
    if np.random.random() <= noise_prop:
        noise_map = np.random.normal(noise_mean, np.random.random()*noise_std, ct.shape)
        ct += noise_map
    return ct, pet, mask

def pproc_random_shift(ct, pet, mask):
    for i in range(ct.shape[0]):
        sx = np.random.randint(-5, 5, 1)[0]
        sy = np.random.randint(-5, 5, 1)[0]
        ct[i, :, :, :] = np.roll(ct[i, :, :, :], (sx*4, sy*4), axis=(0, 1))
        pet[i, :, :, :] = np.roll(pet[i, :, :, :], (sx, sy), axis=(0, 1))
        mask[i, :, :, :] = np.roll(mask[i, :, :, :], (sx, sy), axis=(0, 1))
    return ct, pet, mask


def main(input_h5, output_path, context_levels, stride, sufix, gpu, train_folds, valid_folds, test_folds, fat_masked, nb_elements, fold, lr, sorted_idxs):
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu

    # Datasets loading
    # ----------------
#     data_splitter = DataSplitter(train_folds=train_folds,
#                                  valid_folds=valid_folds,
#                                  test_folds=test_folds,
#                                  h5_path=input_h5,
#                                  ct_key="ct", pet_key="suv",
#                                  nb_elements=nb_elements,
#                                  shuffle_training=True,
#                                  context_levels=context_levels,
#                                  stride=stride,
#                                  th_datarange=[fat_HU, muscle_HU],
# #                                  pproc_func={"train": pproc_function, "validation": None, "test": None},
#                                  normalization_function=normalization_function)
    
    h5_list = input_h5.split(',')
    data_splitter_list = []
    for p in h5_list:
        data_splitter_list.append(DataSplitter(train_folds=train_folds,
                                               valid_folds=valid_folds,
                                               test_folds=test_folds,
                                               h5_path=p,
                                               ct_key="ct", pet_key="suv",
                                               nb_elements=nb_elements,
                                               shuffle_training=True,
                                               context_levels=context_levels,
                                               stride=stride,
                                               th_datarange=fat_HU,
                                               pproc_func={"train": pproc_random_shift, "validation": None, "test": None},
                                               normalization_function=normalization_function,
                                               sorted_idxs=sorted_idxs))
        
    
    # Training
    # --------
    total_folds = train_folds + valid_folds + test_folds
    EPOCHS = 200
    gen_lr = lr
    dis_lr = lr
    fold_list = range(total_folds) if fold is None else [fold]
    for i in fold_list:
#         train_dh, valid_dh, test_dh = data_splitter.get_fold_handlers(i)
#         data_splitter_list[0].set_testing_mode(True)
        train_dh, valid_dh, test_dh = data_splitter_list[0].get_fold_handlers(i)
        for dsi in range(1, len(data_splitter_list)):
#             data_splitter_list[dsi].set_testing_mode(True)
            trn, vld, tst = data_splitter_list[dsi].get_fold_handlers(i)
            train_dh.merge(trn)
            del trn
            trn = None
            valid_dh.merge(vld)
            del vld
            vld = None
            test_dh.merge(tst)
            del tst
            tst = None
        
        fold_path = os.path.join(output_path + "_%s" % sufix, "fold_%s/" % i)
        os.makedirs(fold_path, exist_ok=True)
        
        with open(os.path.join(fold_path, 'train_scanid.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(sorted(train_dh.scan_id))
            
        with open(os.path.join(fold_path, 'valid_scanid.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(sorted(valid_dh.scan_id))
            
        with open(os.path.join(fold_path, 'test_scanid.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(sorted(test_dh.scan_id))
        
        cGAN = CGAN(output_path=fold_path,
                    gen_lr=gen_lr, dis_lr=dis_lr,
                    kernel_size=4, strides=(2, 2), n_slices=context_levels*2+1, fat_masked=fat_masked)

        cGAN.fit(train_dh, EPOCHS, valid_dh)

#         if test_dh is not None:
#             cGAN.eval_network(test_dh, output_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PET/CT training script.')
    parser.add_argument('--input_h5', type=str, help='Input h5 file with datasets.', required=True)
    parser.add_argument('--output_path', type=str, help='Path were logs and checkpoints will be saved.', required=True)
    parser.add_argument('--context_levels', type=int, help='Number of slice levels used as input CT data.', required=False)
    parser.add_argument('--stride', type=int, help='Distance between two data chucks for the same CT (overlayer).',  required=False)
    parser.add_argument('--sufix', type=str, help='Sufix of the output directory.', required=False)
    parser.add_argument('--gpu', type=str, help='GPU number to use.', required=False)
    parser.add_argument('--train_folds', type=int, help='Number of folds for training', required=False)
    parser.add_argument('--valid_folds', type=int, help='Number of folds to use as validation.', required=False)
    parser.add_argument('--test_folds', type=int, help='Number of folds to use as test.', required=False)
    parser.add_argument('--fat_masked', help='Training data will be masked in the loss computation using fat tissue only.', action='store_true')
    parser.add_argument('--nb_elements', type=int, help='Number of input date to be processed in each training batch.', required=False)
    parser.add_argument('--fold', type=int, help='Fold to be processed, if not defined all folds will be computed sequentially.', required=False)
    parser.add_argument('--lr', type=float, help='Learning rate.', required=False)
    parser.add_argument('--sorted', help='Define if the cases indexes will be sorted by PET activation mean previous to split them into folds. Doing this, the distribution of PET activation mean will be more homogeneous across the different sub-folds.', action='store_true')
    parser.add_argument('--histEq', help='Performs a histogram Equalization previous loss computation.', action='store_true')

    parser.set_defaults(context_levels=1)
    parser.set_defaults(stride=1)
    parser.set_defaults(sufix=datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S"))
    parser.set_defaults(gpu="")
    parser.set_defaults(train_folds=5)
    parser.set_defaults(valid_folds=1)
    parser.set_defaults(test_folds=1)
    parser.set_defaults(nb_elements=1)
    parser.set_defaults(fold=None)
    parser.set_defaults(lr=2e-5)
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    print(device_lib.list_local_devices())

    print("input_h5 = %s" % args.input_h5)
    print("output_path = %s" % args.output_path)
    print("context_levels = %s" % args.context_levels)
    print("stride = %s" % args.stride)
    print("sufix = %s" % args.sufix)
    print("gpu = %s" % args.gpu)
    print("train_folds = %s" % args.train_folds)
    print("valid_folds = %s" % args.valid_folds)
    print("test_folds = %s" % args.test_folds)
    print("fat_masked = %s" % args.fat_masked)
    print("nb_elements = %s" % args.nb_elements)
    print("fold = %s" % "all" if args.fold is None else args.fold)
    print("lr = %s" % args.lr)
    print("sorted = %s" % args.sorted)
    print("histEq = %s" % args.histEq)
    
    main(args.input_h5, args.output_path, args.context_levels, args.stride, args.sufix, args.gpu, args.train_folds, args.valid_folds, args.test_folds, args.fat_masked, args.nb_elements, args.fold, args.lr, args.sorted)