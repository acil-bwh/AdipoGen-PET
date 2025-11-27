import h5py
import numpy as np
from numpy.core.defchararray import array
from scipy.ndimage import zoom
from tqdm import tqdm
import cv2
import time

def resample_mask(src, dst):
        res = np.zeros_like(dst)
        for i in range(res.shape[-1]):
            res[...,i] = cv2.resize(src[...,i].astype("uint8"), dsize=res.shape[:-1], interpolation=cv2.INTER_NEAREST)
        return res.astype("bool")


def threshold_data(ct, pet, th):
    th = np.logical_and(ct >= th[0], ct <= th[1]).astype("int")  
    return resample_mask(th, pet)

class DataHandler:
    def __init__(self, h5_path, ct_key, pet_key, scan_id_list=None, idxs_list=None, nb_elements=1, shuffle=False, context_levels=1, stride=1, pproc_func=None, normalization_function=None, th_datarange=[]):
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        self.scan_id_list = scan_id_list
        self.idxs_list = np.arange(0, self.h5_file[ct_key].shape[0]) if idxs_list is None else idxs_list
        self.nb_elements = nb_elements
        self.shuffle = shuffle
        self.stride = stride
        self.pproc_func = pproc_func
        self.normalization_function = normalization_function
        self.th_datarange = th_datarange

        # Map normalized scan IDs to their dataset index
        self.all_ids = {self._normalize_scan_id(v): i for i, v in enumerate(self.h5_file["id"][:])}

        if self.scan_id_list is not None:
            norm_ids = [self._normalize_scan_id(sid) for sid in self.scan_id_list]
            missing = [sid for sid in norm_ids if sid not in self.all_ids]
            if missing:
                raise ValueError(f"Scan IDs not found in file: {missing}")
            self.idxs_list = np.array([self.all_ids[sid] for sid in norm_ids], dtype=int)
                        
        # Memory load:
        # ------------
        self.ct_ds = []
        self.pet_ds = []
        self.mask_ds = []

        self.slices_ds = np.zeros((len(self.idxs_list), self.h5_file["slices"].shape[-1]), dtype="int16")
        self.scan_id = []
        self.margin = int(self.h5_file.attrs["margin"])

        assert context_levels <= self.margin, "ERROR: the maximum number of context layers levels for this dataset is %s" % self.margin
        self.context_levels = context_levels
        self.n_slices = self.context_levels * 2 + 1
        
        self.layer_idxs = []
        self.idxs_bk = np.empty((0,2), dtype="int")
        
        for i, n in enumerate(tqdm(self.idxs_list)):
            self.slices_ds[i] = np.array(self.h5_file["slices"][n])
            self.scan_id.append(self._normalize_scan_id(self.h5_file["id"][n]))
            
            self.ct_ds.append(np.array(self.h5_file[ct_key][n,:,:,:self.slices_ds[i,2],:]))
            self.pet_ds.append(np.array(self.h5_file[pet_key][n,:,:,:self.slices_ds[i,2],:]))
                        
            mask_shape = tuple((self.h5_file[pet_key].shape[1], self.h5_file[pet_key].shape[2], self.slices_ds[i,2], 2))
            mask = np.zeros(mask_shape, dtype="bool")
            
            mask[...,0] = np.array(self.h5_file["bodymask"][n,:,:,:self.slices_ds[i,2],0])
            mask[...,1] = self.__threshold_data(self.ct_ds[i][:,:,:,0], self.pet_ds[i][:,:,:,0], self.th_datarange) * mask[:,:,:,0]
            
            self.mask_ds.append(mask)
            
            # ---- Index list creation ----
            lidxs = np.arange(self.slices_ds[i,0], self.slices_ds[i,2]-self.margin-1, self.stride)
            if not self.slices_ds[i,2]-self.margin-1 in lidxs:
                lidxs = np.append(lidxs, self.slices_ds[i,2]-self.margin-1)

            self.layer_idxs.append(lidxs)
            
            d = np.ones(lidxs.shape[0], dtype="int") * i
            self.idxs_bk = np.concatenate((self.idxs_bk, np.stack((d, lidxs), axis=-1)))
            # ---- End of Index list creation ----
        
        self.idxs = None
        self.epoch_gen()
        
        self.ct_cached = np.ones(((self.nb_elements,) + tuple(self.h5_file[ct_key].shape[1:-2]) + (self.n_slices,)), dtype="float32")
        self.pet_cached = np.zeros(((self.nb_elements,) + tuple(self.h5_file[pet_key].shape[1:-2]) + (1,)), dtype="float32")
        self.mask_cached = np.zeros(((self.nb_elements,) + tuple(self.h5_file[pet_key].shape[1:-2]) + (2,)), dtype="bool")
    

    def epoch_gen(self):
        self.idxs = np.array(self.idxs_bk)
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.idxs = self.idxs.tolist()
        
        
    def merge(self, dh, prop=1.0):
        self.ct_ds += dh.ct_ds
        self.pet_ds += dh.pet_ds
        self.mask_ds += dh.mask_ds
        
        self.slices_ds = np.concatenate((self.slices_ds, dh.slices_ds), axis=0)
        self.scan_id += dh.scan_id
        self.layer_idxs += dh.layer_idxs
        dh_idxs = np.array(dh.idxs_bk)
        dh_idxs[:,0] += len(self.idxs_list)
        self.idxs_bk = np.concatenate((self.idxs_bk, dh_idxs), axis=0)
                
        self.epoch_gen()

    
    def getIndexfromID(self, scan_id):
        norm_id = self._normalize_scan_id(scan_id)
        return self.scan_id.index(norm_id)


    def _normalize_scan_id(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return ""
            value = value[0]
        if isinstance(value, bytes):
            return value.decode()
        if value is None:
            return ""
        return str(value)
    
    
    def __resample_mask(self, src, dst):
        res = np.zeros_like(dst)
        for i in range(res.shape[-1]):
            res[...,i] = cv2.resize(src[...,i].astype("uint8"), dsize=res.shape[:-1], interpolation=cv2.INTER_NEAREST)
        return res.astype("bool")


    def __threshold_data(self, ct, pet, th):
        th = np.logical_and(ct >= th[0], ct <= th[1]).astype("int")  
        return self.__resample_mask(th, pet)
    

    def GetSize(self):
        return self.idxs_bk.shape[0]
    

    def GetTotalBatches(self):
        return np.ceil(self.idxs_bk.shape[0]/float(self.nb_elements)).astype("int")
    

    def GetTotalSamples(self):
        return len(self.ct_ds)
    

    def GetDataSlicedByIndex(self, ix, normalized=False, preprocess=False):
        lidxs = self.layer_idxs[ix]
        ct = np.zeros(((len(lidxs),) + tuple(self.ct_ds[ix].shape[0:2]) + (self.n_slices,)), dtype="float32")
        pet = np.zeros(((len(lidxs),) + tuple(self.pet_ds[ix].shape[0:2]) + (1,)), dtype="float32")
        mask = np.zeros(((len(lidxs),) + tuple(self.mask_ds[ix].shape[0:2]) + (2,)), dtype="bool")
        
        
        for n, sx in enumerate(lidxs):
            ct[n, :, :, :] = self.ct_ds[ix][:, :, sx-self.context_levels:sx+self.context_levels+1, 0]
            pet[n, :, :, 0] = self.pet_ds[ix][:, :, sx, 0]
            mask[n, :, :, :] = self.mask_ds[ix][:, :, sx, :]
            
        if self.pproc_func is not None and preprocess:
            ct, pet, mask = self.pproc_func(ct, pet, mask)
        
        if self.normalization_function is not None and normalized:
            ct = self.normalization_function(ct)
            
        return ct, pet, mask, self.slices_ds[ix]-np.array([1,1,2])*self.context_levels, self.scan_id[ix]
    
    def GetDataSlicedById(self, scan_id, normalized=False, preprocess=False):
        ix = np.where(np.array(self.scan_id) == scan_id)[0][0]
        return self.GetDataSlicedByIndex(ix, normalized, preprocess)

    def take(self, samples):
        num_samples = len(samples)
        if num_samples != self.nb_elements:
            ct = np.zeros(((num_samples,) + tuple(self.ct_ds[samples[0][0]].shape[0:2]) + (self.n_slices,)), dtype="float32")
            pet = np.zeros(((num_samples,) + tuple(self.pet_ds[samples[0][0]].shape[0:2]) + (1,)), dtype="float32")
            mask = np.zeros(((num_samples,) + tuple(self.mask_ds[samples[0][0]].shape[0:2]) + (2,)), dtype="bool")
        else:
            ct = self.ct_cached
            pet = self.pet_cached
            mask = self.mask_cached

        for n, ix in enumerate(samples):
            ct[n, :, :, :] = self.ct_ds[ix[0]][:, :, ix[1]-self.context_levels:ix[1]+self.context_levels+1, 0]
            pet[n, :, :, 0] = self.pet_ds[ix[0]][:, :, ix[1], 0]
            mask[n, :, :, :] = self.mask_ds[ix[0]][:, :, ix[1], :]
            
        if self.pproc_func is not None:
            ct, pet, mask = self.pproc_func(ct, pet, mask)
            
        if self.normalization_function is not None:
            ct = self.normalization_function(ct)
            
        return ct, pet, mask
    
    
    def get_batch(self):
        index = []
        if self.nb_elements <= len(self.idxs):
            for n in range(self.nb_elements):
                index.append(self.idxs.pop())
            if len(self.idxs) == 0:
                self.epoch_gen()
        else:
            index = np.array(self.idxs)
            self.epoch_gen()

        ct, pet, mask = self.take(index)
                
        return ct, pet, mask



class DataSplitter:
    def __init__(self, train_folds, valid_folds, test_folds, h5_path, ct_key, pet_key, nb_elements=1, shuffle_training=False, context_levels=1, stride=1, pproc_func={"train": None, "validation": None, "test": None}, normalization_function=None, th_datarange=[], sorted_idxs=False):
        self.train_folds = train_folds
        self.valid_folds = valid_folds
        self.test_folds = test_folds
        self.total_folds = self.train_folds + self.valid_folds + self.test_folds

        self.h5_path = h5_path
        self.ct_key = ct_key
        self.pet_key = pet_key
        self.nb_elements=nb_elements
        self.shuffle_training=shuffle_training
        self.context_levels=context_levels
        self.stride=stride
        self.pproc_func=pproc_func
        self.normalization_function = normalization_function
        self.th_datarange=th_datarange
        self.testing_mode=False
        self.n = 5
        self.sorted_idxs = sorted_idxs

        h5_file = h5py.File(h5_path, 'r')
        self.folds_idxs = self.__gen_folds_idxs(h5_file[ct_key].shape[0], self.total_folds)
        
        if self.sorted_idxs:
            pet_means = []
            for i in range(h5_file[self.pet_key].shape[0]):
                p = h5_file[self.pet_key][i,...,0]
                c = h5_file[self.ct_key][i,...,0]
                m = threshold_data(c, p, self.th_datarange)
                pm = p[m]
                pet_means.append(np.mean(pm))
            sorted_pet_means_idxs = np.argsort(pet_means)
            self.folds_idxs = sorted_pet_means_idxs[np.array(self.folds_idxs)]
        
        h5_file.close()
        
    def __gen_folds_idxs(self, n, g):
        d = np.floor_divide(n,g)*g
        folds_idxs = np.array_split(np.arange(0, d), n/g)
        folds_idxs = list(np.stack(folds_idxs, axis=-1))
        
        for i, v in enumerate(np.arange(d, n)):
            folds_idxs[i] = np.concatenate([folds_idxs[i], [v]])
            
        return folds_idxs
        
    def set_testing_mode(self, mode=True, n=5):
        self.testing_mode = mode
        self.n = n

    def get_fold_handlers(self, fold_n, get_train=True, get_valid=True, get_test=True):
        test_idxs = None
        if self.test_folds > 0:
            first_ix = fold_n*self.test_folds % self.total_folds
            test_idxs = self.folds_idxs[first_ix]
            for i in range(first_ix+1, first_ix + self.test_folds):
                test_idxs = np.concatenate([test_idxs, self.folds_idxs[i % self.total_folds]])

        train_idxs = None
        if self.train_folds > 0:
            first_ix = (fold_n*self.test_folds + self.test_folds) % self.total_folds
            train_idxs = self.folds_idxs[first_ix]
            for i in range(first_ix + 1, first_ix + self.train_folds):
                train_idxs = np.concatenate([train_idxs, self.folds_idxs[i % self.total_folds]])

        valid_idxs = None
        if self.valid_folds > 0:
            first_ix = (first_ix + self.train_folds) % self.total_folds
            valid_idxs = self.folds_idxs[first_ix]
            for i in range(first_ix + 1, first_ix + self.valid_folds):
                valid_idxs = np.concatenate([valid_idxs, self.folds_idxs[i % self.total_folds]])

        train_handler = None
        valid_handler = None
        test_handler = None
            
        if get_train and train_idxs is not None:
            train_handler = DataHandler(h5_path=self.h5_path,
                                        ct_key=self.ct_key,
                                        pet_key=self.pet_key,
                                        idxs_list=train_idxs if not self.testing_mode else train_idxs[:self.n],
                                        nb_elements=self.nb_elements,
                                        shuffle=self.shuffle_training,
                                        context_levels=5,
                                        stride=self.stride,
                                        pproc_func=self.pproc_func["train"],
                                        normalization_function=self.normalization_function,
                                        th_datarange=self.th_datarange)

        if get_valid and valid_idxs is not None:
            valid_handler = DataHandler(h5_path=self.h5_path,
                                        ct_key=self.ct_key,
                                        pet_key=self.pet_key,
                                        idxs_list=valid_idxs if not self.testing_mode else valid_idxs[:self.n],
                                        nb_elements=self.nb_elements,
                                        shuffle=False,
                                        context_levels=5,
                                        stride=self.stride,
                                        pproc_func=self.pproc_func["validation"],
                                        normalization_function=self.normalization_function,
                                        th_datarange=self.th_datarange)

        if get_test and test_idxs is not None:
            test_handler = DataHandler(h5_path=self.h5_path,
                                        ct_key=self.ct_key,
                                        pet_key=self.pet_key,
                                        idxs_list=test_idxs if not self.testing_mode else test_idxs[:self.n],
                                        nb_elements=self.nb_elements,
                                        shuffle=False,
                                        context_levels=5,
                                        stride=self.stride,
                                        pproc_func=self.pproc_func["test"],
                                        normalization_function=self.normalization_function,
                                        th_datarange=self.th_datarange)

        return train_handler, valid_handler, test_handler
    



