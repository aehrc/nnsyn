#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import SimpleITK
import numpy as np
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util.arraycrop import crop
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter
import torch

class ImageMetrics():
    def __init__(self, debug=False):
        # Use fixed wide dynamic range
        self.dynamic_range = [-1024., 3000.]
        self.debug = debug
    def score_patient(self, gt_img, synthetic_ct, mask):        
        assert gt_img.shape == synthetic_ct.shape 
        if mask is not None:
            assert mask.shape == synthetic_ct.shape 

        # perform masking on the images
        ground_truth = gt_img if mask is None else np.where(mask == 0, -1024, gt_img)
        prediction = synthetic_ct if mask is None else np.where(mask == 0, -1024, synthetic_ct)
        
        # Compute image similarity within the mask
        mae_value = self.mae(ground_truth,
                             prediction,
                             mask)
        
        psnr_value = self.psnr(ground_truth,
                               prediction,
                               mask,
                               use_population_range=True)
        
        ms_ssim_value, ms_ssim_mask_value = self.ms_ssim(ground_truth,
                               prediction, 
                               mask)

        return {
            'mae': mae_value,
            'psnr': psnr_value,
            'ms_ssim': ms_ssim_mask_value,
        }
    
    def mae(self,
            gt: np.ndarray, 
            pred: np.ndarray,
            mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Mean Absolute Error (MAE)
    
        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).
    
        Returns
        -------
        mae : float
            mean absolute error.
    
        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            #binarize mask
            mask = np.where(mask>0, 1., 0.)
            
        mae_value = np.sum(np.abs(gt*mask - pred*mask))/mask.sum() 
        return float(mae_value)
    
    
    def psnr(self,
             gt: np.ndarray, 
             pred: np.ndarray,
             mask: Optional[np.ndarray] = None,
             use_population_range: Optional[bool] = False) -> float:
        """
        Compute Peak Signal to Noise Ratio metric (PSNR)
    
        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).
        use_population_range : bool, optional
            When a predefined population wide dynamic range should be used.
            gt and pred will also be clipped to these values.
    
        Returns
        -------
        psnr : float
            Peak signal to noise ratio..
    
        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            #binarize mask
            mask = np.where(mask>0, 1., 0.)
            
        if use_population_range:            
            # Clip gt and pred to the dynamic range
            gt = np.clip(gt, a_min=self.dynamic_range[0], a_max=self.dynamic_range[1])
            pred = np.clip(pred, a_min=self.dynamic_range[0], a_max=self.dynamic_range[1])
            dynamic_range = self.dynamic_range[1]  - self.dynamic_range[0]
        else:
            dynamic_range = gt.max()-gt.min()
            pred = np.clip(pred, a_min=gt.min(), a_max=gt.max())
            
        # apply mask
        gt = gt[mask==1]
        pred = pred[mask==1]
        psnr_value = peak_signal_noise_ratio(gt, pred, data_range=dynamic_range)
        return float(psnr_value)

    # Compute the luminance, contrast and structure components of the SSIM between two images    
    def structural_similarity_at_scale(self, im1,
        im2,
        *,
        luminance_weight=1,
        contrast_weight=1,
        structure_weight=1,
        win_size=None,
        gradient=False,
        data_range=None,
        channel_axis=None,
        gaussian_weights=False,
        full=False,
        **kwargs,):
            K1 = kwargs.pop('K1', 0.01)
            K2 = kwargs.pop('K2', 0.03)
            sigma = kwargs.pop('sigma', 1.5)
            if K1 < 0:
                raise ValueError("K1 must be positive")
            if K2 < 0:
                raise ValueError("K2 must be positive")
            if sigma < 0:
                raise ValueError("sigma must be positive")
            use_sample_covariance = kwargs.pop('use_sample_covariance', True)
            if gaussian_weights:
                # Set to give an 11-tap filter with the default sigma of 1.5 to match
                # Wang et. al. 2004.
                truncate = 3.5

            if win_size is None:
                if gaussian_weights:
                    # set win_size used by crop to match the filter size
                    r = int(truncate * sigma + 0.5)  # radius as in ndimage
                    win_size = 2 * r + 1
                else:
                    win_size = 7  # backwards compatibility
            if gaussian_weights:
                filter_func = gaussian
                filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': 'reflect'}
            else:
                filter_func = uniform_filter
                filter_args = {'size': win_size}

            ndim = im1.ndim
            NP = win_size**ndim

            # filter has already normalized by NP
            if use_sample_covariance:
                cov_norm = NP / (NP - 1)  # sample covariance
            else:
                cov_norm = 1.0  # population covariance to match Wang et. al. 2004
            # compute (weighted) means
            ux = filter_func(im1, **filter_args)
            uy = filter_func(im2, **filter_args)


            # compute (weighted) variances and covariances
            uxx = filter_func(im1 * im1, **filter_args)
            uyy = filter_func(im2 * im2, **filter_args)
            uxy = filter_func(im1 * im2, **filter_args)
            vx = cov_norm * (uxx - ux * ux)
            vxsqrt = np.clip(vx, a_min=0, a_max=None) ** 0.5 # TODO: this is very ugly
            vy = cov_norm * (uyy - uy * uy)
            vysqrt = np.clip(vy, a_min=0, a_max=None) ** 0.5 # TODO: this is very ugly
            vxy = cov_norm * (uxy - ux * uy)

            R = data_range
            C1 = (K1 * R) ** 2
            C2 = (K2 * R) ** 2
            C3 = C2 / 2

            L = np.clip((2 * ux * uy + C1) / (ux * ux + uy * uy + C1), a_min=0, a_max=None) # TODO is this clipping necessary or do we increase K1 and K2?

            C = np.clip((2 * vxsqrt * vysqrt + C2) / (vx + vy + C2), a_min=0, a_max=None)
            S = np.clip((vxy + C3) / (vxsqrt * vysqrt + C3), a_min=0, a_max=None)

            result = (L ** luminance_weight) * (C ** contrast_weight) * (S ** structure_weight)
            # to avoid edge effects will ignore filter radius strip around edges
            pad = (win_size - 1) // 2

            # compute (weighted) mean of ssim. Use float64 for accuracy.
            mssim = crop(result, pad).mean(dtype=np.float64)

            if full:
                return mssim, result
            return mssim


    # Compute the masked MS-SSIM by masking the SSIM at every resolution level
    def ms_ssim(self, gt: np.ndarray, pred: np.ndarray, mask: Optional[np.ndarray] = None, scale_weights: Optional[np.ndarray] = None) -> float:
        # Clip gt and pred to the dynamic range
        gt = np.clip(gt, min(self.dynamic_range), max(self.dynamic_range))
        pred = np.clip(pred, min(self.dynamic_range), max(self.dynamic_range))

        if mask is not None:
            #binarize mask
            mask = np.where(mask>0, 1., 0.)
            
            # Mask gt and pred
            gt = np.where(mask==0, min(self.dynamic_range), gt)
            pred = np.where(mask==0, min(self.dynamic_range), pred)

        # Make values non-negative
        if min(self.dynamic_range) < 0:
            gt = gt - min(self.dynamic_range)
            pred = pred - min(self.dynamic_range)

        # Set dynamic range for ssim calculation and calculate ssim_map per pixel
        dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]


        # see Eq. 7 in https://live.ece.utexas.edu/publications/2003/zw_asil2003_msssim.pdf
        # Also, the final sentence of section 3.2 (Results)
        scale_weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]) if scale_weights is None else scale_weights
        luminance_weights = np.array([0, 0, 0, 0, 0, 0.1333]) if scale_weights is None else scale_weights 
        levels = len(scale_weights)

        downsample_filter = np.ones((2, 2, 2)) / 8

        gtx, gty, gtz = gt.shape

        # Due to the downsampling in the MS-SSIM, the minimum matrix size must be 97 in every dimension
        target_size = 97

        pad_values = [
            (np.clip((target_size - dim)//2, a_min=0, a_max=None), 
            np.clip(target_size - dim - (target_size - dim)//2, a_min=0, a_max=None)) 
            for dim in [gtx, gty, gtz]]

        gt   = np.pad(gt,   pad_values, mode='edge')
        pred = np.pad(pred, pad_values, mode='edge')
        mask = np.pad(mask, pad_values, mode='edge')

        min_size = (downsample_filter.shape[-1] - 1) * 2 ** (levels - 1) + 1

        ms_ssim_vals, ms_ssim_maps = [], []
        for level in range(levels):
            ssim_value_full, ssim_map = self.structural_similarity_at_scale(gt, pred, 
                luminance_weight=luminance_weights[level], 
                contrast_weight=scale_weights[level],
                structure_weight=scale_weights[level],
                data_range=dynamic_range, full=True)
            pad = 3
            # at every level, we get the ssim_value_full, which is just mean SSIM at a level L, and the 
            # SSIM map. The masked SSIM is the mean SSIM within this mask
            ssim_value_masked  = (crop(ssim_map, pad)[crop(mask, pad).astype(bool)]).mean(dtype=np.float64)

            ms_ssim_vals.append(ssim_value_full)
            ms_ssim_maps.append(ssim_value_masked)

            # The images are cleverly downsampled using an uniform filter
            # the mask is just downsampled by selecting every other line in every dimension
            filtered = [fftconvolve(im, downsample_filter, mode='same')
                for im in [gt, pred]]
            gt, pred, mask = [x[::2, ::2, ::2] for x in [*filtered, mask]]

        ms_ssim_val = np.prod([np.clip(x, a_min=0, a_max=1) for x in ms_ssim_vals])
        ms_ssim_mask_val = np.prod([np.clip(x, a_min=0, a_max=1) for x in ms_ssim_maps])

        return float(ms_ssim_val), float(ms_ssim_mask_val)


# compute image metrics for the predition folders
class ImageMetricsCompute(ImageMetrics):
    def __init__(self):
        super().__init__()
        self.names = ["mae", "psnr", "ms_ssim"]
    
    def init_storage(self, names: list):
        self.storage = dict()
        self.storage_id = []
        self.names = names
        for name in names:
            self.storage[name] = []

    def add(self, res: dict, patient_id=None):
        for key, value in res.items():
            self.storage[key].append(value)
        if patient_id:
            self.storage_id.append(patient_id)

    def aggregate(self):
        res = dict()
        for name in self.names:
            res[name] = dict()

        for key, value in self.storage.items():
            res[key]['mean'] = np.mean(value)
            res[key]['std'] = np.std(value)
            res[key]['max'] = np.max(value)
            res[key]['min'] = np.min(value)
            res[key]['25pc'] = np.percentile(value, 25)
            res[key]['50pc'] = np.percentile(value, 50)
            res[key]['75pc'] = np.percentile(value, 75)
            res[key]['count'] = len(value)
        return res

    def reset(self):
        for key, value in self.storage.items():
            self.storage[key] = []

    def score_array(self, gt_array, pred_array, mask_array=None):
        if torch.is_tensor(gt_array):
            gt_array = gt_array.cpu().numpy().squeeze()
        if torch.is_tensor(pred_array):
            pred_array = pred_array.cpu().numpy().squeeze()
        if torch.is_tensor(mask_array):
            mask_array = mask_array.cpu().numpy().squeeze()

        # Calculate image metrics
        res = dict()
        if self.names and 'mae' in self.names:
            mae_value = self.mae(gt_array,
                                 pred_array,
                                 mask_array)
            res['mae'] = mae_value

        if self.names and 'psnr' in self.names:
            psnr_value = self.psnr(gt_array,
                                   pred_array,
                                   mask_array, use_population_range=True)
            res['psnr'] = psnr_value

        if self.names and 'ms_ssim' in self.names:
            ms_ssim_value, ms_ssim_mask_value = self.ms_ssim(gt_array,
                                   pred_array, 
                                   mask_array)
            res['ms_ssim'] = ms_ssim_mask_value


        return res


if __name__=='__main__':
    metrics = ImageMetrics()
    ground_truth_path = "path/to/ground_truth.mha"
    predicted_path = "path/to/prediction.mha"
    mask_path = "path/to/mask.mha"
    print(metrics.score_patient(ground_truth_path, predicted_path, mask_path))