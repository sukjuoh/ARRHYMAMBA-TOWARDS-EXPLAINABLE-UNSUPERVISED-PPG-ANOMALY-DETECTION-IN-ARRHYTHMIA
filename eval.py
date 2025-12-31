"""
ref: https://github.com/ML4ITS/TimeVQVAE-AnomalyDetection/blob/main/evaluate.py
"""


from argparse import ArgumentParser
import pickle
from multiprocessing import Process
import numpy as np
from exp.set_stage1 import SetStage1
from exp.set_stage2 import SetStage2

from evaluation import load_data
import torch.nn as nn
import torch
import time
from sklearn.metrics import roc_auc_score

from einops import rearrange, repeat
from utils import get_root_dir, load_yaml_param_settings, set_window_size
from evaluation import evaluate_fn, save_final_summarized_figure

import os





def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.", default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_ind', default=[1,], help='e.g., 1 2 3. Indices of datasets to run experiments on.', nargs='+', required=True)
    parser.add_argument('--latent_window_size_rates', default=[0.3, 0.6, 0.9], nargs='+', type=float)
    parser.add_argument('--rolling_window_stride_rate', default=0.1, type=float, help='stride = rolling_window_stride_rate * window_size')
    parser.add_argument('--q', default=0.99, type=float)
    parser.add_argument('--explainable_sampling', default=False, help='Note that this script will run more slowly with this option being True.')
    parser.add_argument('--n_explainable_samples', type=int, default=2, help='how many explainable samples to get per window.')
    parser.add_argument('--max_masking_rate_for_explainable_sampling', type=float, default=0.9, help='it prevents complete masking and ensures the minimum valid tokens to leave a minimum context for explainable sampling.')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--n_workers', default=4, type=int, help='multi-processing for latent_window_size_rate.')

    return parser.parse_args()


def process_list_arg(arg, dtype):
    arg = np.array(arg, dtype=dtype)
    return arg

def process_bool_arg(arg):
    if str(arg) == 'True':
        arg = True
    elif str(arg) == 'False':
        arg = False
    else:
        raise ValueError
    return arg
def compute_auc(TP_indices, FP_indices, FN_indices, length):
   

    # Initialize binary labels
    y_true = np.zeros(length)
    y_pred = np.zeros(length)

    # Assign values based on TP, FP, FN, TN
    y_true[TP_indices] = 1
    y_true[FN_indices] = 1  

    y_pred[TP_indices] = 1
    y_pred[FP_indices] = 1  

    try:
        auc_score = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc_score = 0  

    return auc_score


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)
    args.dataset_ind = process_list_arg(args.dataset_ind, int)
    args.latent_window_size_rates = process_list_arg(args.latent_window_size_rates, float)
    args.explainable_sampling = process_bool_arg(args.explainable_sampling)


    acc = []
    f1 = []
    auc = []
    recalls = []
    specipicitys = []
    for idx in args.dataset_ind:
        print(f'\nidx: {idx}')
        idx = int(idx)

        start = time.perf_counter()

        for worker_idx in range(len(args.latent_window_size_rates)):
            latent_window_size_rates = args.latent_window_size_rates[worker_idx*args.n_workers: (worker_idx+1)*args.n_workers]
            if len(latent_window_size_rates) == 0:
                break

            procs = []
            for wsr in latent_window_size_rates:
                proc = Process(target=evaluate_fn, args=(config, args, idx, wsr, args.rolling_window_stride_rate, args.q, args.device))  # make sure to put , (comma) at the end
                procs.append(proc)
                proc.start()
            for p in procs:
                p.join()  # make each process wait until all the other process ends.
        
        end = time.perf_counter()

        print(print(f"run time: {end - start:.6f}sec"))
       
        # integrate all the joint anomaly scores across `latent_window_size_rates`
        a_s_star = 0.
        joint_threshold = 0.
        for wsr in args.latent_window_size_rates:
            result_fname = get_root_dir().joinpath('evaluation', 'results', f'{idx}-anomaly_score-latent_window_size_rate_{wsr}.pkl')
            with open(str(result_fname), 'rb') as f:
                result = pickle.load(f)
                a_t = result['a_T']  # (n_freq, ts_len')
                a_T += a_t  # (n_freq, ts_len')
                joint_threshold += result['anom_threshold']

        # \bar{a}
        a_bar = a_T[0]*0.6 + a_T[1]*0.3 + a_T[2]*0.1  # (ts_len',)


        # \doublebar{a}_s^star
        data = np.hstack(load_data(idx, config, 'train'))
        window_size = set_window_size(data) *config['dataset']['n_periods']
        
        a_bar_max = np.zeros_like(a_bar)  # (ts_len',)
        for j in range(len(a_bar)):
            rng = slice(max(0, j - window_size // 2), j + window_size // 2)
            a_bar_max[j] = np.max(a_bar[rng])
        
        
        
        # a_final
        a_final = (a_bar + a_bar_max)/2


        # final threshold
        final_threshold = joint_threshold[0]*0.6 + joint_threshold[1]*0.3 + joint_threshold[2]*0.1
        #final_threshold = (joint_threshold[0] + joint_threshold[1] + joint_threshold[2])/3
        anom_ind = a_final > final_threshold

        # plot
        save_final_summarized_figure(idx, result['X_test_unscaled'], result['Y'], result['timestep_rng_test'],
                                     a_T, a_bar, a_bar_max, a_final,
                                     joint_threshold, final_threshold, anom_ind, window_size, config, args)
        



        # save: resulting data
        joint_resulting_data = {'dataset_index': idx,
                                'X_test_unscaled': result['X_test_unscaled'],  # time series
                                'Y': result['Y'],  # label

                                'a_T': a_T,  # (n_freq, ts_len')
                                'bar{a}_T': a_bar,  # (ts_len',)
                                'bar{a}_max': a_bar_max,  # (ts_len',)
                                'a_final': a_final,  # (ts_len',)

                                'joint_threshold': joint_threshold,  # (n_freq,)
                                'final_threshold': final_threshold  # (,)
                                }
        
        # Compute Accuracy with 10-point tolerance

        true_anomalies = np.where(result['Y'] > 0)[0]
        pred_anomalies = np.where(a_final > final_threshold)[0]

        tolerance = 10 
        TP = 0
        for pred_idx in pred_anomalies:
            if np.any(np.abs(true_anomalies - pred_idx) <= tolerance):  
                TP += 1

        FP = len(pred_anomalies) - TP  
        
        FN = 0
        for true_idx in true_anomalies:
            if not np.any(np.abs(pred_anomalies - true_idx) <= tolerance):  
                FN += 1
        
        total_points = len(result['Y'])
        
        TN = total_points - (TP + FP + FN)  
        
        sensitivity = (TP) / (TP + FN) if (TP + FN) > 0 else 0 
        recall = sensitivity
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        TP_indices = [pred_idx for pred_idx in pred_anomalies if np.any(np.abs(true_anomalies - pred_idx) <= tolerance)]
        FP_indices = [pred_idx for pred_idx in pred_anomalies if not np.any(np.abs(true_anomalies - pred_idx) <= tolerance)]
        FN_indices = [true_idx for true_idx in true_anomalies if not np.any(np.abs(pred_anomalies - true_idx) <= tolerance)]
        auc_score = compute_auc(TP_indices, FP_indices, FN_indices, total_points)
        
        acc.append(accuracy)
        f1.append(f1_score)
        auc.append(auc_score)
        recalls.append(recall)
        specipicitys.append(specificity)
  
        
        print(f"accuracy: {accuracy:.3f} | sensitivity: {sensitivity:.3f}")

        # Save Accuracy in Results
        joint_resulting_data['accuracy'] = accuracy
        joint_resulting_data['sensitivity'] = sensitivity

         
        if args.explainable_sampling:
            # load model
            data = np.hstack(load_data(idx, config, 'train'))
            window_size = set_window_size(data) *config['dataset']['n_periods']
            
            stage2 = SetStage2.load_from_checkpoint(os.path.join('saved_models', 'stage2_arrhymamba.ckpt'), 
                                                config=config, 
                                                map_location=f'cuda:{args.device}', strict=False)            
                        

        
            model = stage2.model
            model.decoder.interp = nn.Upsample(size=window_size, mode='linear')
            model.eval()


            counter_factual_data = []
            timestep_rng_test = result['timestep_rng_test']
            for timestep_idx, timestep in enumerate(timestep_rng_test):
                
                window_rng = slice(timestep, timestep + window_size)
                x_unscaled = result['X_test_unscaled'][window_rng]  
                
               
                mu = np.nanmean(x_unscaled, axis=-1, keepdims=True)  
                sigma = np.nanstd(x_unscaled, axis=-1, keepdims=True) 
                min_std = 1.e-4  
                sigma = np.clip(sigma, min_std, None)
                x_normalized = (x_unscaled - mu) / sigma  
                
                
                z_q, s = model.encode_to_z_q(
                    torch.from_numpy(x_normalized[None, None, :]).to(args.device),
                    model.encoder,
                    model.vq_model
                )
                latent_height, latent_width = z_q.shape[2], z_q.shape[3]
                
                anom_window = a_final[window_rng]  # (ts_len',)
                anom_window = torch.from_numpy(anom_window)[None, None, :]
                anom_window = torch.nn.functional.interpolate(anom_window, size=(latent_width,))[0, 0]
                
                is_anom = anom_window > final_threshold
                
                if is_anom.float().mean().item() > args.max_masking_rate_for_explainable_sampling:
                    tau = torch.quantile(anom_window, 1 - args.max_masking_rate_for_explainable_sampling).item()
                    is_anom = anom_window > tau

                if is_anom.float().mean().item() > 0.:
                    s_star = rearrange(s, '1 (h w) -> 1 h w', h=latent_height)  
                    s_star[:, :, is_anom] = model.mask_token_id
                    s_star = rearrange(s_star, '1 h w -> 1 (h w)')  
                    s_star = repeat(s_star, '1 n -> b n', b=args.n_explainable_samples)

                    masking_ratio = is_anom.int().sum() / len(is_anom)
                    t_star = int(np.floor(2 * np.arccos(masking_ratio) / np.pi * config['ArrhyMamba']['T']))
                    s_Tstar = model.explainable_sampling(t_star, s_star)
                   
                    with torch.no_grad():
                        xhat = model.decode_token_ind_to_timeseries(s_Tstar).cpu().numpy()
                    
                    xhat = (xhat * sigma) + mu  
                    counter_factual_data.append((timestep, xhat))  
         
            joint_resulting_data['counter_factual'] = counter_factual_data
        joint_resulting_data['window_size'] = window_size
    

        saving_fname = get_root_dir().joinpath('evaluation', 'results', f'{idx}-joint_anomaly_score.pkl')
        with open(saving_fname, 'wb') as f:
            pickle.dump(joint_resulting_data, f, pickle.HIGHEST_PROTOCOL)
    
    print(f'acc:{np.mean(acc)}')
    print(f'recall:{np.mean(recalls)}')
    print(f'f1:{np.mean(f1)}')
    print(f'auc:{np.mean(auc)}')
    print(f'specificity:{np.mean(specipicitys)}')
