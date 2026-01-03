"""
ref: https://github.com/ML4ITS/TimeVQVAE-AnomalyDetection/blob/main/evaluation/__init__.py
"""

"""
compute the accuracy, as suggested in [1]

[1] Wu, Renjie, and Eamonn Keogh. "Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress." IEEE Transactions on Knowledge and Data Engineering (2021).
"""
import os
from argparse import ArgumentParser
import copy
import pickle
import gc

import pandas as pd


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import torch.nn.functional as F



from exp.set_stage2 import SetStage2
from sklearn.metrics import roc_auc_score

from preprocessing.load_data import scale
from utils import get_root_dir, set_window_size
from preprocessing.load_data import PPG_TestSequence, PPGTestDataset
from models.stage2.ArrhyMamba import ArrhyMamba
from preprocessing.serve_data import build_data_pipeline


def get_state_dict(state_dict):
    return {k: v for k, v in state_dict.items() if not any(substr in k for substr in ["encoder", "decoder", "stage1", "vq_model"])}


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



def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    return parser.parse_args([])


def load_data(dataset_idx, config, kind: str):
    """used during evaluation"""
    assert kind in ['train', 'test']

    dataset_importer = PPG_TestSequence.create_by_id(dataset_idx)
    n_period = config['dataset']['n_periods']
    interval = config['dataset']['window_size']
    dataset = PPGTestDataset(kind, dataset_importer, n_period, interval)

    X = dataset.X  # (ts_len, 1)
    X = rearrange(X, 'l c -> c l')  # (1, ts_len)

    if kind == 'train':
        return X
    elif kind == 'test':
        Y = dataset.Y  # (ts_len,)
        return X, Y


def mask_prediction(s, height, slice_rng, model:ArrhyMamba):
    # mask
    s_m = copy.deepcopy(s)  # (1 n)
    s_m = rearrange(s_m, '1 (h w) -> 1 h w', h=height)  # (1 h w)
    s_m[:, :, slice_rng] = model.mask_token_id

    # mask-prediction
    logits = model.transformer(rearrange(s_m, 'b h w -> b (h w)'))  # (1 n K)
    #logits = logits*0.9 + hidden_states*0.1
    logits = rearrange(logits, '1 (h w) K -> 1 h w K', h=height)  # (1 h w K)
    logits_prob = F.softmax(logits, dim=-1)  # (1 h w K)

    return logits, logits_prob



@torch.no_grad()
def detect(args,
           X_unscaled,
           maskgit:ArrhyMamba,
           window_size: int,
           rolling_window_stride: int,
           latent_window_size: int,
           compute_reconstructed_X: bool,
           device: int,
           ):
    """
    :param X_unscaled: (1, ts_len)
    :param maskgit:
    :param window_size:
    :param rolling_window_stride:
    :param latent_window_size:
    :param compute_reconstructed_X:
    :return:
    """
    assert latent_window_size % 2 == 1  # latent_window_size must be an odd number.

    ts_len = X_unscaled.shape[1]
    n_channels = X_unscaled.shape[0]
    end_time_step = (ts_len - 1) - window_size
    timestep_rng = range(0, end_time_step, rolling_window_stride)
    logs = {'a_T': np.zeros((n_channels, maskgit.H_prime, ts_len)),
            'reconsX': np.zeros((n_channels, ts_len)),
            'count': np.zeros((n_channels, ts_len)),
            'last_window_rng': None,
            }
    for timestep_idx, timestep in enumerate(timestep_rng):
        if timestep_idx % int(0.3 * len(timestep_rng)) == 0:
            print(f'timestep/total_time_steps*100: {timestep}/{end_time_step} | {round(timestep / end_time_step * 100, 2)} [%]')

        # fetch a window at each timestep
        window_rng = slice(timestep, timestep+window_size)
        x_unscaled = X_unscaled[None, :, window_rng]  # (1, 1, window_size)
        x, (mu, sigma) = scale(x_unscaled, return_scale_params=True)

        # encode
        z_q, s = maskgit.encode_to_z_q(x.to(device), maskgit.encoder, maskgit.vq_model)  # s: (1 n)
        latent_height, latent_width = z_q.shape[2], z_q.shape[3]

        # compute the anomaly scores (negative log likelihood, notated as nll)
        a_tilde = np.zeros((n_channels, latent_height, latent_width),)  # (1 h w)
       
        for w in range(latent_width):
            kernel_rng = slice(w, w+1) if latent_window_size == 1 else slice(max(0, w - (latent_window_size - 1) // 2), w + (latent_window_size - 1) // 2 + 1)
            
            if w < 5:
                kernel_rng = slice(0, w+5) if latent_window_size == 1 else slice(max(0, w - (latent_window_size - 1) // 2), w + (latent_window_size - 1) // 2 + 1)
            else:
                kernel_rng = slice(w-5, w+5) if latent_window_size == 1 else slice(max(0, w - (latent_window_size - 1) // 2), w + (latent_window_size - 1) // 2 + 1)
            
            #kernel_rng = slice(w, w+1) if latent_window_size == 1 else slice(max(0, w - (latent_window_size - 1) // 2), w + (latent_window_size - 1) // 2 + 1)

            # mask-prediction
            logits, logits_prob = mask_prediction(s, latent_height, kernel_rng, maskgit)  # (1 h w K)

            # prior-based anomaly score
            s_rearranged = rearrange(s, '1 (h w) -> 1 h w', h=latent_height)
            
            
            p = torch.gather(logits_prob, -1, s_rearranged[:, :, :, None])  # (1 h w 1)
            p = p[:, :, kernel_rng, 0]  # (1 h r)
            a_w = -1 * torch.log(p + 1e-30).mean(dim=-1).detach().cpu().numpy()  # (1 h); 1e-30 for numerical stability.
            
            
            a_tilde[:, :, w] = a_w

        a_tilde_m = F.interpolate(torch.from_numpy(a_tilde[None,:,:,:]), size=(latent_height, window_size), mode='nearest')[0].numpy()  # (1 h window_size)
        logs['a_T'][:, :, window_rng] += a_tilde_m

        # reconstructed X
        if compute_reconstructed_X:
            x_recons = maskgit.decode_token_ind_to_timeseries(s).cpu().numpy()  # (1 1 window_size)
            x_recons = (x_recons * sigma.numpy()) + mu.numpy()  # (1 1 window_size)
            x_recons = F.interpolate(torch.from_numpy(x_recons), size=(window_size,), mode='linear').numpy()  # (1 1 window_size)
            logs['reconsX'][:, window_rng] += x_recons[0]

        # log per window
        logs['count'][:, window_rng] += 1

    # resulting log
    logs['last_window_rng'] = window_rng
    logs['count'] = np.clip(logs['count'], 1, None)  # to prevent zero division.
    logs['reconsX'] = logs['reconsX'] / logs['count']
    logs['timestep_rng'] = timestep_rng

    return logs


def postprocess_XY(X):
    """
    X|Y: (n_rollowing_window_steps, n_channels, window_size)
    return X|Y_flat (b, ts_len)|
    """
    first_steps_at_every_window = X[:, 0, 0]
    last_window = X[-1, 0, 1:]
    X_pp = np.concatenate((first_steps_at_every_window, last_window), axis=-1)
    return X_pp


def postprocess_dist(dist, window_size, X_test_unscaled_pp, H_prime, latent_timesteps):
    """
    dist: (rolling_window_time_steps, latent_height, latent_width)
    """
    ts_length = X_test_unscaled_pp.shape[0]
    dist_interp = F.interpolate(torch.from_numpy(dist)[None, :, :],
                                size=(H_prime, window_size),
                                mode='nearest')[0].numpy()
    dist_pp = np.ones((dist_interp.shape[1], ts_length)) * dist_interp.min()  # (height, ts_len)
    dist_pp_count = np.ones_like(dist_pp)
    for i, t in enumerate(latent_timesteps):
        s = dist_interp[i]  # (height, window_size)
        dist_pp[:, t:t + window_size] += s
        dist_pp_count[:, t:t + window_size] += 1
    dist_pp /= dist_pp_count  # (height, ts_len)


    return dist_pp


def postprocess_Xhat(Xhat, rolling_window_stride):
    """
    Xhat: (rolling window timesteps, n_channels, window_size)
    """
    # part 1
    Xhat_pp = {}
    total_rolling_window_steps = Xhat.shape[0]
    for timestep in range(0, total_rolling_window_steps, rolling_window_stride):
        xhat = Xhat[[timestep]]  # (1, n_channels, window_size)
        window_size = xhat.shape[-1]
        for l in range(window_size):
            Xhat_pp.setdefault(timestep + l, [])
            Xhat_pp[timestep + l].append(xhat[0, 0, l])

    # part 2
    idx_middle = len(Xhat_pp) // 2  # to select the timestep with the largest list length
    max_list_len = len(Xhat_pp[idx_middle])
    Xhat_pp_new = {i: [] for i in range(max_list_len)}
    for timestep in Xhat_pp.keys():
        for i in range(max_list_len):
            try:
                Xhat_pp_new[i].append(Xhat_pp[timestep][i])
            except IndexError:
                Xhat_pp_new[i].append(np.nan)

    for i in range(max_list_len):
        Xhat_pp_new[i] = np.array(Xhat_pp_new[i])
    Xhat_pp = Xhat_pp_new

    return Xhat_pp


def compute_latent_window_size(latent_width, latent_window_size_rate):
    latent_window_size = latent_width * latent_window_size_rate
    if np.floor(latent_window_size) == 0:
        latent_window_size = 1
    elif np.floor(latent_window_size) % 2 != 0:
        latent_window_size = int(np.floor(latent_window_size))
    elif np.ceil(latent_window_size) % 2 != 0:
        latent_window_size = int(np.ceil(latent_window_size))
    elif latent_window_size % 2 == 0:
        latent_window_size = int(latent_window_size + 1)
    else:
        raise ValueError
    return latent_window_size


@torch.no_grad()
def evaluate_fn(config,
                args,
                dataset_idx: int,
                latent_window_size_rate: float,
                rolling_window_stride_rate: float,
                q: float,
                device: int = 0):
    """
    @ settings
    - device: gpu device index
    - dataset_idx: dataset index
    - rolling_window_stride_rate: stride = rolling_window_stride_rate * window_size
    - latent_window_size_rate: latent_window_size = latent_window_size_rate * latent_window_size (i.e., latent width)
    """
    # load model
    '''
    input_length = window_size = set_window_size(dataset_idx, config['dataset']['n_periods'])
    '''

    data = np.hstack(load_data(dataset_idx, config, 'train'))
    window_size = set_window_size(data) *config['dataset']['n_periods']
    anom_type = PPG_TestSequence.get_name_by_id(dataset_idx)
    

    stage2 = SetStage2.load_from_checkpoint(os.path.join('saved_models', 'ArrhyMamba.ckpt'), 
                                                config=config, 
                                                map_location=f'cuda:{device}', strict=False)

    stage2.model.decoder.interp = nn.Upsample(size=window_size, mode='linear')
        
    model = stage2.model

    model.eval()
     


    # kernel size (needs to be an odd number just like for the convolutional layers.)
    rolling_window_stride = round(window_size * rolling_window_stride_rate)
    latent_window_size = compute_latent_window_size(model.W_prime.item(), latent_window_size_rate)


    

    print('===== compute the anomaly scores of the training set... =====')
    X_train_unscaled = load_data(dataset_idx, config, 'train')  # (1, ts_len)
    logs_train = detect(args,
                        X_train_unscaled,
                        model,
                        window_size,
                        rolling_window_stride,
                        latent_window_size,
                        compute_reconstructed_X=False,
                        device=device)

    # anomaly threshold
    if q <= 1.0:
        anom_threshold = np.quantile(logs_train['a_T'], q=q, axis=-1)  # (n_channels H')
    else:
        anom_threshold = np.quantile(logs_train['a_T'], q=1.0, axis=-1)  # (n_channels H')
        anom_threshold += anom_threshold * (q - 1.0)

    print('===== compute the anomaly scores of the test set... =====')
    X_test_unscaled, Y = load_data(dataset_idx, config, 'test')  # (1, ts_len), (ts_len,)
    logs_test = detect(args,
                       X_test_unscaled,
                       model,
                       window_size,
                       rolling_window_stride,
                       latent_window_size,
                       compute_reconstructed_X=True,
                       device=device)
    
 

    # clip up to the last timestep
    X_test_unscaled = X_test_unscaled[:, :logs_test['last_window_rng'].stop]
    a_T = logs_test['a_T'][:, :, :logs_test['last_window_rng'].stop]
    X_recons_test = logs_test['reconsX'][:, :logs_test['last_window_rng'].stop]
    Y = Y[:logs_test['last_window_rng'].stop]

    # anomaly score
    # univariate time series; choose the first channel
    X_test_unscaled = X_test_unscaled[0]  # (ts_len')
    a_T = a_T[0]  # (n_freq, ts_len')
    X_recons_test = X_recons_test[0]  # (ts_len')
    anom_threshold = anom_threshold[0]  # univariate time series; (n_freq,)

    # ================================ plot ================================
    n_rows = 6
    fig, axes = plt.subplots(n_rows, 1, figsize=(25, 1.5 * n_rows))
    fontsize= 15

    # plot: X_test & labels
    i = 0
    axes[i].plot(X_test_unscaled, color='black')
    axes[i].set_xlim(0, X_test_unscaled.shape[0] - 1)
    axes[i].set_title(f"{dataset_idx}_{anom_type} | latent window size rate: {latent_window_size_rate}", fontsize=20)
    ax2 = axes[i].twinx()
    ax2.plot(Y, alpha=0.5, color='C1')

    # plot: anomaly score
    i += 1
    vmin = np.nanquantile(np.array(a_T).flatten(), q=0.5)
    axes[i].imshow(a_T, interpolation='nearest', aspect='auto', cmap='magma', vmin=vmin)
    axes[i].invert_yaxis()
    axes[i].set_xticks([])
    ylabel = 'clipped\n' + r'$a_T$'
    axes[i].set_ylabel(ylabel, fontsize=fontsize, rotation=0, labelpad=10, ha='right', va='center')

    ylim_max = np.max(a_T) * 1.05
    for j in range(a_T.shape[0]):
        i += 1
        axes[i].plot(a_T[j])
        axes[i].set_xticks([])
        xlim = (0, a_T[j].shape[0] - 1)
        axes[i].set_xlim(*xlim)
        axes[i].set_ylim(None, ylim_max)

        h_idx = f'H={j}'
        axes[i].set_ylabel(r'$(a_T)_{{{}}}$'.format(h_idx),
                           fontsize=fontsize, rotation=0, labelpad=35, va='center')
        threshold = 1e99 if anom_threshold[j] == np.inf else anom_threshold[j]
        axes[i].hlines(threshold, xmin=xlim[0], xmax=xlim[1], linestyle='--', color='black')

    # plot: reconstruction
    i += 1
    axes[i].plot(X_test_unscaled, color='black')
    axes[i].plot(X_recons_test, alpha=0.5, color='C1')
    axes[i].set_xticks([])
    axes[i].set_xlim(0, X_test_unscaled.shape[0] - 1)
    axes[i].set_ylabel('recons', fontsize=fontsize)

    # save: plot
    plt.tight_layout()
    plt.savefig(get_root_dir().joinpath('evaluation', 'results', f'{dataset_idx}_{anom_type}-anomaly_score-latent_window_size_rate_{latent_window_size_rate}.png'))
    plt.close()

    # save: resulting data
    resulting_data = {'dataset_index': dataset_idx,
                      'latent_window_size_rate': latent_window_size_rate,
                      'latent_window_size': latent_window_size,
                      'rolling_window_stride_rate': rolling_window_stride_rate,
                      'q': q,

                      'X_test_unscaled': X_test_unscaled,
                      'Y': Y,
                      'a_T': a_T,
                      'X_recons_test': X_recons_test,

                      'timestep_rng_test': logs_test['timestep_rng'],
                      'anom_threshold': anom_threshold,  # (n_freq,)
                      }

    saving_fname = get_root_dir().joinpath('evaluation', 'results', f'{dataset_idx}_{anom_type}-anomaly_score-latent_window_size_rate_{latent_window_size_rate}.pkl')
    with open(str(saving_fname), 'wb') as f:
        pickle.dump(resulting_data, f, pickle.HIGHEST_PROTOCOL)
    

@torch.no_grad()
def save_final_summarized_figure(dataset_idx, X_test_unscaled, Y, timestep_rng_test,
                                 a_T, a_bar_t, a_bar_max, a_final,
                                 joint_threshold, final_threshold, anom_ind,
                                 window_size, config, args):
    
    n_rows = 9
    fig, axes = plt.subplots(n_rows, 1, figsize=(25, 1.5 * n_rows))
    fontsize= 15
    anom_type = PPG_TestSequence.get_name_by_id(dataset_idx)
    
    # plot: X_test & labels
    i = 0
    axes[i].plot(X_test_unscaled, color='black')
    axes[i].set_xlim(0, X_test_unscaled.shape[0] - 1)
    axes[i].set_title(f'{dataset_idx} | {anom_type}', fontsize=fontsize)
    ax2 = axes[i].twinx()
    ax2.plot(Y, alpha=0.5, color='C1')

    # plot (imshow): a_s^*
    i += 1
    a_T_clipped = np.copy(a_T)
    a_T_clipped[:, ~anom_ind] = 0. if anom_ind.mean() == 0 else np.min(a_T_clipped[:, anom_ind])
    axes[i].imshow(a_T_clipped, interpolation='nearest', aspect='auto', cmap='magma')  # , vmin=vmin)
    axes[i].invert_yaxis()
    axes[i].set_xticks([])
    ylabel = 'clipped\n' + r'$a_T$'
    axes[i].set_ylabel(ylabel, fontsize=fontsize, rotation=0, labelpad=10, ha='right', va='center')

    # plot: a_T
    n_freq = a_T.shape[0]
    max_anom = a_T.max()
    for j in range(n_freq):
        i += 1
        axes[i].plot(a_T[j], color='green')
        axes[i].set_xticks([])
        axes[i].set_xlim(0, a_T.shape[1] - 1)
        h_idx = f'h={j}'
        axes[i].set_ylabel(r'$(a_T)_{{{}}}$'.format(h_idx), fontsize=fontsize, rotation=0, labelpad=30, va='center')
        axes[i].set_ylim(None, max_anom + 0.05 * max_anom)
        axes[i].hlines(joint_threshold[j], xmin=0, xmax=len(a_T[j]) - 1, linestyle='--', color='black')





    # plot: bar{a}_T
    i += 1
    axes[i].plot(a_bar_t, color='darkturquoise')
    axes[i].set_xticks([])
    axes[i].set_xlim(0, len(a_bar_t) - 1)
    axes[i].set_ylabel(r'$\bar{a}$', fontsize=fontsize, rotation=0, labelpad=15, va='center')
    axes[i].hlines(final_threshold, xmin=0, xmax=len(a_bar_t) - 1, linestyle='--', color='black')

    # plot: bar{a}_max
    i += 1
    rng = np.arange(len(a_bar_max))
    axes[i].plot(rng, a_bar_max, color='royalblue')
    axes[i].set_xticks([])
    axes[i].set_xlim(0, len(a_bar_max) - 1)
    axes[i].set_ylabel(r'${\bar{a}}_{max}$', fontsize=fontsize, rotation=0, labelpad=15, va='center')
    axes[i].hlines(final_threshold, xmin=0, xmax=len(a_bar_max) - 1, linestyle='--', color='black')

    # plot: a_final
    i += 1
    rng = np.arange(len(a_final))
    axes[i].plot(rng, a_final, color='purple')

    axes[i].set_xticks([])
    axes[i].set_xlim(0, len(a_final) - 1)
    axes[i].set_ylabel(r'$a_{F}$', fontsize=fontsize, rotation=0, labelpad=25, va='center')
    axes[i].hlines(final_threshold, xmin=0, xmax=len(a_final) - 1, linestyle='--', color='black')

    true_anomalies = np.where(Y > 0)[0]
    pred_anomalies = np.where(a_final > final_threshold)[0]

    
    result = []
    tolerance = [10, 30, 50] 
    for tol in tolerance:
        TP = 0
        for pred_idx in pred_anomalies:
            if np.any(np.abs(true_anomalies - pred_idx) <= tol):  
                TP += 1

        FP = len(pred_anomalies) - TP 

        FN = 0
        for true_idx in true_anomalies:
            if not np.any(np.abs(pred_anomalies - true_idx) <= tol):  
                FN += 1

        total_points = len(Y)

        TN = total_points - (TP + FP + FN)  

        sensitivity = (TP) / (TP + FN) if (TP + FN) > 0 else 0
        recall = sensitivity
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        accuracy = (TP + TN) / (TP + FN + FP + TN) if (TP + FN + FP + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        TP_indices = [pred_idx for pred_idx in pred_anomalies if np.any(np.abs(true_anomalies - pred_idx) <= tol)]
        FP_indices = [pred_idx for pred_idx in pred_anomalies if not np.any(np.abs(true_anomalies - pred_idx) <= tol)]
        FN_indices = [true_idx for true_idx in true_anomalies if not np.any(np.abs(pred_anomalies - true_idx) <= tol)]
       
        

        tp = np.array([1 if true_idx in pred_anomalies else 0 for true_idx in true_anomalies])


        auc = compute_auc(TP_indices, FP_indices, FN_indices, len(Y))
                            
        result.append({
        "Tolerance": tol,
        "Accuracy": accuracy,
        "Recall": recall,
        "F1-Score": f1_score,
        "AUC": auc,
        "Specificity": specificity
    })
    result = pd.DataFrame(result)
    result.to_csv(f'evaluation/results/{dataset_idx}_metric.csv')
    

    # plot: explainable sampling
    if args.explainable_sampling:
        data = np.hstack(load_data(dataset_idx, config, 'train'))
        window_size = set_window_size(data) *config['dataset']['n_periods']
       
        stage2 = SetStage2.load_from_checkpoint(os.path.join('saved_models', 'ArrhyMamba.ckpt'), 
                                            config=config, 
                                            map_location=f'cuda:{args.device}', strict=False)

        model = stage2.model
        model.decoder.interp = nn.Upsample(size=window_size, mode='linear')
        model.eval()

        
        
        i+= 1
        for timestep_idx, timestep in enumerate(timestep_rng_test):
            print(f'explainable sampling.. {round(timestep_idx / len(timestep_rng_test) * 100)}%')
            
            window_rng = slice(timestep, timestep + window_size)
            x_unscaled = X_test_unscaled[window_rng]  
            mu = np.nanmean(x_unscaled, axis=-1, keepdims=True)
            sigma = np.nanstd(x_unscaled, axis=-1, keepdims=True)
            sigma = np.clip(sigma, 1.e-4, None)
            x = (x_unscaled - mu) / sigma

            # **Encode using the fine-tuned encoder & vq_model**
            z_q, s = model.encode_to_z_q(torch.from_numpy(x[None, None, :]).to(args.device), model.encoder, model.vq_model)

            latent_height, latent_width = z_q.shape[2], z_q.shape[3]
            anom_window = a_final[window_rng]
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

                
                xhat = model.decode_token_ind_to_timeseries(s_Tstar).cpu().numpy()  
                xhat = (xhat * sigma) + mu  
                xhat = xhat[:, 0, :]
                colors = plt.get_cmap('tab10').colors
                
                for b in range(xhat.shape[0]):
                    color = colors[b % len(colors)]
                    axes[i].plot(np.arange(timestep, timestep + window_size), xhat[b], alpha=0.5, color=color)
            else:
                axes[i].plot(np.arange(timestep, timestep + window_size), x_unscaled, alpha=0.5, color='black')

        axes[i].set_xticks([])
        axes[i].set_xlim(0, len(X_test_unscaled) - 1)
        axes[i].set_ylabel(r'counter\nfactual', fontsize=fontsize)

    # save: fig
    plt.tight_layout()

    accuracy = result['Accuracy'][0]
    sensitivity = result['Recall'][0]

    plt.savefig(get_root_dir().joinpath('evaluation', 'results', f'{dataset_idx}_{anom_type} | accuracy : {accuracy:.3f}, sensitivity: {sensitivity:.3f}.png'))
    plt.close()

