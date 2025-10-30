from model import SimpleRNN, VRNN
import os
from pathlib import Path
import torch
import sys
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('error')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from PreProParadigm.audit_gm import NonHierachicalAuditGM, HierarchicalAuditGM


# Check which folder exists and append the correct path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
kalman_folder = None
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
if os.path.exists(os.path.join(base_path, 'Kalman')):
    from Kalman.kalman import kalman_tau, plot_estim, kalman_batch, kalman_fit_batch
elif os.path.exists(os.path.join(base_path, 'KalmanFilterViz1D')):
    from KalmanFilterViz1D.kalman import kalman_tau, plot_estim, kalman_batch
else:
    raise ImportError("Neither 'Kalman' nor 'KalmanFilterViz1D' folder found.")




DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FREQ_MIN = 1400
FREQ_MAX = 1650



# TODO CHECK-LIST
# - a loss tolerance?
# - an early stopping in case of overfitting (ABSOLUTELY according to Seyma) --> include IF i observe overfitting
# - dropout?
# - assess need to change activation function or not (hand in hand with changing the models' inner layers' architecture)
# - set requires_grad / autograd
# y_norm training + validation in eval mode


# TODO:
# - load params


class MinMaxScaler():
    def __init__(self):
        self.freq_min = FREQ_MIN
        self.freq_max = FREQ_MAX

    def fit_normalize(self, x, freq_min=None, freq_max=None, margin=None):
        if freq_min is None:
            self.freq_min = x.min()
        if freq_max is None:
            self.freq_max = x.max()
        if margin is not None:
            self.freq_min -= margin
            self.freq_max += margin
        return self.normalize(x)

    def normalize(self, x):
        return (x - self.freq_min) / (self.freq_max - self.freq_min)
    
    def denormalize_mean(self, mu_norm):
        return mu_norm * (self.freq_max - self.freq_min) + self.freq_min
    
    def denormalize_var(self, var_norm):
        return var_norm * (self.freq_max - self.freq_min)**2



def compute_benchmarks(gm, N_samples, param_bins=None, n_iter=5, benchmarkpath='kalman_benchmark.pkl', save=True):
    """
    Benchmarks the Kalman filter on a batch of data, tracking MSE along parameter configurations.

    Args:
        y_batch (np.ndarray): Batch of observations, shape (N_samples, seq_len).
        pars_batch (np.ndarray): Batch of parameters, shape (N_samples, n_params).
        param_bins (dict, optional): Dictionary with bin edges for each parameter.
        n_iter (int): Number of iterations for kalman_fit_batch.

    Returns:
        dict: Mapping from parameter bin tuples to {'mse': avg_mse, 'count': count}.
    """
    # Generate a batch of data
    _, _, y_batch, pars_batch = gm.generate_batch(N_samples, return_pars=True)

    # Fit Kalman filter to each sample in the batch
    kalman_mu, kalman_sigma = kalman_fit_batch(y_batch, n_iter=n_iter)  # shape: (N_samples, seq_len)
    mses = ((kalman_mu - y_batch) ** 2).mean(axis=1)  # shape: (N_samples,)

    # Return observations, Kalman estimates, parameters, and binned performance metrics
    benchmark_kit = {'y': y_batch, 'mu_kal': kalman_mu, 'sigma_kal': kalman_sigma, 'pars': pars_batch, 'perf': mses}

    if not os.path.exists(os.path.split(benchmarkpath)[0]):
        os.mkdir(os.path.split(benchmarkpath)[0])

    if save:
        with open(benchmarkpath, 'wb') as f:
            pickle.dump(benchmark_kit, f)

    return benchmark_kit




def train(model, model_config, lr, lr_id, gm_name, data_config, kalman=False, device=DEVICE):
    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=model_config["weight_decay"])
    
    # Pass model to GPU or CPU
    model.to(device)

    # As a check:
    if data_config["N_ctx"] > 1: kalman=False
    
    # Define data generative model
    if gm_name == 'NonHierarchicalGM': gm = NonHierachicalAuditGM(data_config)
    elif gm_name == 'HierarchicalGM': gm = HierarchicalAuditGM(data_config)
    else: raise ValueError("Invalid GM name")

    # Load validation batch and computed benchmarks
    if kalman:
        with open('kf_benchmark.pkl', 'rb') as f:
            kf_benchmark = pickle.load(f) # dict with fields: 'y' (observation), 'mu_kal' (mean of the estimation distribution of hidden state by KF), 'sigma_kal' (variance of the estimation distribution of hidden state by KF), 'pars' (KF parameters fit), 'perf' (MSE of predicted next observation with true next observation)

    # Prepare to save the results
    lr_title = f"Learning rate: {lr:>6.0e}"
    if kalman:
        save_path = Path(os.path.abspath(os.path.dirname(__file__))) / f'training_results/N_ctx_{data_config["N_ctx"]}/kalman/{model.name}/'
    else:
        save_path = Path(os.path.abspath(os.path.dirname(__file__))) / f'training_results/N_ctx_{data_config["N_ctx"]}/observ/{model.name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path / 'samples/')

    epoch_steps         = []
    train_losses_report = []
    train_steps         = []
    valid_losses_report = []
    valid_steps         = []
    valid_mse_report    = []
    valid_sigma_report  = []
    if kalman:
        mse_kal_report   = []
        model_kf_mse_report = []

    # Define loss instance
    loss_function   = torch.nn.GaussianNLLLoss(reduction='sum') # GaussianNLL

    # Track learning of model parameters 
    weights_updates   = []
    param_names     = model.state_dict().keys()

    # Epochs
    for epoch in tqdm(range(model_config["num_epochs"]), desc="Epochs"):

        ### TRAINING
        print("TRAINING \n")
        train_losses = []
        tt = time.time()
        model.train()

        # Generate N_samples=batch_size per batch in N_batch=n_batches
        for batch in tqdm(range(model_config["n_batches"]), desc="Batches", leave=False):

            optimizer.zero_grad()
            
            # Generate data (N_samples samples)
            _, _, y = gm.generate_batch(return_pars=False)
            y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(device)

            # Transform data if specified
            if model_config["use_minmax"]:
                minmax_train = MinMaxScaler()
                y_norm = minmax_train.fit_normalize(y) #, margin=data_config["si_r"]+data_config["si_q"])
            else:
                y_norm = y

            # Get model prediction and compute loss
            if kalman:
                # If learning the Kalman filter, train to estimate the next timestep
                # loss_obs = lossfunc(x[:,1:,0], u[:,1:,0], s[:,1:,0]**2)
                # dim: batch_size, seq_len, input_dim
                # Call model #  # u[:, 1:, :], s[:, 1:, :], l[:, 1:, :] = self.call(x[:, :-1, :])

                model_output = model(y_norm[:, :-1, :]) # can contain output(, mu_latent, logvar_latent(, mu_prior, logvar_prior))
                    
                # Compute loss
                loss = model.loss(y_norm[:, 1:, :], model_output, loss_func=loss_function)
                
            else:
                # Or train to estimate the observations:
                # Call model #  # u[:, 1:, :], s[:, 1:, :], l[:, 1:, :] = self.call(x[:, :-1, :])
                model_output = model(y_norm[:, :-1, :]) # can contain output(, mu_latent, logvar_latent(, mu_prior, logvar_prior))
                # Compute loss
                loss = model.loss(y_norm, model_output, loss_func=loss_function)

            # Learning model weigths
            weights_before = [w.detach().clone() for w in model.parameters()]
            loss.backward()     
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().numpy().item()))
            weights_after = [w.detach().clone() for w in model.parameters()]

            
            # Logging and reporting
            if batch % model_config["batch_res"] == model_config["batch_res"]-1:
                train_steps.append(epoch * model_config["n_batches"] + batch)
                
                # Store loss
                train_losses_report.append(float(loss.detach().cpu().numpy().item()))
                loss_report = np.mean(train_losses[-model_config["batch_res"]:])
                sprint=f'LR: {lr:>6.0e}; epoch: {epoch:0>3}; batch: {batch:>3}; loss: {float(loss.detach().cpu().numpy().item()):>7.4f}; batch loss: {loss_report:7.4f}; time: {time.time()-tt:>.2f}; training step: {epoch * model_config["n_batches"] + batch}'
                logfilename = f'{save_path}/training_loss_lr{lr_id}.txt'
                with open(logfilename, 'a') as f:
                    f.write(f'{sprint}\n')
                tt = time.time()

                # Store evolution of parameters
                weights_update = torch.stack([torch.mean((w_after-w_before)**2) for w_before, w_after in zip(weights_before, weights_after)])
                weights_updates.append(weights_update)
            
        avg_train_loss = np.mean(train_losses) # average loss for current epoch


        ### VALIDATION
        print("VALIDATION \n")
        model.eval()
        with torch.no_grad():    
            # If kalman, use benchmark batch, else, generate new batch
            if kalman:
                y = kf_benchmark['y'] # TODO: sanity check that they are the same sizes
                mu_kal = kf_benchmark['mu_kal']
                sigma_kal = kf_benchmark['sigma_kal']
                mse_kal = kf_benchmark['perf'] # TODO: not sure if interesting here
                pars = kf_benchmark['pars']
            else:
                _, _, y, pars = gm.generate_batch(return_pars=True)
                y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(device)
            
            # Transform data
            # minmax_valid = MinMaxScaler()
            if model_config["use_minmax"]:
                y_norm = minmax_train.normalize(y)
            else:
                y_norm = y

            # Get model prediction and compute loss
            if kalman:
                # If learning the Kalman filter, train to estimate the next timestep
                # Call model
                model_output = model(y_norm[:, :-1, :]) # can contain output(, mu_latent, logvar_latent(, mu_prior, logvar_prior))

                # Compute loss
                valid_loss = model.loss(y_norm[:, 1:, :], model_output, loss_func=loss_function).detach().cpu().numpy()

            else:
                # Else train to estimate the observations
                # Call model
                model_output = model(y_norm) # can contain output_dist(, mu_latent, logvar_latent(, mu_prior, logvar_prior))
                # Compute loss
                valid_loss = model.loss(y_norm, model_output, loss_func=loss_function).detach().cpu().numpy()                
        

            # Get estimated distribution
            if model.name=='vrnn': model_output_dist = model_output[0]
            else: model_output_dist = model_output

            # Extract model's output
            mu_estim = F.sigmoid(model_output_dist[...,0]).detach().cpu().numpy()
            var_estim = F.softplus(model_output_dist[..., 1] + 1e-6).detach().cpu().numpy()
            
            # Rescale data according to observation scale
            if model_config["use_minmax"]:
                mu_estim    = minmax_train.denormalize_mean(mu_estim)
                var_estim   = minmax_train.denormalize_var(var_estim)

            # Compute standard deviation                    
            sigma_estim = np.sqrt(var_estim)

            # Reshape variables
            y = y.detach().cpu().numpy().squeeze()

            # Evaluate MSE to true value that the model was trained on (observations or Kalman)
            if kalman:
                # If trying to learn the Kalman filter, compare the prediction of the next observation with the next observation
                mse_model = ((mu_estim - y[:, 1:])**2).mean()

                # Compare network's output with Kalman filter's estimation
                mse_model2kal = ((mu_estim - mu_kal[:, 1:])**2).mean()  # TODO: check shape of Kalman estimate // model 2 true, kf 2 true, but interested in model 2 kf? If yes, at kf - model

            else:
                # If trying to learn the observations, compare with observations
                mse_model = ((mu_estim - y)**2).mean()
            
            # Save valid samples for this epoch
            if epoch % model_config["epoch_res"] == model_config["epoch_res"]-1:
                if kalman:
                    plot_samples(y, mu_estim, sigma_estim, params=pars, save_path=f'{save_path}/samples/lr{lr_id}-epoch-{epoch:0>3}_samples', title=lr_title, kalman_mu=kalman_mu, kalman_sigma=kalman_sigma)
                else:
                    plot_samples(y, mu_estim, sigma_estim, params=pars, save_path=f'{save_path}/samples/lr{lr_id}-epoch-{epoch:0>3}_samples', title=lr_title)


            # Store valid metrics for this epoch
            valid_mse_report.append(mse_model)
            valid_losses_report.append(valid_loss) # avg loss for epoch (over batches)
            valid_steps.append(epoch*model_config["n_batches"]+model_config["n_batches"])
            epoch_steps.append(epoch)
            valid_sigma_report.append(sigma_estim)
            if kalman:
                model_kf_mse_report.append(mse_model2kal) # TODO: now not reporting KF (bc was a bit useless), but MSE of model to KF

            # Print epoch report
            print(f"Model: {model.name:>4}, LR: {lr:>6.0e}, Epoch: {epoch:>3}, Training Loss: {avg_train_loss:>7.2f}, Valid Loss: {valid_loss:>7.2f}")
            sprint=f'LR: {lr:>6.0e}; epoch: {epoch:>3}; var: {sigma_estim:>7.2f}; MSE: {mse_model:>7.2f}; time: {time.time()-tt:>.2f}; training step: {epoch * model_config["n_batches"] + model_config["n_batches"]}'
            if kalman:
                sprint += f'; Model-KF MSE: {mse_model2kal:>7.2f}'
            logfilename = save_path / f'training_log_lr{lr_id}.txt'
            with open(logfilename, 'a') as f:
                f.write(f'{sprint}\n')

    # Save training logs at the end of epochs loop
    lossplotfile = save_path/f'loss_trainvalid_lr{lr_id}.png'
    plot_losses(train_steps, valid_steps, train_losses_report, valid_losses_report, x_label='Training steps', y_label='Loss', title=lr_title, save_path=lossplotfile)
    # plot_losses(train_steps, valid_steps, epoch_steps, train_losses_report, valid_losses_report, x_label='Training steps', y_label='Loss', title=lr_title, save_path=lossplotfile)
    plot_variance(epoch_steps, valid_sigma_report, title=lr_title, save_path=save_path/f'variance_valid_lr{lr_id}.png')
    if kalman:
        plot_mse(epoch_steps, valid_mse_report, title=lr_title, save_path=save_path/f'mse_valid_lr{lr_id}.png', mse_kal=mse_kal_report, model_mse_kal=model_kf_mse_report)
    else:
        plot_mse(epoch_steps, valid_mse_report, title=lr_title, save_path=save_path/f'mse_valid_lr{lr_id}.png')
    

    # Save model's weights
    torch.save(model.state_dict(), f'{save_path}/lr{lr_id}_weights.pth')

    # Plot training weights updates
    if len(weights_updates) > 0:
        weights_updates = torch.stack(weights_updates, dim=1)
        plot_weights(train_steps, weights_updates, list(param_names), lr_title, save_path=save_path/f'weights_updates_lr{lr_id}.png')

    if model_config["use_minmax"]:
        return minmax_train
    else:
        return None



def test(model, model_config, lr, lr_id, gm_name, data_config, minmax_train, device=DEVICE, kalman=False):

    ### TESTING
    print("TESTING \n")

    # Paths 
    if kalman:
        save_path = Path(os.path.abspath(os.path.dirname(__file__))) / f'training_results/N_ctx_{data_config["N_ctx"]}/kalman/{model.name}/'
    else:
        save_path = Path(os.path.abspath(os.path.dirname(__file__))) / f'training_results/N_ctx_{data_config["N_ctx"]}/observ/{model.name}/'
    if not os.path.exists(save_path):
        raise ValueError("Model has not been trained yet - no folder found")

    # Load the saved weights
    model.load_state_dict(torch.load(f'{save_path}/lr{lr_id}_weights.pth'))
    model.to(device)
    model.eval()  # Switch to evaluation mode

    # As a check:
    if data_config["N_ctx"] > 1: kalman=False
    
    # Define data generative model
    if gm_name == 'NonHierarchicalGM': gm = NonHierachicalAuditGM(data_config)
    elif gm_name == 'HierarchicalGM': gm = HierarchicalAuditGM(data_config)
    else: raise ValueError("Invalid GM name")

    # If testing data parameters and performance
    if data_config["params_testing"]:
        # NOTE: same spacing as sampled in audit_gm.py "if param_testing" but regularly spaced
        # NOTE: here assuming that all 3 params are being tested
        tau_bins = np.logspace(np.log10(data_config["mu_tau_bounds"]["low"]), np.log10(data_config["mu_tau_bounds"]["high"]), 10)
        si_stat_bins = np.logspace(np.log10(data_config["si_stat_bounds"]["low"]), np.log10(data_config["si_stat_bounds"]["high"]), 10)
        si_r_bins = np.logspace(np.log10(data_config["si_r_bounds"]["low"]), np.log10(data_config["si_r_bounds"]["high"]), 10)

        # Create a grid of all parameter combinations
        param_bins = {'tau': tau_bins,
                      'si_stat': si_stat_bins,
                      'si_r': si_r_bins}
        param_pairs = np.array(np.meshgrid(tau_bins, si_stat_bins, si_r_bins)).T.reshape(-1, 3)

        # Initialize storage array: each row stores tau, lim, si_stat, si_q, and the corresponding mse
        binned_metrics = {tuple(param_combination): {'mse': [], 'count': 0} for param_combination in param_pairs}


    valid_sigma = []
    valid_mse = []
    mse_kals = []
    model.eval()

    # Generate data
    _, _, y, pars = gm.generate_batch(N_samples=model_config["batch_size"], return_pars=True)
    y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(device)
   
    # Transform data
    if model_config["use_minmax"]:
        y_norm = minmax_train.normalize(y)
    else:
        y_norm = y


    with torch.no_grad():
        # # Call model
        # model_output = model(y_norm) # can contain output_dist(, mu_latent, logvar_latent(, mu_prior, logvar_prior))

        # Get model prediction
        if kalman:
            # If learned the Kalman filter, estimate the next timestep
            # Call model
            model_output = model(y_norm[:, :-1, :]) # can contain output(, mu_latent, logvar_latent(, mu_prior, logvar_prior))
        else:
            # Else if trained to estimate the observations
            # Call model
            model_output = model(y_norm) # can contain output_dist(, mu_latent, logvar_latent(, mu_prior, logvar_prior))

        # Get estimated distribution
        if model.name=='vrnn': model_output_dist = model_output[0]
        else: model_output_dist = model_output
        
        # Extract model's output
        mu_estim = F.sigmoid(model_output_dist[...,0]).detach().cpu().numpy()
        var_estim = F.softplus(model_output_dist[..., 1] + 1e-6).detach().cpu().numpy()
        
        # Rescale data according to observation scale
        if model_config["use_minmax"]:
            mu_estim    = minmax_train.denormalize_mean(mu_estim)
            var_estim   = minmax_train.denormalize_var(var_estim)

        # Compute standard deviation                    
        sigma_estim = np.sqrt(var_estim)
        valid_sigma.append(sigma_estim)

        # Reshape variables
        y = y.detach().cpu().numpy().squeeze()

        # Evaluate MSE to true value that the model was trained on (observations or next observation)
        if kalman:
            # If trying to learn the Kalman filter, compare prediction of next observation with next observation
            mse = ((mu_estim-y[:, 1:])**2).mean()
        else:
            # If trying to learn the observations, compare with observations
            mse = ((mu_estim-y)**2).mean()
        valid_mse.append(mse)
        
        if kalman:
            # TODO: USE BENCHMARK HERE
            

            # Compare network's output with Kalman filter's estimation MSE
            kalman_mu, kalman_sigma = kalman_fit_batch(y, n_iter=5) 
            # Compare Kalman filter's predicted observations to true observations
            mse_kal = ((kalman_mu-y)**2).mean()
            mse_kals.append(mse_kal)
        
        # Track performance along data parameters
        if data_config["params_testing"]:
            # Digitize each of these parameters to find the corresponding bin
            tau_bin = np.digitize(pars[:,0], param_bins['tau']) - 1
            si_stat_bin = np.digitize(pars[:,2], param_bins['si_stat']) - 1
            si_r_bin = np.digitize(pars[:,4], param_bins['si_r']) - 1
            # si_q_bin = np.digitize(si_q, param_bins['si_q']) - 1

            # Use the bins found to get the corresponding combination of parameters
            param_combination = (param_bins['tau'][tau_bin], param_bins['si_stat'][si_stat_bin], param_bins['si_r'][si_r_bin])
            
            # Get MSE per sample in batch
            if kalman:
                # If trying to learn the Kalman filter, compare with hidden prediction of next observation
                mse_per_sample = ((mu_estim-y[:, 1:])**2).mean(dim=1).cpu().numpy() # shape (N_samples,)
            else:
                # If trying to learn the observations, compare with observations
                mse_per_sample = ((mu_estim-y)**2).mean(dim=1)

            # Then, zip batch's param_combination array and MSE array and store MSE
            for *pc, m in zip(*param_combination, mse_per_sample):
                binned_metrics[tuple(pc)]['mse'].append(m)


    # If tracking performance along data parameters, average stored MSEs for each parameter combination
        if data_config["params_testing"]:
            for param_combination in binned_metrics.keys():
                binned_metrics[param_combination]['count'] = len(binned_metrics[param_combination]['mse'])
                if binned_metrics[param_combination]['count'] > 0:
                    binned_metrics[param_combination]['mse'] = np.mean(binned_metrics[param_combination]['mse'])
                else:
                    binned_metrics[param_combination]['mse'] = np.nan

    
    # Save binned metrics
    if data_config["params_testing"]:
        binned_metrics_list = [
            {
            'tau': tau,
            'si_stat': si_stat,
            'si_r': si_r,
            'mse': metrics['mse'],
            'count': metrics['count']
            }
            for (tau, si_stat, si_r), metrics in binned_metrics.items()
        ]
        binned_metrics_df = pd.DataFrame(binned_metrics_list)
        binned_metrics_df['si_q'] = binned_metrics_df['si_stat'] * ((2 * binned_metrics_df['tau'] - 1) ** 0.5) / binned_metrics_df['tau']
        binned_metrics_df.to_csv(save_path/f'binned_metrics_lr{lr_id}.csv', index=False)




def plot_weights(train_steps, weights_updates, names, title, save_path):
    plt.figure(figsize=(10, 5))
    for param in range(weights_updates.shape[0]):
        plt.plot(train_steps, weights_updates[param], label=names[param])
    plt.yscale('log')
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_mse(valid_steps, valid_diff, title, save_path, mse_kal=None, model_mse_kal=None):
    # Plot MSE depending on what's given
    plt.figure(figsize=(10, 5))
    plt.plot(valid_steps, valid_diff, label='model')
    if mse_kal is not None:
        plt.plot(valid_steps, mse_kal, label='kalman')
    if model_mse_kal is not None:
        plt.plot(valid_steps, model_mse_kal, label='model-kalman')
    plt.xlabel("Epoch steps")
    plt.yscale('log')
    plt.ylabel(f"Validation MSE")
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_variance(valid_steps, valid_sigma, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(valid_steps, valid_sigma, label="valid std")
    plt.xlabel("Epoch steps")
    plt.ylabel("Valid variance (std)")
    plt.yscale('log')
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_losses(train_steps, valid_steps, train_losses_report, valid_losses_report, x_label, y_label, title, save_path):
# def plot_losses(train_steps, valid_steps, epoch_steps, train_losses_report, valid_losses_report, x_label, y_label, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_losses_report, label="train loss")
    plt.plot(valid_steps, valid_losses_report, label="valid loss")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    # sec = plt.gca().secondary_xaxis(location=0)
    # sec.set_xticks(epoch_steps)
    # sec.set_xlabel(f'epoch=', loc='left', labelpad=-9)
    plt.savefig(save_path)
    plt.close()


def plot_samples(obs, mu_estim, sigma_estim, save_path, title=None, params=None, kalman_mu=None, kalman_sigma=None, states=None):
    obs = obs.squeeze() # for safety
            
    # for some 8 randomly sampled sequences out of the whole batch of length obs.size(0)
    N = min(8, obs.size(0))
    for id, i in enumerate(np.random.choice(range(N), size=(N,), replace=False)):
            
        plt.figure(figsize=(20, 6))

        # Plot hidden process states if provided (NOTE - caution: can only work if states is 1D, i.e. only 1 context)
        if states is not None:
            plt.plot(range(len(states[i])), states[i], label='x_hid', color='orange', linewidth=2)
        
        # Plot observation
        plt.plot(range(len(obs[i])), obs[i], color='tab:blue', label='y_obs')

        # Plot estimation as a distribution (with uncertainty)
        plt.plot(range(len(mu_estim[i])), mu_estim[i], color='k', label='y_hat')
        plt.fill_between(range(len(mu_estim[i])), mu_estim[i]-sigma_estim[i], mu_estim[i]+sigma_estim[i], color='k', alpha=0.2)
        
        # Plot Kalman estimation if provided
        if kalman_mu is not None and kalman_sigma is not None:
            plt.plot(range(len(kalman_mu[i])), kalman_mu[i], label='y_kal', color='green', linewidth=2)
            plt.fill_between(range(len(kalman_mu[i])), kalman_mu[i]-kalman_sigma[i], kalman_mu[i]+kalman_sigma[i], color='green', alpha=0.2)

        plt.legend()
        if params is not None:
            title = f"{title}; tau: {params[i,0]:.2f}, lim: {params[i,1]:.2f}, si_stat: {params[i,2]:.2f}, si_q: {params[i,3]:.2f}"
        plt.title(title)
        plt.savefig(f'{save_path}_s{id}.png')
        plt.close()



def pipeline_single_param(model_config, data_config, gm_name, device=DEVICE):

    for nctx in [1]:    # , 2
        data_config["N_ctx"]=nctx
        for kalman_on in [True]: # [True, False]
            for lr_id, learning_rate in enumerate([0.01]): # [0.1, 0.05, 0.01, 0.005] # 0.001, 0.0005, 0.0001, 0.00005, 0.00001]):
                for model_name in ['vrnn']: # , 'vrnn'
                    # Define models
                    if model_name == 'rnn':
                        model = SimpleRNN(x_dim=model_config['input_dim'], output_dim=model_config['output_dim'], hidden_dim=model_config['rnn_hidden_dim'], n_layers=model_config['rnn_n_layers'], device=device)
                    else:
                        model = VRNN(x_dim=model_config['input_dim'], output_dim=model_config['output_dim'], latent_dim=model_config['latent_dim'], phi_x_dim=model_config['phi_x_dim'], phi_z_dim=model_config['phi_z_dim'], phi_prior_dim=model_config['phi_prior_dim'], rnn_hidden_states_dim=model_config['rnn_hidden_dim'], rnn_n_layers=model_config['rnn_n_layers'], device=device)
                    
                    # Train
                    train(model, model_config=model_config, lr=learning_rate, lr_id=lr_id, gm_name=gm_name, data_config=data_config, device=device, kalman=kalman_on)



def pipeline_multi_param(model_config, data_config, gm_name, device=DEVICE):
    
    data_config.update({
        "lim_bounds": {'low': 500, 'high': 1800},
        "mu_tau_bounds": {'low': 1, 'high': 50},
        "si_stat_bounds": {'low': 1, 'high': 50},
        "params_testing": True    
    })

    for nctx in [1]:    # , 2
        data_config["N_ctx"]=nctx
        for kalman_on in [True]: # [True, False]
            for lr_id, learning_rate in enumerate([0.01]): # [0.1, 0.05, 0.01, 0.005] # 0.001, 0.0005, 0.0001, 0.00005, 0.00001]):
                for model_name in ['rnn']: # , 'vrnn'
                    # Define models
                    if model_name == 'rnn':
                        model = SimpleRNN(x_dim=model_config['input_dim'], output_dim=model_config['output_dim'], hidden_dim=model_config['rnn_hidden_dim'], n_layers=model_config['rnn_n_layers'], device=device)
                    else:
                        model = VRNN(x_dim=model_config['input_dim'], output_dim=model_config['output_dim'], latent_dim=model_config['latent_dim'], phi_x_dim=model_config['phi_x_dim'], phi_z_dim=model_config['phi_z_dim'], phi_prior_dim=model_config['phi_prior_dim'], rnn_hidden_states_dim=model_config['rnn_hidden_dim'], rnn_n_layers=model_config['rnn_n_layers'], device=device)


                    # Train
                    minmax = train(model, model_config=model_config, lr=learning_rate, lr_id=lr_id, gm_name=gm_name, data_config=data_config, device=device, kalman=kalman_on)
                    
                    # Test
                    test(model, model_config, lr=learning_rate, lr_id=lr_id, gm_name=gm_name, data_config=data_config, minmax_train=minmax, device=device, kalman=kalman_on)




if __name__=='__main__':
    # DO NOTHING: run relevant pipeline_single.py or pipeline_multi.py
    pass
