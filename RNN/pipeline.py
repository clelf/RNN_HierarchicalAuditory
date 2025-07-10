from model import SimpleRNN, VAE, VRNN
import os
from pathlib import Path
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paradigm.audit_gm import NonHierachicalAuditGM, HierarchicalAuditGM
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Kalman.kalman import kalman_tau, plot_estim, kalman_batch


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





def train(model, n_batches, num_epochs, batch_res, epoch_res, optimizer, lr, lr_id, gm_name, data_config, device=DEVICE, kalman=False):
    # Pass model to GPU or CPU
    model.to(device)

    # As a check:
    if data_config["N_ctx"] > 1: kalman=False
    
    # Define data generative model
    if gm_name == 'NonHierarchicalGM': gm = NonHierachicalAuditGM(data_config)
    elif gm_name == 'HierarchicalGM': gm = HierarchicalAuditGM(data_config)
    else: raise ValueError("Invalid GM name")

    # Prepare to save the results
    lr_title = f"Learning rate: {lr:>6.0e}"
    if kalman:
        save_path = Path(f'/home/clevyfidel/Documents/Workspace/RNN_paradigm/training_results/N_ctx_{data_config["N_ctx"]}/kalman/{model.name}/')
    else:
        save_path = Path(f'/home/clevyfidel/Documents/Workspace/RNN_paradigm/training_results/N_ctx_{data_config["N_ctx"]}/observ/{model.name}/')
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
        kalman_mse_report   = []

    # Define loss instance
    loss_function   = torch.nn.GaussianNLLLoss(reduction='sum') # GaussianNLL

    # Track learning of parameters 
    param_updates   = []
    param_names     = model.state_dict().keys()

    for epoch in range(num_epochs):

        ### TRAINING
        train_losses = []
        tt = time.time()
        model.train()

        for batch in range(n_batches):

            optimizer.zero_grad()

            # Generate data
            _, states, y = gm.generate_batch(N_batch=batch_size)
            y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(DEVICE)
            if kalman: # and if data_config["N_ctx"] == 1
                states = torch.tensor(states[0], dtype=torch.float, requires_grad=False).unsqueeze(2).to(DEVICE)

            # Transform data
            minmax_train = MinMaxScaler()
            y_norm = minmax_train.fit_normalize(y, margin=data_config["si_r"]+data_config["si_q"])

            # Call model
            model_output = model(y_norm) # can contain output(, mu_latent, logvar_latent(, mu_prior, logvar_prior))

            # Compute loss
            if kalman:
                # Transform data
                minmax_train_states = MinMaxScaler()
                states_norm = minmax_train_states.fit_normalize(states, margin=data_config["si_q"])
                # If learning the Kalman filter, train to estimate the hidden states
                loss = model.loss(states_norm, model_output, loss_func=loss_function)
            else:
                # Or train to estimate the observations
                loss = model.loss(y_norm, model_output, loss_func=loss_function)

            # Learning
            params_before = [p.detach().clone() for p in model.parameters()]
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            params_after = [p.detach().clone() for p in model.parameters()]

            


            # Logging and reporting
            if batch % batch_res == batch_res-1:
                train_steps.append(epoch * n_batches + batch)
                
                # Store loss
                train_losses_report.append(float(loss.detach().cpu().item()))
                loss_report = np.mean(train_losses[-batch_res:])
                sprint=f'LR: {lr:>6.0e}; epoch: {epoch:0>3}; batch: {batch:>3}; loss: {float(loss.detach().cpu().item()):>7.4f}; batch loss: {loss_report:7.4f}; time: {time.time()-tt:>.2f}; training step: {epoch * n_batches + batch}'
                logfilename = f'{save_path}/training_loss_lr{lr_id}.txt'
                with open(logfilename, 'a') as f:
                    f.write(f'{sprint}\n')
                tt = time.time()

                # Store evolution of parameters
                param_update = torch.stack([torch.mean((p_after-p_before)**2) for p_before, p_after in zip(params_before, params_after)])
                param_updates.append(param_update)
            
        avg_train_loss = np.mean(train_losses) # average loss for current epoch


        ### VALIDATION
        valid_losses = []
        valid_mse = []
        valid_sigma = []
        if kalman:
            kalman_mses = []

        model.eval()
        with torch.no_grad():
            for batch in range(n_batches):
                # Generate data
                _, states, y, pars = gm.generate_batch(N_batch=batch_size, return_pars=True)
                y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(DEVICE)
                if kalman:
                    states = torch.tensor(states[0], dtype=torch.float, requires_grad=False).unsqueeze(2).to(DEVICE)

                # Transform data
                # minmax_valid = MinMaxScaler()
                y_norm = minmax_train.normalize(y)

                # Call model
                model_output = model(y_norm) # can contain output_dist(, mu_latent, logvar_latent(, mu_prior, logvar_prior))

                # Compute loss
                if kalman:
                    # Transform data
                    states_norm = minmax_train_states.normalize(states)
                    # If learning the Kalman filter, train to estimate the hidden states
                    loss = model.loss(states_norm, model_output, loss_func=loss_function)
                else:
                    # Or train to estimate the observations
                    loss = model.loss(y_norm, model_output, loss_func=loss_function)                
                
                # Store valid loss
                valid_losses.append(float(loss.detach().item()))

                # Get estimated distribution
                if model.name=='vrnn': model_output_dist = model_output[0]
                else: model_output_dist = model_output
                estim_mu    = minmax_train.denormalize_mean(F.sigmoid(model_output_dist[...,0]))
                estim_var   = minmax_train.denormalize_var(F.softplus(model_output_dist[..., 1]) + 1e-6)
                estim_sigma = torch.sqrt(estim_var)
                valid_sigma.append(estim_sigma)

                # Reshape variables
                y = y.squeeze()
                if kalman:
                    states = states.squeeze()

                # Evaluate MSE or MAE to true value that the model was trained on (observations or hidden process states)
                if kalman:
                    mse = ((estim_mu-states)**2).mean()
                else:
                    mse = ((estim_mu-y)**2).mean()
                valid_mse.append(mse)
                
                if kalman:
                    # Compare with Kalman
                    kalman_mu, kalman_sigma = kalman_batch(y, pars, C=1, Q=data_config["si_q"], R=data_config["si_r"], x0s=y[...,0]) 
                    kalman_mse = ((kalman_mu-states.numpy())**2).mean()
                    kalman_mses.append(kalman_mse)

            # Save valid samples for this epoch
            if epoch % epoch_res == epoch_res-1:                         
                if kalman:
                    plot_samples(y, estim_mu, estim_sigma, save_path=f'{save_path}/samples/lr{lr_id}-epoch-{epoch:0>3}_samples', title=lr_title, kalman_mu=kalman_mu, kalman_sigma=kalman_sigma, states=states)
                else:
                    plot_samples(y, estim_mu, estim_sigma, save_path=f'{save_path}/samples/lr{lr_id}-epoch-{epoch:0>3}_samples', title=lr_title)


            avg_valid_loss = np.mean(valid_losses)
            avg_valid_mae = np.mean(valid_mse)
            avg_valid_sigma = np.mean(valid_sigma)
            if kalman:
                avg_kalman_mse = np.mean(kalman_mse)
        
        valid_mse_report.append(avg_valid_mae)
        valid_losses_report.append(avg_valid_loss) # avg loss for epoch (over batches)
        valid_steps.append(epoch*n_batches+n_batches)
        epoch_steps.append(epoch)
        valid_sigma_report.append(avg_valid_sigma)
        if kalman:
            kalman_mse_report.append(avg_kalman_mse)

        print(f"Model: {model.name:>4}, LR: {lr:>6.0e}, Epoch: {epoch:>3}, Training Loss: {avg_train_loss:>7.2f}, Valid Loss: {avg_valid_loss:>7.2f}")
        sprint=f'LR: {lr:>6.0e}; epoch: {epoch:>3}; batch: {batch:>3}; loss: {loss:>7.2f}; batch loss: {loss_report:7.2f}; MSE: {avg_valid_mae:>7.2f}; time: {time.time()-tt:>.2f}; training step: {epoch * n_batches + batch}'
        if kalman:
            sprint += f'; Kalman MSE: {avg_kalman_mse:>7.2f}'
        logfilename = save_path / f'training_log_lr{lr_id}.txt'
        with open(logfilename, 'a') as f:
            f.write(f'{sprint}\n')

    # Save training logs
    lossplotfile = save_path/f'loss_trainvalid_lr{lr_id}.png'
    plot_losses(train_steps, valid_steps, train_losses_report, valid_losses_report, x_label='Training steps', y_label='Loss', title=lr_title, save_path=lossplotfile)
    # plot_losses(train_steps, valid_steps, epoch_steps, train_losses_report, valid_losses_report, x_label='Training steps', y_label='Loss', title=lr_title, save_path=lossplotfile)
    plot_var(epoch_steps, valid_sigma_report, title=lr_title, save_path=save_path/f'variance_valid_lr{lr_id}.png')
    if kalman:
        plot_diff(epoch_steps, valid_mse_report, title=lr_title, save_path=save_path/f'mse_valid_lr{lr_id}.png', kalman_mse=kalman_mse_report)
    else:
        plot_diff(epoch_steps, valid_mse_report, title=lr_title, save_path=save_path/f'mse_valid_lr{lr_id}.png')


    # Save model's weights
    torch.save(model.state_dict(), f'{save_path}/lr{lr_id}_weights.pth')

    # Plot training weights updates
    param_updates = torch.stack(param_updates, dim=1)
    plot_weights(train_steps, param_updates, list(param_names), lr_title, save_path=save_path/f'param_updates_lr{lr_id}.png')

    return minmax_train
 

def plot_weights(train_steps, param_updates, names, title, save_path):
    plt.figure(figsize=(10, 5))
    for param in range(param_updates.shape[0]):
        plt.plot(train_steps, param_updates[param], label=names[param])
    plt.yscale('log')
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_diff(valid_steps, valid_diff, title, save_path, kalman_mse=None):
    # Plot MSE or MAE depending on what's given
    plt.figure(figsize=(10, 5))
    plt.plot(valid_steps, valid_diff, label='model')
    if kalman_mse is not None:
        plt.plot(valid_steps, kalman_mse, label='kalman')
    plt.xlabel("Epoch steps")
    plt.ylabel(f"Validation MSE")
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_var(valid_steps, valid_sigma, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(valid_steps, valid_sigma, label="valid std")
    plt.xlabel("Epoch steps")
    plt.ylabel("Valid variance (std)")
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


def plot_samples(obs, estim_mu, estim_sigma, save_path, title=None, kalman_mu=None, kalman_sigma=None, states=None):
    obs = obs.squeeze() # for safety
            
    # for some 8 samples out of the whole batch of length obs.size(0)
    N = min(8, obs.size(0))
    for id, i in enumerate(np.random.choice(range(N), size=(N,), replace=False)):
            
        plt.figure(figsize=(20, 6))

        # Plot hidden process states if provided (NOTE - caution: can only work if states is 1D, i.e. only 1 context)
        if states is not None:
            plt.plot(range(len(states[i])), states[i], label='x_hid', color='orange', linewidth=2)
        
        # Plot observation
        plt.plot(range(len(obs[i])), obs[i], color='tab:blue', label='y_obs')

        # Plot estimation as a distribution (with uncertainty)
        plt.plot(range(len(estim_mu[i])), estim_mu[i], color='k', label='y_hat')
        plt.fill_between(range(len(estim_mu[i])), estim_mu[i]-estim_sigma[i], estim_mu[i]+estim_sigma[i], color='k', alpha=0.2)
        
        # Plot Kalman estimation if provided
        if kalman_mu is not None and kalman_sigma is not None:
            plt.plot(range(len(kalman_mu[i])), kalman_mu[i], label='y_kal', color='green', linewidth=2)
            plt.fill_between(range(len(kalman_mu[i])), kalman_mu[i]-kalman_sigma[i], kalman_mu[i]+kalman_sigma[i], color='green', alpha=0.2)

        plt.legend()
        plt.title(title)
        plt.savefig(f'{save_path}_s{id}.png')
        plt.close()




if __name__=='__main__':

    # Define model dimensions
    # Input and output dimensions
    input_dim   = 1 # number of features in each observation: 1 --> need to write y_norm = y_norm.unsqueeze(-1)  at one point
    dim_out_obs = 2 # learn the sufficient statistics mu and var
    # RNN bit
    output_dim      = input_dim
    rnn_hidden_dim  = 8 # Give comparable number of units as number of data parameters # In VRNN paper: 2000
    rnn_n_layers    = 1 # from goin
    # VRNN specific
    latent_dim      = 8 # 16 # needs to be < input_dim --> in a VAE, yes, but in a VRNN, needs to be <input_dim + rnn_hidden_dim
    phi_x_dim       = rnn_hidden_dim
    phi_z_dim       = rnn_hidden_dim
    phi_prior_dim   = rnn_hidden_dim

    
    # Define training parameters
    num_epochs      = 10 # 200 # 150
    epoch_res       = 10
    batch_res       = 10    # Store and report loss every batch_res batches
    batch_size      = 32 # 128   # batch_size = N_batch too # TODO: maximize it
    n_batches       = 20 # 20
    # learning_rate   = 5e-4
    weight_decay    = 1e-5 

    # Define experiment parameters (non-hierarchical)
    n_trials    = 150 # Single tones
    gm_name = 'NonHierarchicalGM'

    config_NH = {
        # "N_ctx": 1,
        "N_batch": batch_size,
        "N_blocks": 1,
        "N_tones": n_trials,
        "mu_rho_ctx": 0.9,
        "si_rho_ctx": 0.05,
        "tones_values": [1455, 1500, 1600],
        "mu_tau": 4,
        "si_tau": 1,
        "si_lim": 5,
        "si_q": 2,  # process noise
        "si_r": 2,  # measurement noise
    }

    for nctx in [1, 2]:
        config_NH["N_ctx"]=nctx

        for kalman_on in [True, False]: # [True, False]
            for lr_id, learning_rate in enumerate([0.1, 0.2, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]):
                # if lr_id in [4]: #, 8, 9, 10]:
                for model_name in ['rnn']: # 'vrnn
                    # Define models
                    if model_name == 'rnn':
                        model   = SimpleRNN(x_dim=input_dim, output_dim=dim_out_obs, hidden_dim=rnn_hidden_dim, n_layers=rnn_n_layers, batch_size=batch_size, device=DEVICE)
                    else:
                        model   = VRNN(x_dim=input_dim, output_dim=dim_out_obs, latent_dim=latent_dim, phi_x_dim=phi_x_dim, phi_z_dim=phi_z_dim, phi_prior_dim=phi_prior_dim, rnn_hidden_states_dim=rnn_hidden_dim, rnn_n_layers=rnn_n_layers, batch_size=batch_size, device=DEVICE)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                    train(model, n_batches=n_batches, num_epochs=num_epochs, batch_res=batch_res, epoch_res=epoch_res, optimizer=optimizer, 
                        lr=learning_rate, lr_id=lr_id, gm_name=gm_name, data_config=config_NH, device=DEVICE, kalman=kalman_on)
