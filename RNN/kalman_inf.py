import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Kalman.kalman import kalman_A, kalman_tau, plot_estim, kalman_batch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paradigm.audit_gm import NonHierachicalAuditGM

if __name__=='__main__':

    data_config = config_NH = {
        "N_ctx": 1,
        "N_batch": 3,
        "N_blocks": 1,
        "N_tones": 150,
        "mu_rho_ctx": 0.9,
        "si_rho_ctx": 0.05,
        "tones_values": [1455, 1500, 1600],
        "mu_tau": 4,
        "si_tau": 1,
        "si_lim": 5,
        "si_q": 2,  # process noise
        "si_r": 2,  # measurement noise
    }


    gm = NonHierachicalAuditGM(data_config)

    return_pars = True
    if return_pars:
        _, _, ys, pars = gm.generate_batch(return_pars=return_pars)
    else:
        _, _, ys = gm.generate_batch(return_pars=return_pars)

    # for y, par in zip(ys, pars):
    #     # y_hat, s_hat = kalman_A(y, A=1, Q=2, R=2, x0=y[0], s0=0)
    #     y_hat, s_hat = kalman_tau(y, tau=par[0], x_lim=par[1], C=1, Q=data_config["si_q"], R=data_config["si_r"], x0=y[0], s0=0)
    #     plot_estim(y, y_hat, s_hat, process=None)

    y_hats, s_hats = kalman_batch(ys, pars, C=1, Q=data_config["si_q"], R=data_config["si_r"], x0s=ys[...,0])

    for y, y_hat, s_hat in zip(ys, y_hats, s_hats):
        plot_estim(y, y_hat, s_hat, process=None)
