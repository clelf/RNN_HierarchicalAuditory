import numpy as np
import time

import sys
import os
import scipy.stats as ss


import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm

from matplotlib.ticker import MaxNLocator

import random


# TODO: fill classes and methods descriptions


class AuditGenerativeModel:
    """Generic class for building a generative model.
    Not meant to be used on its own but provide reusable methods for contexts, process and observation states generation.
    NonHierachicalGenerativeGM and HierachicalGenerativeGM build on top of it.

    Attributes
    ----------
        N_batch: int; number of data batches to generate
        N_blocks: int; number of blocks a batch contains
        N_tones: int; number of tones per block (usually 8)
        tones_values: list-like; a set of tone frequencies to sample a pair from, to assign to the pair of standard and deviant tones
        mu_tau, si_tau, si_lim: float; define the linear Gaussian dynamics of the standard and deviant processes
        si_q: float; variance of the noise in the process LGD
        si_r: float; variance of the noise in the observation LGD

    """

    def __init__(self, params):

        self.N_batch = params["N_batch"]
        self.N_blocks = params["N_blocks"]
        self.N_tones = params["N_tones"]
        self.tones_values = params["tones_values"]
        self.mu_tau = params["mu_tau"]  # Std and dvt process
        self.si_tau = params["si_tau"]  # Std and dvt process
        self.si_lim = params["si_lim"]  # Std and dvt process
        self.si_q = params["si_q"]
        self.si_r = params["si_r"]

        if "N_ctx" not in params.keys(): self.N_ctx = 2 # context refers to being a std / dvt
        else: self.N_ctx = params["N_ctx"]

    # Auxiliary samplers from goin.coin.GenerativeModel

    def _sample_N_(self, mu, si, size=1):
        """Samples from a normal distribution

        Parameters
        ----------
        mu : float
            Mean of the normal distribution
        si : float
            Standard deviation of the normal distribution
        size  : int or tuple of int (optional)
            Size of samples

        Returns
        -------
        np.array
            samples
        """

        return np.array(ss.norm.rvs(mu, si, size))

    def _sample_TN_(self, a, b, mu, si, size=1):
        """Samples from a truncated normal distribution

        Parameters
        ----------
        a  : float
            low truncation point
        b  : float
            high truncation point
        mu : float
            Mean of the normal distribution before truncation (i.e, location)
        si : float
            Standard deviation of the normal distribution before truncation (i.e, size)
        size  : int or tuple of int (optional)
            Size of samples

        Returns
        -------
        np.array
            samples
        """

        return np.array(ss.truncnorm.rvs((a - mu) / si, (b - mu) / si, mu, si, size))

    def sample_uniform_choice(self, set_values):
        """ Sample a choice of one value from a set of possible values
        """
        return np.random.choice(set_values)

    def sample_uniform_set(self, set_values, N=2):
        """ Sample a set of N value choices without replacement from a set of possible values
        Note: len(values) should be > N
        """
        set = np.random.choice(set_values, size=(N,), replace=False)
        return set

    def sample_next_markov_state(
        self, current_state, states_values, states_trans_matrix
    ):
        return np.random.choice(states_values, p=states_trans_matrix[current_state])

    def sample_pi(self, N, mu_rho, si_rho):
        """A transition matrix with a sticky diagonal, controlled by the concentration parameter rho.
        rho is comprised between 0 and 1 and is is sampled from a truncated normal distribution of 
        mean mu_rho and standard deviation si_rho

        Parameters
        ----------
        mu_rho : float
            Mean parameter of the normal distribution to sample 
        si_rho : float
            _description_

        Returns
        -------
        np.array (N, N)
            Transition matrix
        """

        if N>1:
            # Sample parameters
            rho = self._sample_TN_(0, 1, mu_rho, si_rho).item()
            eps = [np.random.uniform() for n in range(N)]

            # Delta has a zero diagonal and the rest of the elements of a row (for a rule) are partitions from 1 using the corresponding eps[row] (parameter for that rule), controlling for the sum to be 1
            delta = np.array([[(0 if i == j else (eps[i] * (1 - eps[i]) ** j if (j < i and j < N - 2) else (eps[i] * (1 - eps[i]) ** (j - 1) if i < j < N - 1 else 1 - sum([eps[i] * (1 - eps[i]) ** k for k in range(N - 2)])))) for j in range(N)] for i in range(N)])
            
            # Transition matrix
            pi = rho * np.eye(N) + (1 - rho) * delta
        else:
            # if N==1:
            pi = np.eye(N)
        return pi

    def sample_contexts(self, N, N_ctx, mu_rho_ctx, si_rho_ctx, return_pi=False):
        """Samples a sequence of N events that can each be associated with a context out of of N_ctx values in range(N_ctx), and evolve
        through a Markov-chain manner with a transition matrix of parameters mu_rho_ctx and si_rho_ctx.

        Parameters
        ----------


        Returns
        -------
        list
            List of contexts for each of the N events
        """

        # Sample contexts transition matrix is sampled initially from a parametric distribution
        pi_ctx = self.sample_pi(N_ctx, mu_rho_ctx, si_rho_ctx)
        # pi_rules_0   = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]) # A very sticky transition matrix

        # Sequence of N contexts
        ctx = np.zeros(N, dtype=np.int64)

        # Initilize context (assign to 0, randomly, or from the distribution from which the transition probas also come from)
        ctx[0] = 0
        # rules[0] = np.random.choice(N_rules)

        for s in range(1, N):
            # Markov chain
            ctx[s] = self.sample_next_markov_state(
                current_state=ctx[s - 1],
                states_values=range(N_ctx),
                states_trans_matrix=pi_ctx,
            )

        if return_pi:
            return ctx, pi_ctx
        else:
            return ctx

    def sample_states(self, contexts, return_pars=False):
        """Generates a dictionary of data sequence for each context (std or dvt) dynamics given a sequence of contexts

        Here contexts is the sequence of tone-by-tone boolean value standing for {std: 0, dvt: 1} that has been
        hierarchically defined prior to the call of sample_states


        Parameters
        ----------
        contexts : integer np.array
            2-dimensional sequence of contexts filled with 0 or 1 (std or dvt), of size (N_blocks, N_tones)
        return_pars: bool
            also returns the time constant and sationary value (previously: retention and drift) parameters for each state at each block


        Returns
        -------
        states : dict
            dictionary encoding the hidden state values (one-dimensional np.array) for each
            context c (keys).
        pars (optional):
            tau: time constant parameter for each context (only if return_pars set to True)
            lim: stationary value parameter for each context
       
        # Note that time constant and sationary value (previously: retention and drift) are sampled in every call 

        """

        # Initialize arrays of params
        tau, lim = np.zeros((self.N_ctx, self.N_blocks)), np.zeros((self.N_ctx, self.N_blocks))  # self.N_ctx normally = 2 as for len({std, dvt})

        # Sample params for each block
        for b in range(self.N_blocks):
            # Sample one pair of std/dvt lim values for each block
            mu_lim_Cs = self.sample_uniform_set(self.tones_values) 
            
            for c in range(self.N_ctx):  # 2 contexts: std or dvt
                # Sample dynamics params for each context (std and dvt)
                tau[c, b] = self._sample_TN_(1, 50, self.mu_tau, self.si_tau).item()  # A high boundary
                lim[c, b] = self._sample_N_(mu_lim_Cs[c], self.si_lim).item()  # TODO: check values

        states = dict([(int(c), np.zeros(contexts.shape)) for c in range(self.N_ctx)])

        for c in range(self.N_ctx):  # self.N_ctx == 2

            # Initialize with a sample from distribution of mean and std the LGD stationary values
            # states[c][:,0] = self._sample_N_(d[c]/(1-a[c]), self.si_q/((1-a[c]**2)**.5), (contexts.shape[0], 1))
            states[c][:, 0] = self._sample_N_(lim[c, :], self.si_q * tau[c, :] / ((2 * tau[c, :] - 1) ** 0.5), (contexts.shape[0],))

            for b in range(self.N_blocks):

                # Sample noise
                w = self._sample_N_(0, self.si_q, contexts.shape)

                # Here the states exist independently of the contexts

                for t in range(1, contexts.shape[1]):
                    # states[c][:,t] = a[c] * states[c][:,t-1] + d[c] + w[:,t-1]
                    states[c][b, t] = states[c][b, t - 1] + 1 / tau[c, b] * (lim[c, b] - states[c][b, t - 1]) + w[b, t - 1]

        if return_pars:
            # return states, a, d
            return states, (tau.squeeze(), lim.squeeze())
        else:
            return states

    def sample_observations(self, contexts, states):
        """Generates a single data sequence y_t given a sequence of contexts c_t and a sequence of
        states x_t^c

        Parameters
        ----------
        contexts : integer np.array
            2-dimensional sequence of contexts filled with 0 or 1 (std or dvt), of size (N_blocks, N_tones)
        states : dict
            dictionary encoding the hidden state values (one-dimensional np.array) for each
            context c (keys).

        Returns
        -------
        y  : np.array
            2-dimensional sequence of observations of size (N_blocks, N_tones)
        """

        obs = np.zeros(contexts.shape)
        v = self._sample_N_(0, self.si_r, contexts.shape)

        for (s, t), c in np.ndenumerate(contexts):
            # Noisy observation of one of the two states (std or dvt), as imposed by the current context c
            obs[s, t] = states[c][s, t] + v[s, t]

        return obs

    def plot_contexts_states_obs(self, Cs, ys, y_stds, y_dvts, T, figsize=(10, 6)):
        """For a non-hierarchical situation (only contexts std/dvt, no rules)

        Parameters
        ----------
        Cs : _type_
            sequence of contexts
        ys : _type_
            observations
        y_stds : _type_
            states of std
        y_dvts : _type_
            states of dvt
        """

        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(y_stds, label="y_std", color="green", linestyle="dotted", linewidth=2)
        ax1.plot(y_dvts, label="y_dvt", color="blue", linestyle="dotted", linewidth=2)
        ax1.plot(ys, label="y", color="red", linestyle="dashed", linewidth=2)
        ax1.set_ylabel("y")

        ax2 = ax1.twinx()
        ax2.plot(range(T), Cs, "o", color="black", label="context")
        ax2.set_ylabel("context")
        ax2.set_yticks(ticks=[0, 1], labels=["std", "dvt"])

        fig.legend()

        fig.tight_layout()
        plt.show()

    def generate_batch(self, N_batch=None, return_pars=False):
        """Calls generate_run N_batch times and concatenates the return obsjects as (N_batch, object_size) size objects

        Parameters
        ----------
        N_batch : int, optional
            number of batches if None takes value defined upon init of instance, by default None
        return_pars : bool, optional
            to return the hidden states (individual std and dvt) dynamics parameters tau and lim for each block in each batch, by default False

        Returns
        -------
        objects as in generate_run
            rules, rules_long, dpos, timbres, timbres_long, contexts, states, obs(, pars) (im the case of HGM) // contexts, states, obs(, pars) (in the case of N-HGM)
        """

        # Store latent rules and timbres, states and observations from N_batch batches
        # TODO: find a better way to store batches

        if N_batch is None:
            N_batch = self.N_batch

        batches = []

        for batch in range(N_batch):
            # Generate a batch of N_blocks sequences, sampling parameters and generating the paradigm's observations
            # *res == rules, rules_long, dpos, timbres, timbres_long, contexts, states, obs(, pars) (HGM) // contexts, states, obs(, pars) (NHGM)
            res = self.generate_run(return_pars=return_pars)
            batches.append([*res])

        # Reorganize data as objects of size (N_batch, {obj_len}, 1) rather than a N_batch-long list of objects of size ({obj_len}, 1)
        # ! Except for the dictionary variable like states, that should keep the keys separate
        # res_reshaped = [np.stack([x for x in var_list], axis=0) for var_list in zip(*batches)]
        res_reshaped = (reshape_batch_variable(var_list) for var_list in zip(*batches))

        return res_reshaped


def reshape_batch_variable(var):
    if isinstance(var[0], dict):
        # Recursively process each dict key
        return {key: np.stack([batch[key] for batch in var], axis=0) for key in var[0]}
    else:
        # Direct array
        return np.stack([x for x in var], axis=0)


class NonHierachicalAuditGM(AuditGenerativeModel):
    """A generative model that only presents one level of context for a tone: to be a standard or a deviant tone
    Since data is not clustered in blocks defined by rules, there is only one "block" (N_block = 1)
    """

    def __init__(self, params):
        """
        Parameters
        ----------
        mu_rho_ctx :
            Mean of the truncated normal distribution to sample rho, the concentration (sticky) parameter of the transition matrix of contexts
        si_rho_ctx :
            Std of rho
        """

        super().__init__(params)
        self.N_blocks = 1
        self.mu_rho_ctx = params["mu_rho_ctx"]
        self.si_rho_ctx = params["si_rho_ctx"]

    def generate_run(
        self, N_tones=None, mu_rho_ctx=None, si_rho_ctx=None, return_pars=False
    ):
        """Generate data for one run of experiment: contexts, hidden states dynamics, observation

        Returns
        -------
        contexts:
            List of whether a tone is considered a dvt or a std --> contexts[t] = (current tone == dvt) (length = N_tones*N_blocks)
        states:
            List ynamics of both std (states[0]) and dvt (states[1]) at each "time step" (length = N_tones*N_blocks)
        obs:
            Observed tone at each time step (length = N_tones*N_blocks)
        pars: optional
            Time constant and sationary value parameters for each state at each block
        """

        if N_tones is None:     N_tones     = self.N_tones
        if mu_rho_ctx is None:  mu_rho_ctx  = self.mu_rho_ctx
        if si_rho_ctx is None:  si_rho_ctx  = self.si_rho_ctx

        # Get std/dvt contexts
        contexts = self.sample_contexts(
            N=N_tones, N_ctx=self.N_ctx, mu_rho_ctx=mu_rho_ctx, si_rho_ctx=si_rho_ctx
        )
        contexts = contexts.reshape((self.N_blocks, N_tones))

        # Sample states and observations
        if return_pars:
            states, pars = self.sample_states(contexts, return_pars)
        else:
            states = self.sample_states(contexts, return_pars)

        obs = self.sample_observations(contexts, states)

        # Flatten rules_long, contexts, (states, ) timbres and obs
        contexts = contexts.flatten()
        pass
        states = dict([(key, states[key].flatten()) for key in states.keys()])
        obs = obs.flatten()

        if return_pars:
                    return contexts, states, obs, pars
        else:
            return contexts, states, obs


class HierarchicalAuditGM(AuditGenerativeModel):

    def __init__(self, params):

        super().__init__(params)

        self.N_blocks = params["N_blocks"]
        self.N_rules = params["N_rules"]
        self.rules_dpos_set = params["rules_dpos_set"]
        self.mu_rho_rules = params["mu_rho_rules"]
        self.si_rho_rules = params["si_rho_rules"]
        self.mu_rho_timbres = params["mu_rho_timbres"]
        self.si_rho_timbres = params["si_rho_timbres"]

    def sample_rules(
        self, N_blocks, N_rules, mu_rho_rules, si_rho_rules, return_pi=False
    ):
        """Sample rules for a run consisting in a sequence of blocks of tones (each sequence being associated with one rule).
        Rules evolve in a Markov chain manner.

        Parameters
        ----------
        N_blocks : int
            Number of blocks of tones
        N_rules : int
            Number of rules
        mu_rho_rules : float
            _description_
        si_rho_rules : float
            _description_

        Returns
        -------
        np.array (N_blocks,)
            Sequence of rules associated with blocks for each block of N_blocks blocks
        """

        return self.sample_contexts(N=N_blocks, N_ctx=N_rules, mu_rho_ctx=mu_rho_rules, si_rho_ctx=si_rho_rules, return_pi=return_pi)

    def sample_timbres(self, rules_seq, N_timbres, mu_rho_timbres, si_rho_timbres):
        """Sample timbres, mediated by a Markov chain transition process too

        TODO: to check if correct

        Parameters
        ----------
        rules_seq : np.array
            _description_
        N_timbres : int
            _description_
        mu_rho_timbres : float
            _description_
        si_rho_timbres : float
            _description_

        Returns
        -------
        list
            _description_
        """

        # Sample timbres transition (emission from rule) matrix
        pi_timbre = self.sample_pi(N_timbres, mu_rho_timbres, si_rho_timbres)

        # timbres = np.array([np.random.choice(range(N_timbres), p=pi_timbre[seq]) for seq in rules_seq])
        timbres = np.array(
            [
                self.sample_next_markov_state(
                    current_state=seq,
                    states_values=range(N_timbres),
                    states_trans_matrix=pi_timbre,
                )
                for seq in rules_seq
            ]
        )

        return timbres

    def sample_dpos(self, rules, rules_dpos_set):
        """Sample positions of the deviant tones for each block of tones

        Parameters
        ----------
        rules : np.array
            Sequence of blocks rules_
        rules_dpos_set : np,array (N_rules, 3)
            Mapping of the 3 indexes of possible deviant positions for each of the rules

        Returns
        -------
        list
            List of deviant position indexes for each block of tones
        """

        # Given the trial's rule stored in rules and the possible positions for deviant associated with it,
        # sample the position where the deviant will be located in this trial
        # return [np.random.choice(rules_dpos_set[rule]) for rule in rules]
        return np.array(
            [self.sample_uniform_choice(rules_dpos_set[rule]) for rule in rules]
        )

    def get_contexts(self, dpos, N_blocks, N_tones):
        contexts = np.zeros((N_blocks, N_tones), dtype=np.int64)
        for i, pos in enumerate(dpos):
            contexts[i, pos] = 1
        return contexts

    def generate_run(
        self,
        N_blocks=None,
        N_rules=None,
        N_tones=None,
        rules_dpos_set=None,
        mu_rho_rules=None,
        si_rho_rules=None,
        mu_rho_timbres=None,
        si_rho_timbres=None,
        return_pars = False
    ):
        """Generate data for one run of experiment: rules, dvt positions, timbres, contexts (std or dvt),
        states (hidden states dynamics), observation

        Returns
        -------
        rules:
            List of the rules that apply to each block of N_tones tones --> rules[b] = rule for current block (length = N_blocks)
        rules_long:
            List of rules that apply to each tone in the whole list of tones --> rules_long[t] is the same for every 8 consecutive t (length = N_tones*N_blocks)
        dpos:
            Positions of the dvt within each block (length = N_blocks)
        timbres:
            List of timbres associated with each block (NOTE: dynamics correct but physical values TBD, not implemented ATM) (length = N_blocks)
        timbres_long:
            Same as in rules_long, per tone list of timbres (length = N_tones*N_blocks)
        contexts:
            List of whether a tone is considered a dvt or a std --> contexts[t] = (current tone == dvt) (length = N_tones*N_blocks)
        states:
            List ynamics of both std (states[0]) and dvt (states[1]) at each "time step" (length = N_tones*N_blocks)
        obs:
            Observed tone at each time step (length = N_tones*N_blocks)
        pars: optional
            Time constant and sationary value parameters for each state at each block
        """

        if N_blocks is None:        N_blocks = self.N_blocks
        if N_rules is None:         N_rules = self.N_rules
        if N_tones is None:         N_tones = self.N_tones
        if rules_dpos_set is None:  rules_dpos_set = self.rules_dpos_set
        if mu_rho_rules is None:    mu_rho_rules = self.mu_rho_rules
        if si_rho_rules is None:    si_rho_rules = self.si_rho_rules
        if mu_rho_timbres is None:  mu_rho_timbres = self.mu_rho_timbres
        if si_rho_timbres is None:  si_rho_timbres = self.si_rho_timbres

        # Sample sequence of rules ids
        rules = self.sample_rules(N_blocks, N_rules, mu_rho_rules, si_rho_rules)
        # Store latent rules in a per-tone array # This is equivalent to matlab's repmat
        rules_long = np.tile(rules[:, np.newaxis], (1, N_tones))

        # Sample timbres (here we consider that there are as many different timbres as there are different rules -- self.N_rules)
        timbres = self.sample_timbres(rules, N_rules, mu_rho_timbres, si_rho_timbres)
        # Store timbres in a per-tone array # This is equivalent to matlab's repmat
        timbres_long = np.tile(timbres[:, np.newaxis], (1, N_tones))

        # Sample deviant position
        dpos = self.sample_dpos(rules, rules_dpos_set)

        # Get contexts
        contexts = self.get_contexts(dpos, self.N_blocks, self.N_tones)

        # Sample states and observations
        # Sample states and observations
        if return_pars:
            states, pars = self.sample_states(contexts, return_pars)
        else:
            states = self.sample_states(contexts, return_pars)
        obs = self.sample_observations(contexts, states)

        # Flatten rules_long, contexts, (states, ) timbres and obs
        rules_long = rules_long.flatten()
        timbres_long = timbres_long.flatten()
        contexts = contexts.flatten()
        states = dict([(key, states[key].flatten()) for key in states.keys()])
        obs = obs.flatten()

        if return_pars:
            return rules, rules_long, dpos, timbres, timbres_long, contexts, states, obs, pars
        else:
            return rules, rules_long, dpos, timbres, timbres_long, contexts, states, obs

    def plot_contexts_rules_states_obs(self, y_stds, y_dvts, ys, Cs, rules, dpos):
        """For the hierachical evolution of rules and contexts (NOTE: timbres not included in this viz atm)

        Parameters
        ----------
        y_stds : _type_
            _description_
        y_dvts : _type_
            _description_
        ys : _type_
            _description_
        Cs : _type_
            _description_
        rules : _type_
            _description_
        """

        # Visualize tone frequencies
        fig, ax1 = plt.subplots(figsize=(20, 6))
        ax1.plot(
            y_stds,
            label="y_std",
            color="green",
            marker="o",
            markersize=4,
            linestyle="dotted",
            linewidth=2,
            alpha=0.5,
        )
        ax1.plot(
            y_dvts,
            label="y_dvt",
            color="blue",
            marker="o",
            markersize=4,
            linestyle="dotted",
            linewidth=2,
            alpha=0.5,
        )
        ax1.plot(
            ys,
            label="y",
            color="k",
            marker="o",
            markersize=4,
            linestyle="dashed",
            linewidth=2,
        )
        ax1.set_ylabel("y")

        ax2 = ax1.twinx()
        ax2.plot(Cs, "o", color="black", label="context", markersize=2)
        ax2.set_ylabel("context")
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_yticks(ticks=[0, 1], labels=["std", "dvt"])

        rules_cmap = {0: "tab:blue", 1: "tab:red", 2: "tab:orange"}
        for i, rule in enumerate(rules):
            plt.axvspan(
                i * self.N_tones,
                i * self.N_tones + self.N_tones,
                facecolor=rules_cmap[rule],
                alpha=0.25,
            )

        for i in range(self.N_blocks):
            plt.axvline(i * self.N_tones, color="tab:gray", linewidth=0.9)
            ax2.text(
                x=i * self.N_tones + 0.35 * self.N_tones,
                y=0.95,
                s=f"rule {rules[i]}",
                color=rules_cmap[rules[i]],
            )
            ax2.text(
                x=i * self.N_tones + 0.35 * self.N_tones,
                y=0.85,
                s=f"dvt {dpos[i]}",
                color=rules_cmap[rules[i]],
            )

        # , loc='upper left'bbox_to_anchor=(1.05, 1), frameon=False)
        fig.legend(bbox_to_anchor=(1.1, 1))
        # fig.tight_layout(rect=[0, 0, 0.85, 1])
        # fig.legend()
        plt.tight_layout()
        plt.show()

    def plot_rules_dpos(self, rules, dpos):

        # Visualize hierarchical information: dvt pos and rule

        rules_cmap = {0: "tab:blue", 1: "tab:red", 2: "tab:orange"}

        fig, ax = plt.subplots(figsize=(20, 6))
        for i, y in enumerate(dpos):
            ax.vlines(
                x=i,
                ymin=0,
                ymax=y,
                color="tab:gray",
                linewidth=0.9,
                zorder=1,
                alpha=0.5,
            )
        ax.scatter(
            range(len(dpos)), dpos, c=[rules_cmap[rule] for rule in rules], zorder=2
        )
        ax.set_ylabel("dvt pos")
        ax.set_xlabel("trial")
        ax.set_ylim(1, 8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(range(len(dpos)))

        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color, markersize=10
            )
            for color in rules_cmap.values()
        ]
        labels = rules_cmap.keys()
        ax.legend(handles, labels, title="rule")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":

    config_H = {
        "N_batch": 1,
        "N_blocks": 20,
        "N_tones": 8,
        "N_rules": 3,
        "rules_dpos_set": np.array([[2, 3, 4], [3, 4, 5], [4, 5, 6]]),
        "tones_values": [1455, 1500, 1600],
        "mu_tau": 4,
        "si_tau": 1,
        "si_lim": 5,
        "mu_rho_rules": 0.9,
        "si_rho_rules": 0.05,
        "mu_rho_timbres": 0.8,
        "si_rho_timbres": 0.05,
        "si_q": 2,  # process noise
        "si_r": 2,  # measurement noise
    }

    gm = HierarchicalAuditGM(config_H)

    rules, rules_long, dpos, timbres, timbres_long, contexts, states, obs = gm.generate_run()
    rules_, rules_long_, dpos_, timbres_, timbres_long_, contexts_, states_, obs_ = gm.generate_batch(N_batch=2) # calls generate_run N_batch times and concatenates the return obsjects as (N_batch, object_size) size objects

    # States, current blocks' rules, contexts based on rules and sampled deviant positions, and observations sampled from states based on context
    gm.plot_contexts_rules_states_obs(states[0][0 : gm.N_tones], states[1], obs, contexts, rules, dpos)

    # Deviant position for each rule
    gm.plot_rules_dpos(rules, dpos)

    # An example of the states and observation sampling for one block
    gm.plot_contexts_states_obs(
        contexts[0 : gm.N_tones],
        obs[0 : gm.N_tones],
        states[0][0 : gm.N_tones],
        states[1][0 : gm.N_tones],
        gm.N_tones,
    )

    config_NH = {
        "N_batch": 1,
        "N_blocks": 1,
        "N_tones": 160,
        "mu_rho_ctx": 0.9,
        "si_rho_ctx": 0.05,
        "tones_values": [1455, 1500, 1600],
        "mu_tau": 4,
        "si_tau": 1,
        "si_lim": 5,
        "si_q": 2,  # process noise
        "si_r": 2,  # measurement noise
    }

    gm_NH = NonHierachicalAuditGM(config_NH)

    contexts_NH, states_NH, obs_NH = gm_NH.generate_run()

    # States and observation sampled based on contexts
    gm_NH.plot_contexts_states_obs(contexts_NH, obs_NH, states_NH[0], states_NH[1], gm_NH.N_tones, figsize=(20, 6))

    contexts_NH, states_NH, obs_NH = gm_NH.generate_batch(N_batch=2)

    pass
