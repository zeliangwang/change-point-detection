"""Using a Bayesian model to detect the change point in Eniscope metering data.

7 different examples are included:
    Exe_1a: aircon data channel from KFC in Bracknell (before and after ACE2 installation);
    Exe_1b: main incomer data channel from KFC in Bracknell (before and after ACE2 installation);
    Exe_2a: aircon data channel from KFC in Bracknell (before and after COVID-19 lockdown on 16/03/2020);
    Exe_2b: main incomer data channel from KFC in Bracknell (before and after COVID-19 lockdown on 16/03/2020);
    Exe_3a: main incomer data channel from KFC (Kota Kemuning, Malaysia) (before and after COVID-19 lockdown on 18/03/2020);
    Exe_4a: main incomer data channel from KFC (Kelana Jaya, Malaysia) (before and after COVID-19 lockdown on 18/03/2020);
    Exe_5a: main incomer data channel from 7-Eleven (050 - Ved Vesterport St, Denmark) (before and after COVID-19 lockdown on 13/03/2020);
    Exe_6a: main incomer data channel from 7-Eleven (049 - Vesterbrogade, Denmark) (before and after COVID-19 lockdown on 13/03/2020);

Assuming the energy data (in Wh) are all positive values, it can be modelled as 
Poisson distributions.

Created by Z. Wang
21/07/2020

"""
# %%
from IPython.core.pylabtools import figsize
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as ss
import pickle 
import os
import pymc3 as pm
import theano.tensor as tt

# global settings for figures
fig_params = {'legend.fontsize': 'large',
            #   'figure.figsize': (15, 5),
            'savefig.dpi':300,
            'axes.labelsize': 'x-large',
            'axes.titlesize':'xx-large',
            'xtick.labelsize':'large',
            'ytick.labelsize':'large'}
plt.rcParams.update(fig_params)

# Convert energy data into integer values (to fit Poisson distribution)
def convert2count(metadata, **kwargs):
    """Convert positive energy values to integers in order to fit Poisson distribution.
    
    Inputs:
    -------
        metadata: metadata for the chosen case studies, containing "store_name", "date_rng"
                    "data_path", "dc_id", "dc_name"
        kwargs:
            remove_nwh: enable whether to filter samples for non working hours, 
                        False by default
            hours: list of hours to be filtered, such as non-working hours

    Returns:
    --------
        df_sel: dataframe stores the energy data for the selected meter channel. 
    """
    # Update input arguments
    opts = {'remove_nwh': False,
                    'hours': []}  
    opts.update(kwargs)  

    df = pickle.load(open(metadata['data_path'], "rb"))  # load energy data from pickle file

    d1 = metadata['date_rng'][0]
    d2 = metadata['date_rng'][1]
    dcid = metadata['dc_id']
    df_sel = df[['datetime', f'E_{dcid}']].set_index('datetime')[d1:d2]

    # Create a new column for rounded data in order to fit Poisson distribution
    df_sel[f'E_{dcid}_new'] = df_sel[f'E_{dcid}'].apply(lambda x: 0 if x < 0 else x)  # force negative vals to zeros
    df_sel[f'E_{dcid}_new'] = df_sel[f'E_{dcid}_new'].round()  # round energy vals to integer

    # Filter out non-working hours
    if opts['remove_nwh'] == True and len(opts['hours']) > 0:
        working_hours = [h for h in np.arange(24) if h not in opts['hours']]
        df_sel = df_sel[df_sel.index.hour.isin(working_hours)]

    return df_sel

#
# Plot histogram for the data
def plot_hist(df, title):    
    sns.set(context='notebook', style='white', palette='muted', font='sans-serif', font_scale=1.2, color_codes=True)
    
    # Set up the matplotlib figure
    f, axes = plt.subplots(1, 1, figsize=(6, 4))
    f.suptitle(f'{title}',  y=1.01, fontsize=20)
    # plot histogram 

    # axes.hist(df.iloc[:,0].values, density=True, bins=12,
    #                 alpha=0.7, color = 'blue', edgecolor = 'black')  
    sns.distplot(df, color="m", ax=axes); 

    # plt.setp(axes, yticks=[])
    plt.tight_layout();
    plt.show();

    return f, axes

#
def plot_bar(data, dcid, dvname, **kwargs):
    """Plot the energy data as barchart from the selected data channel.

    Inputs:
    -------
        data: dataframe containing energy data
        dcid: data channel id
        dvname: data channel name
        kwargs: save_fig is set to False by default
                fig_name

    Returns:
    --------
        a saved .svg file if "save_fig" is set to True

    """

    # input options
    opts = {'save_fig': False, # set it True to save the figure
            'affix': '',
            }
    opts.update(kwargs)

    affix = opts['affix']
    fig_name = f'observed-eniscope-data-E_{dcid}{affix}.svg'
    count_data = data[f'E_{dcid}_new'].values
    n_count_data = len(count_data)

    # sns.set(font_scale=1.2)
    # sns.set_style("whitegrid", {'grid.linestyle': ':'})
    fig, ax = plt.subplots(figsize=(16, 5))

    # ax.bar(np.arange(n_count_data), count_data, color="#348ABD", label='Energy')
    ax.bar(data.index.strftime('%d/%m/%y %H:%M').tolist(), count_data, color="#348ABD", label=f'E_{dcid}')
    ax.set_title(f"Did the {dvname}'s energy usage pattern change over time?", y=1.05)
    ax.set_xlabel("Time", labelpad=8)
    ax.set_ylabel("Energy consumption [Wh]", labelpad=8)
    ax.set_xlim(-1, n_count_data)
    ax.grid(which='major', linestyle=':', linewidth='0.5', color='gray')
    ax.legend(loc="upper right")

    # set sparse xticks
    dt_ticks = data.index.strftime('%d/%m/%y %H:%M').tolist()[::4]
    # dt_ticklabels = [i.strftime('%d/%m') for i in dt_ticks]
    ax.set_xticks(dt_ticks) # set sparse xticks
    # ax.set_xticklabels(dt_ticklabels)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, horizontalalignment='right')  # rotate x ticks
    plt.tight_layout()

    # Save the fig
    if opts['save_fig'] == True:
        plt.savefig(f'./plots/{fig_name}', bbox_inches='tight')

    plt.show()

# Detect changepoint using MCMC
def detect_changepoint(data, dcid, **kwargs):
    """Detect the changepoint for the energy data using MCMC.

    Inputs:
    -------
        data: dataframe containing energy data
        dcid: data channel id
        kwargs: key words arguments
            save_pkl: Boolean, save_pkl is set to False by default
            draw: int, number of samples to draw for MCMC sampling
            tune: int, number of burnin samples
            model: str, choose either 'Poisson' or 'Gamma' as the data's distribution,
                    it is set to be 'Poisson' by default
            sampler:, str, choose MCMC sampling algorithm, 'Metropolis' by default
            chains: int, number of chains to run MCMC
            cores: int, number of chains to run in parallel. If None, set to the number of CPUs in the system, but at most 4.
            seed: int, random seed value, 123 by default
    Returns:
    --------

    """
    # input options
    opts = {'save_pkl': False,
            'draw': 10000,  # number of samples to draws
            'tune': 5000,  # number of burnin samples
            'model': 'Poisson',  # default distribution for observed data
            'sampler': 'Metropolis',  # default MCMC sampler
            'chains': 3,  # number of MCMC chains
            'cores': None,
            'affix': '',  # affix for the saved pickle file name
            'seed': 123
            }
    opts.update(kwargs)

    affix = opts['affix']
    if opts['model'] == 'Poisson':
        fname = f'posterior-dists-E_{dcid}-Poisson{affix}.pkl'  # pickle file name
        obs_data = data[f'E_{dcid}_new'].values
    elif opts['model'] == 'Gamma':
        fname = f'posterior-dists-E_{dcid}-Gamma{affix}.pkl'  # pickle file name
        obs_data_ = data[f'E_{dcid}'].values
        # Enforce data to be positive to fit Gamma distribution
        obs_data = [0.001 if i <=0 else i for i in obs_data_]
    else:
        pass
    num_records = len(obs_data)

    np.random.seed(opts['seed'])   # set random seed
    
    # Define a Bayesian model using PyMC3
    with pm.Model() as model:
        if opts['model'] == 'Poisson':
            # initial parameter value for Exp distribution
            alpha = 1.0/obs_data.mean() 

            # Prior
            lambda_1 = pm.Exponential("lambda_1", alpha)
            lambda_2 = pm.Exponential("lambda_2", alpha)
            tau = pm.DiscreteUniform("tau", lower=0, upper=num_records - 1)

            # Likelihood
            idx = np.arange(num_records) # Index
            lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)
            observation = pm.Poisson("obs", lambda_, observed=obs_data)

        elif opts['model'] == 'Gamma':
            # Fit Gamma distribution to the data to find initial vals for proposal distribution
            a, loc, scale = ss.gamma.fit(obs_data)

            # Prior
            a1 = pm.Exponential('a1', 1/a)  # shape parameter of Gamma distribution
            a2 = pm.Exponential('a2', 1/a) 
            b1 = pm.Exponential('b1', scale)  # rate parameter of Gamma distribution (rate=1/scale)
            b2 = pm.Exponential('b2', scale)  
            tau = pm.DiscreteUniform("tau", lower=0, upper=num_records - 1)

            # Likelihood
            idx = np.arange(num_records) # Index
            a_ = pm.math.switch(tau > idx, a1, a2)
            b_ = pm.math.switch(tau > idx, b1, b2)

            observation = pm.Gamma("obs", alpha=a_, beta=b_, observed=obs_data)

        else:
            pass

    # Inference using MCMC sampling
    with model:
        start = pm.find_MAP()  # To improve convergence, use MAP estimate (optimization) as the initial state for MCMC
        if opts['sampler'] == 'Metropolis':
            step = pm.Metropolis()  # Metropolis-Hasting 
        elif opts['sampler'] == 'NUTS':
            step = pm.NUTS()  # Hamiltonian MCMC with No U-Turn Sampler
        else:
            pass
        trace = pm.sample(draws=opts['draw'], tune=opts['tune'], step=step, start=start,
                        random_seed=opts['seed'], progressbar=True, chains=opts['chains'], cores=opts['cores'])

    # save posterior distributions
    if opts['save_pkl'] == True:
        with open(os.path.join('data', fname), 'wb') as f:
            pickle.dump([model, trace], f)

    return trace

# Plot posterior distributions of the estimated parameters
def plot_posterior(num_records, dcid, trace, **kwargs):
    """Plot the posterior distributions for the estimated parameters.

    Inputs:
    -------
        num_records: number of data records
        dcid: data channel id
        trace: PYMC3 trace object

        kwargs: save_fig, set to False by default
                fig_name

    Returns:
    --------
        a saved .svg file if "save_fig" is set to True

    """

    # input options
    opts = {'save_fig': False,
            'model': 'Poisson',  # distribution of the observed data, 'Poisson' by default
            'affix':''
            }
    opts.update(kwargs)
    affix = opts['affix']

    if opts['model'] == 'Poisson':
        fig_name = f'changepoint-posterior-E_{dcid}-Poisson{affix}.svg'

        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

        # samples from posterior distributions
        lambda_1_samples = trace['lambda_1']
        lambda_2_samples = trace['lambda_2']
        tau_samples = trace['tau']

        ax[0].hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
                    label="posterior of $\lambda_1$", color="#A60628", density=True)
        ax[0].legend(loc="upper left")
        # ax[0].set_title(r"""Posterior distributions of the variables $\lambda_1,\;\lambda_2,\;\tau$""")
        # ax[0].set_xlim(320,2500)
        ax[0].set_xlabel("$\lambda_1$ value", labelpad=8)
        ax[0].set_ylabel("Density", labelpad=8);
        ax[0].grid(which='major', linestyle=':', linewidth='0.3', color='gray')

        #
        ax[1].hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
                    label="posterior of $\lambda_2$", color="#7A68A6", density=True)
        ax[1].legend(loc="upper left")
        ax[1].set_xlabel("$\lambda_2$ value", labelpad=8)
        ax[1].set_ylabel("Density", labelpad=8);
        ax[1].grid(which='major', linestyle=':', linewidth='0.3', color='gray')

        #
        w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)  # weights
        ax[2].hist(tau_samples, bins=num_records, alpha=1,
                    label=r"posterior of $\tau$",
                    color="#467821", edgecolor= "#467821", weights=w, linewidth='2', rwidth=0)
        ax[2].set_xlabel(r"$\tau$ (in hours)", labelpad=8)
        ax[2].set_ylabel("Probability", labelpad=8)
        # ax[2].set_xlim([25, num_records-35])
        ax[2].legend(loc="upper left")
        ax[2].set_xticks(np.arange(num_records)[::6])
        ax[2].grid(which='major', linestyle=':', linewidth='0.3', color='gray')

        #
        fig.suptitle(r"""Posterior distributions of the variables $\lambda_1,\;\lambda_2,\;\tau$""", fontsize=20, y=1.02)
        plt.tight_layout()

    elif opts['model'] == 'Gamma':
        fig_name = f'changepoint-posterior-E_{dcid}-Gamma{affix}.svg'

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 9))
        ax_1D = ax.ravel()

        # samples from posterior distributions
        tau_samples =trace['tau']  # changepoint
        a1_samples  =trace['a1']  # Gamma shape parameter before changepoint
        a2_samples  =trace['a2']  # Gamma shape parameter after changepoint
        b1_samples  =trace['b1']  # Gamma rate parameter before changepoint
        b2_samples  =trace['b2']  # Gamma rate parameter after changepoint

        # 
        ax_1D[0].hist(a1_samples, histtype='stepfilled', bins=30, alpha=0.85,
                    label="posterior of $a_1$", color="#A60628", density=True)
        ax_1D[0].legend(loc="upper left")
        ax_1D[0].set_xlabel("$a_1$ value", labelpad=8)
        ax_1D[0].set_ylabel("Density", labelpad=8);
        ax_1D[0].grid(which='major', linestyle=':', linewidth='0.3', color='gray')

        #
        ax_1D[1].hist(a2_samples, histtype='stepfilled', bins=30, alpha=0.85,
                    label="posterior of $a_2$", color="#A60628", density=True)
        ax_1D[1].legend(loc="upper left")
        ax_1D[1].set_xlabel("$a_2$ value", labelpad=8)
        ax_1D[1].set_ylabel("Density", labelpad=8);
        ax_1D[1].grid(which='major', linestyle=':', linewidth='0.3', color='gray')

        #
        ax_1D[2].hist(b1_samples, histtype='stepfilled', bins=30, alpha=0.85,
                    label="posterior of $b_1$", color="#7A68A6", density=True)
        ax_1D[2].legend(loc="upper left")
        ax_1D[2].set_xlabel("$b_1$ value", labelpad=8)
        ax_1D[2].set_ylabel("Density", labelpad=8);
        ax_1D[2].grid(which='major', linestyle=':', linewidth='0.3', color='gray')

        #
        ax_1D[3].hist(b2_samples, histtype='stepfilled', bins=30, alpha=0.85,
                    label="posterior of $b_2$", color="#7A68A6", density=True)
        ax_1D[3].legend(loc="upper left")
        ax_1D[3].set_xlabel("$b_2$ value", labelpad=8)
        ax_1D[3].set_ylabel("Density", labelpad=8);
        ax_1D[3].grid(which='major', linestyle=':', linewidth='0.3', color='gray')

        #
        w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)  # weights
        ax_1D[4].hist(tau_samples, bins=num_records, alpha=1,
                    label=r"posterior of $\tau$", color="#467821", 
                    edgecolor= "#467821", weights=w, linewidth='2', rwidth=0)
        ax_1D[4].set_xlabel(r"$\tau$ (in hours)", labelpad=8)
        ax_1D[4].set_ylabel("Probability", labelpad=8)
        # ax_1D[4].set_xlim(15, num_records-35)
        ax_1D[4].legend(loc="upper left")
        ax_1D[4].set_xticks(np.arange(num_records)[::6])
        ax_1D[4].grid(which='major', linestyle=':', linewidth='0.3', color='gray')

        #
        ax_1D[-1].axis('off')

        #
        fig.suptitle(r"""Posterior distributions of the variables $a_1,\;a_2,\;b_1,\;b_2,\;\tau$""", 
                    fontsize=20, y=1.03)
        plt.tight_layout()

    else:
        pass

    # Save the fig
    if opts['save_fig'] == True:
        plt.savefig(f'./plots/{fig_name}', bbox_inches='tight')
        
    plt.show()

# Plot changepoint results  
def plot_changepoint(data, dcid, trace, **kwargs):
    """Plot the detected changepoint and expected consumption.

    Inputs:
    -------
        data: dataframe containing energy data
        dcid: data channel id
        trace: PYMC3 trace object

        kwargs: save_fig, set to False by default
                model, specify the type of distribution for the observed data (Poisson or Gamma)
                dc_name, data channel name
                store_name, store name
                annotation_y_loc, change point text location along y axis

    Returns:
    --------
        a saved .svg file if "save_fig" is set to True

    """

    # Input options
    opts = {'save_fig': False,  # set it True to save the figure
            'model': 'Poisson',  # distribution of the observed data, 'Poisson' by default
            'dc_name':'',  # data channel name
            'store_name':'',  # store name
            'annotation_y_loc':0,  # change point text location along y axis
            'affix':''
            } 
    opts.update(kwargs)
    dc_name = opts['dc_name']
    store_name = opts['store_name']
    affix = opts['affix']
    p=5  # lower bound of percentile range

    if opts['model'] == 'Poisson':
        fig_name = f'estimated-changepoint-E_{dcid}-Poisson{affix}.svg'
        # select data column that fits Poisson distribution
        obs_data = data[f'E_{dcid}_new'].values

        # Assign posterior distributions from PyMC3 trace object
        tau_samples =trace['tau']  # changepoint
        lambda_1_samples  =trace['lambda_1']  # Gamma shape parameter before changepoint
        lambda_2_samples  =trace['lambda_2']  # Gamma shape parameter after changepoint
        N = tau_samples.shape[0]

        num_records = len(obs_data)
        expected_consumption = np.zeros(num_records)
        perc_rng = np.zeros((num_records,2))  # credible interval
        for day in range(0, num_records):
            # ix is a bool index of all tau samples corresponding to
            # the switchpoint occurring prior to value of 'day'
            ix = day < tau_samples
            expected_consumption[day] = (lambda_1_samples[ix].sum()
                                        + lambda_2_samples[~ix].sum()) / N

            v = np.concatenate([lambda_1_samples[ix], lambda_2_samples[~ix]])
            perc_rng[day,:] = np.percentile(v, [p, 100-p])  # 90% CI 

    elif opts['model'] == 'Gamma':
        fig_name = f'estimated-changepoint-E_{dcid}-Gamma{affix}.svg'
        # Enforce data to be positive to fit Gamma distribution
        obs_data_ = data[f'E_{dcid}'].values
        obs_data = [0.001 if i <=0 else i for i in obs_data_]  

        # Assign posterior distributions from PyMC3 trace object
        tau_samples =trace['tau']  # changepoint
        a1_samples  =trace['a1']  # Gamma shape parameter before changepoint
        a2_samples  =trace['a2']  # Gamma shape parameter after changepoint
        b1_samples  =trace['b1']  # Gamma rate parameter before changepoint
        b2_samples  =trace['b2']  # Gamma rate parameter after changepoint
        N = tau_samples.shape[0]

        # Calculate mean for Gamma distribution
        mu1 = np.divide(a1_samples, b1_samples)  # mean before the changepoint
        mu2 = np.divide(a2_samples, b2_samples)  # mean after the changepoint

        # Calculate expected consumptions
        num_records = len(obs_data)
        expected_consumption = np.zeros(num_records)
        perc_rng = np.zeros((num_records,2))  # credible interval
        for day in range(0, num_records):
            # ix is a bool index of all tau samples corresponding to
            # the switchpoint occurring prior to value of 'day'
            ix = day < tau_samples
            expected_consumption[day] = (mu1[ix].sum()
                                        + mu2[~ix].sum()) / N

            v = np.concatenate([mu1[ix], mu2[~ix]])    
            perc_rng[day,:] = np.percentile(v, [p, 100-p])  # 90% CI 
    else:
        pass

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))

    # Observed data
    ax.bar(np.arange(len(obs_data)), obs_data, color="#348ABD", alpha=0.85, zorder=0,
            label="observed consumption")

    # Expected value
    ax.plot(range(num_records), expected_consumption, lw=4, color="#E24A33", zorder=5,
            alpha = 0.95, label="expected consumption")

    # Mean of changepoint samples
    hour_idx = int(np.ceil(tau_samples.mean()))  # rounded to the ceil integer
    print('\nTime of change: {}\n'.format(hour_idx))
    ax.axvline(hour_idx, 
            linestyle = ':', linewidth = 4,
            label = "Time of change", 
            zorder=10,
            color = "black")

    # CI for expected values
    ax.fill_between(range(num_records), 
                    perc_rng[:,0], 
                    perc_rng[:,1], 
                    alpha = 0.25,
                    label = "90% Credible Interval",
                    zorder=15,
                    color = "orange")

    # CI for changepoint
    tau_ = np.percentile(tau_samples, [p, 100-p])  # 90% CI for tau
    ax.fill_betweenx([0, 1], 
                    np.floor(tau_[0]), # 5th percentile
                    np.ceil(tau_[1]),  # 95th percentile
                    alpha = 0.25,
                    label = "90% Credible Interval \n for tau",
                    zorder=20,
                    color = "green",
                    transform=ax.get_xaxis_transform())

    # Decoration
    style = dict(size=16, color='black')
    y_loc = opts['annotation_y_loc']
    change_time = data.index.strftime('%d-%b-%Y, %H:%M').tolist()[hour_idx]  # convert index to readable datetime
    ax.text(hour_idx+2, y_loc, change_time, transform=ax.transData, **style)  # annotation text

    ax.set_xlim(0, num_records)
    ax.set_xlabel("Time (in hours)")
    ax.set_ylabel("Energy [Wh]")
    ax.set_title(f"Expected energy consumption - {dc_name}, {store_name}", y=1.02)
    # plt.ylim(0, 60)
    ax.grid(which='major', linestyle=':', linewidth='0.3', color='gray')

    plt.legend(loc="upper right")
    plt.tight_layout()

    # Save the fig
    if opts['save_fig'] == True:
        plt.savefig(f'./plots/{fig_name}', bbox_inches='tight')

    plt.show()


# Case studies
# Example 1: KFC in Bracknell (before & after ACE2 installation)
case_1a = {'store_name': 'KFC (Bracknell, UK)', 
            'date_rng': ['20191101', '20191106'],  # installation of ACE2
            'data_path': os.path.join('data', 'kfc-uk-energy&influences-20191001-20200430-17227.pkl'),
            'dc_id': '59672',  # Aircon rest.
            'dc_name': 'Aircon Rest.'}
case_1b = {'store_name': 'KFC (Bracknell, UK)', 
            'date_rng': ['20191101', '20191106'],  # installation of ACE2
            'data_path': os.path.join('data', 'kfc-uk-energy&influences-20191001-20200430-17227.pkl'),
            'dc_id': '59668',  # Main incomer.
            'dc_name': 'Main Incomer'}

# Example 2: KFC in Bracknell (before & after COVID-19 lockdown, 16/03/2020)
case_2a = {'store_name': 'KFC (Bracknell, UK)', 
            'date_rng': ['20200312', '20200318'],  
            'data_path': os.path.join('data', 'kfc-uk-energy&influences-20191001-20200430-17227.pkl'),
            'dc_id': '59672',  # Aircon rest.
            'dc_name': 'Aircon Rest.'}
case_2b = {'store_name': 'KFC (Bracknell, UK)', 
            'date_rng': ['20200312', '20200318'], 
            'data_path': os.path.join('data', 'kfc-uk-energy&influences-20191001-20200430-17227.pkl'),
            'dc_id': '59668',  # Main incomer.
            'dc_name': 'Main Incomer'}

# Example 3: KFC Kota Kemuning (before & after COVID-19 lockdown in Malaysia, 18/03/2020)
case_3a = {'store_name': 'KFC (Kota Kemuning, Malaysia)', 
            'date_rng': ['20200316', '20200322'],  
            'data_path': os.path.join('data', 'kfc-malaysia-energy-20200221-20200712-17839.pkl'),
            'dc_id': '66982',  # Main incomer-1F.
            'dc_name': 'Main Incomer-1F'}

# Example 4: KFC Kelana Jaya, Malaysia (before & after COVID-19 lockdown in Malaysia, 18/03/2020)
case_4a = {'store_name': 'KFC (Kelana Jaya, Malaysia)', 
            'date_rng': ['20200316', '20200322'],  
            'data_path': os.path.join('data', 'kfc-malaysia-energy-20200221-20200712-17840.pkl'),
            'dc_id': '67040',  
            'dc_name': 'Main Incomer-1F'}

# Example 5: 7-Eleven Denmark (before & after COVID-19 lockdown in Denmark, 13/03/2020)
case_5a = {'store_name': '7-Eleven (050 - Ved Vesterport St, Denmark)', 
            'date_rng': ['20200310', '20200316'], 
            'data_path': os.path.join('data', '7-eleven-denmark-energy&influences-20180601-20200531-11818.pkl'),
            'dc_id': '15781',  # Main incomer
            'dc_name': 'Main Incomer'}

# Example 6: 7-Eleven Denmark (before & after COVID-19 lockdown in Denmark, 13/03/2020)
case_6a = {'store_name': '7-Eleven (049 - Vesterbrogade, Denmark)', 
            'date_rng': ['20200310', '20200316'], 
            'data_path': os.path.join('data', '7-eleven-denmark-energy&influences-20180601-20200531-12855.pkl'),
            'dc_id': '16105',  # Main incomer
            'dc_name': 'Main Incomer'}


#%%
# all_cases = [case_1a, case_1b, case_2a, case_2b, case_3a, case_4a, case_5a]
# for i, case in enumerate(all_cases):
#     print("\nExample {}: {}, DCID: {}".format(i+1, case['store_name'], case['dc_id']))
#     data_h = convert2count(case)
#     print(data_h.head())

# %%
###################################################################################
# Example 1a  Air con, KFC Bracknell
###################################################################################
# Convert energy values to integers in order to fit Poisson distribution
data_sel = convert2count(case_1a, remove_nwh=True, hours=[1,2,3,4,5,6,7,8])
print(data_sel.head())
dc_id = case_1a['dc_id']
dc_name = case_1a['dc_name']
store_name = case_1a['store_name']
num_records = len(data_sel[f'E_{dc_id}_new'])

# Plot the energy from the chosen data channel
plot_bar(data_sel, dc_id, dc_name, save_fig=False, affix = '-ACE2')

dist_type = 'Gamma'  # specify distribution type of the observed data

# trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=5000, model='Poisson',
#                             sampler='NUTS', chains=3, seed=123, save_pkl=True,
#                             affix = '-ACE2')

# trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=15000, model='Gamma',
#                             sampler='NUTS', chains=2, seed=123, save_pkl=True,
#                             affix = '-ACE2')


# Load the saved pkl file
fname = f'posterior-dists-E_{dc_id}-Gamma-ACE2.pkl'
[_model, trace] = pickle.load(open(os.path.join('data', fname), "rb"))

pm.traceplot(trace);  # trace plot
plot_posterior(num_records, dc_id, trace, model=dist_type, save_fig=True, affix = '-ACE2')
plot_changepoint(data_sel, dc_id, trace, model=dist_type,
                dc_name= dc_name, store_name= store_name, annotation_y_loc=3950,
                save_fig=True, affix = '-ACE2')

# %%
###################################################################################
# Example 1b
###################################################################################
# Convert energy values to integers in order to fit Poisson distribution
data_sel = convert2count(case_1b, remove_nwh=True, hours=[1,2,3,4,5,6,7,8])
print(data_sel.head())
dc_id = case_1b['dc_id']
dc_name = case_1b['dc_name']
store_name = case_1b['store_name']
num_records = len(data_sel[f'E_{dc_id}_new'])

# Plot the energy from the chosen data channel
plot_bar(data_sel, dc_id, dc_name, save_fig=False, affix = '-ACE2')

dist_type = 'Poisson'  # specify distribution type of the observed data

trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=5000, model='Poisson',
                            sampler='NUTS', chains=3, seed=123, save_pkl=True,
                            affix = '-ACE2')

# trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=15000, model='Gamma',
#                             sampler='NUTS', chains=2, seed=123, save_pkl=True,
#                             affix = '-ACE2')


# Load the saved pkl file
# fname = f'posterior-dists-E_{dc_id}-Gamma-ACE2.pkl'
# [_model, trace] = pickle.load(open(os.path.join('data', fname), "rb"))

pm.traceplot(trace);  # trace plot
plot_posterior(num_records, dc_id, trace, model=dist_type, save_fig=True, affix = '-ACE2')
plot_changepoint(data_sel, dc_id, trace, model=dist_type,
                dc_name= dc_name, store_name= store_name, annotation_y_loc=90000,
                save_fig=True, affix = '-ACE2')


# %%
###################################################################################
# Example 2a  
###################################################################################
# Convert energy values to integers in order to fit Poisson distribution
data_sel = convert2count(case_2a, remove_nwh=True, hours=[1,2,3,4,5,6,7,8])
print(data_sel.head())
dc_id = case_2a['dc_id']
dc_name = case_2a['dc_name']
store_name = case_2a['store_name']
num_records = len(data_sel[f'E_{dc_id}_new'])

# Plot the energy from the chosen data channel
plot_bar(data_sel, dc_id, dc_name, save_fig=False)

dist_type = 'Gamma'  # specify distribution type of the observed data

# trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=5000, model='Poisson',
#                             sampler='NUTS', chains=3, seed=123, save_pkl=True)

trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=15000, model='Gamma',
                            sampler='NUTS', chains=2, seed=123, save_pkl=True)


# Load the saved pkl file
# fname = f'posterior-dists-E_{dc_id}-Gamma.pkl'
# [_model, trace] = pickle.load(open(os.path.join('data', fname), "rb"))

pm.traceplot(trace);  # trace plot
plot_posterior(num_records, dc_id, trace, model=dist_type, save_fig=True)
plot_changepoint(data_sel, dc_id, trace, model=dist_type,
                dc_name= dc_name, store_name= store_name, annotation_y_loc=1500,
                save_fig=True)

# %%
###################################################################################
# Example 2b  
###################################################################################
# Convert energy values to integers in order to fit Poisson distribution
data_sel = convert2count(case_2b, remove_nwh=True, hours=[1,2,3,4,5,6,7,8])
print(data_sel.head())
dc_id = case_2b['dc_id']
dc_name = case_2b['dc_name']
store_name = case_2b['store_name']
num_records = len(data_sel[f'E_{dc_id}_new'])

# Plot the energy from the chosen data channel
plot_bar(data_sel, dc_id, dc_name, save_fig=False)

dist_type = 'Poisson'  # specify distribution type of the observed data

trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=5000, model='Poisson',
                            sampler='NUTS', chains=3, seed=123, save_pkl=True)

# trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=15000, model='Gamma',
#                             sampler='NUTS', chains=2, seed=123, save_pkl=True)


# Load the saved pkl file
# fname = f'posterior-dists-E_{dc_id}-Poisson.pkl'
# [_model, trace] = pickle.load(open(os.path.join('data', fname), "rb"))

pm.traceplot(trace);  # trace plot
plot_posterior(num_records, dc_id, trace, model=dist_type, save_fig=True)
plot_changepoint(data_sel, dc_id, trace, model=dist_type,
                dc_name= dc_name, store_name= store_name, annotation_y_loc=90000,
                save_fig=True)

# %%
###################################################################################
# Example 3a  
###################################################################################
# Convert energy values to integers in order to fit Poisson distribution
data_sel = convert2count(case_3a, remove_nwh=True, hours=[0,1,2,3,4,5,6,7])
print(data_sel.head())
dc_id = case_3a['dc_id']
dc_name = case_3a['dc_name']
store_name = case_3a['store_name']
num_records = len(data_sel[f'E_{dc_id}_new'])

# Plot the energy from the chosen data channel
plot_bar(data_sel, dc_id, dc_name, save_fig=False)

dist_type = 'Gamma'  # specify distribution type of the observed data

# trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=5000, model='Poisson',
#                             sampler='NUTS', chains=3, seed=123, save_pkl=True)

trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=15000, model='Gamma',
                            sampler='NUTS', chains=2, seed=123, save_pkl=True)


# Load the saved pkl file
# fname = f'posterior-dists-E_{dc_id}-Poisson.pkl'
# [_model, trace] = pickle.load(open(os.path.join('data', fname), "rb"))

pm.traceplot(trace);  # trace plot
plot_posterior(num_records, dc_id, trace, model=dist_type, save_fig=True)
plot_changepoint(data_sel, dc_id, trace, model=dist_type,
                dc_name= dc_name, store_name= store_name, annotation_y_loc=18000,
                save_fig=True)

# %%
###################################################################################
# Example 4a  
###################################################################################
# Convert energy values to integers in order to fit Poisson distribution
data_sel = convert2count(case_4a, remove_nwh=True, hours=[0,1,2,3,4,5,6,7])
print(data_sel.head())
dc_id = case_4a['dc_id']
dc_name = case_4a['dc_name']
store_name = case_4a['store_name']
num_records = len(data_sel[f'E_{dc_id}_new'])

# Plot the energy from the chosen data channel
plot_bar(data_sel, dc_id, dc_name, save_fig=False)

dist_type = 'Poisson'  # specify distribution type of the observed data

trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=5000, model='Poisson',
                            sampler='NUTS', chains=3, seed=123, save_pkl=True)

# trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=15000, model='Gamma',
#                             sampler='NUTS', chains=2, seed=123, save_pkl=True)

# Load the saved pkl file
# fname = f'posterior-dists-E_{dc_id}-Poisson.pkl'
# [_model, trace] = pickle.load(open(os.path.join('data', fname), "rb"))

pm.traceplot(trace);  # trace plot
plot_posterior(num_records, dc_id, trace, model=dist_type, save_fig=True)
plot_changepoint(data_sel, dc_id, trace, model=dist_type,
                dc_name= dc_name, store_name= store_name, annotation_y_loc=9000,
                save_fig=True)

# %%
###################################################################################
# Example 5a  7-Eleven in Denmark open for 24 hours
###################################################################################
# Convert energy values to integers in order to fit Poisson distribution
data_sel = convert2count(case_5a, remove_nwh=False)
print(data_sel.head())
dc_id = case_5a['dc_id']
dc_name = case_5a['dc_name']
store_name = case_5a['store_name']
num_records = len(data_sel[f'E_{dc_id}_new'])

# Plot the energy from the chosen data channel
plot_bar(data_sel, dc_id, dc_name, save_fig=False)

dist_type = 'Gamma'  # specify distribution type of the observed data

# trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=5000, model='Poisson',
#                             sampler='NUTS', chains=3, seed=123, save_pkl=True)

trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=15000, model='Gamma',
                            sampler='NUTS', chains=2, seed=123, save_pkl=True)

# Load the saved pkl file
# fname = f'posterior-dists-E_{dc_id}-Poisson.pkl'
# [_model, trace] = pickle.load(open(os.path.join('data', fname), "rb"))

pm.traceplot(trace);  # trace plot
plot_posterior(num_records, dc_id, trace, model=dist_type, save_fig=True)
plot_changepoint(data_sel, dc_id, trace, model=dist_type,
                dc_name= dc_name, store_name= store_name, annotation_y_loc=18000,
                save_fig=True)

# %%
###################################################################################
# Example 6a  7-Eleven in Denmark open for 24 hours
###################################################################################
# Convert energy values to integers in order to fit Poisson distribution
data_sel = convert2count(case_6a, remove_nwh=False)
print(data_sel.head())
dc_id = case_6a['dc_id']
dc_name = case_6a['dc_name']
store_name = case_6a['store_name']
num_records = len(data_sel[f'E_{dc_id}_new'])

# Plot the energy from the chosen data channel
plot_bar(data_sel, dc_id, dc_name, save_fig=False)

dist_type = 'Poisson'  # specify distribution type of the observed data

trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=5000, model='Poisson',
                            sampler='NUTS', chains=3, seed=123, save_pkl=True)

# trace = detect_changepoint(data_sel, dc_id, draw=10000, tune=15000, model='Gamma',
#                             sampler='NUTS', chains=2, seed=123, save_pkl=True)

# Load the saved pkl file
# fname = f'posterior-dists-E_{dc_id}-Poisson.pkl'
# [_model, trace] = pickle.load(open(os.path.join('data', fname), "rb"))

pm.traceplot(trace);  # trace plot
plot_posterior(num_records, dc_id, trace, model=dist_type, save_fig=True)
plot_changepoint(data_sel, dc_id, trace, model=dist_type,
                dc_name= dc_name, store_name= store_name, annotation_y_loc=21000,
                save_fig=True)
