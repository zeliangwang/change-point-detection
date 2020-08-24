#%%

from IPython.core.pylabtools import figsize
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as ss
import pickle 


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


# Example 6: 7-Eleven Denmark (before & after COVID-19 lockdown in Denmark, 13/03/2020)
case_6a = {'store_name': '7-Eleven (049 - Vesterbrogade, Denmark)', 
            'date_rng': ['20200310', '20200316'], 
            'data_path': os.path.join('data', '7-eleven-denmark-energy&influences-20180601-20200531-12855.pkl'),
            'dc_id': '16105',  # Main incomer
            'dc_name': 'Main Incomer'}

data_sel = convert2count(case_6a, remove_nwh=False)


#%%
# Use Gamma distribution for the data
E_sel = data_sel.iloc[:, [0]]
plot_hist(E_sel, E_sel.columns.tolist()[0])  # plot histogram

# Force all values to positive in order to fit gamma distribution
E_sel_new = [1 if i <=0 else i for i in E_sel.iloc[:,0].values]

# fit Gamma distribution to the data
a, loc, scale = ss.gamma.fit(E_sel_new)  
x = np.linspace(ss.gamma.ppf(0.001, a, loc=loc, scale=scale), 
                ss.gamma.ppf(0.999, a, loc=loc, scale=scale), len(E_sel_new))

f, axes = plt.subplots(1, 1, figsize=(6, 4))
E_pdf = ss.gamma.pdf(x, a, loc=loc, scale=scale)
# # E_pdf = ss.gamma.pdf(x, a, scale=scale)
axes.plot(x, E_pdf,'r-', lw=2, alpha=0.8, label='Gamma pdf')

axes.hist(E_sel_new, density=True,
                alpha=0.5, color = 'blue', edgecolor = 'black');    
plt.legend(loc="upper right")
plt.show()

# %%
