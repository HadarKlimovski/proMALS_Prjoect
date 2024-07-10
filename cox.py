#!/usr/bin/env python
# coding: utf-8

# # cox univariant

# In[15]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter, KaplanMeierFitter, statistics

import pickle
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold






from configs import IMMUNE_PROTEINS, KERATIN, MIN_FEATURES,COX_LIST
from utils import load_raw_proteomics_data, rename_columns, drop_protein, describe_proteins_and_samples, norm_and_log,prepare_data_to_model,describe_clinical_data

from survival_analysis_utils import cox_proportional_hazard_model




final_df = pd.read_csv("data_prossesing_final.csv").set_index('sample number')
data = prepare_data_to_model(final_df)




# In[17]:






# In[18]:



# In[19]:


# ### cox univariant - drop nan values - my code

# In[20]:


####provide me 19 sig results


data = data
cox_p = {}
concordance_index = {}
hazard_ratios = {}


protein_names = [col for col in data.columns if col not in ['Status dead=1',  'Survival_from_onset (months)']]
for protein_name in protein_names:
    
    cox_data = data.dropna(subset=[protein_name])
    cph = cox_proportional_hazard_model(cox_data[[protein_name, 'Status dead=1',  'Survival_from_onset (months)']],
                                        'Survival_from_onset (months)',
                                        'Status dead=1',
                                        protein_name,
                                        strata=None,
                                        covariate_groups=None,
                                        show_plot=False)
    
    p_value = cph.summary.loc[protein_name]['p']
    c_index = cph.concordance_index_
    hazard_ratio = cph.hazard_ratios_[protein_name]

    cox_p[protein_name] = p_value
    concordance_index[protein_name] = c_index
    hazard_ratios[protein_name] = hazard_ratio

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'p_value': pd.Series(cox_p),
    'concordance_index': pd.Series(concordance_index),
    'hazard_ratio': pd.Series(hazard_ratios)
})

# Display the DataFrame
results_df




# In[21]:


# Filter significant proteins
significant_proteins_df = results_df[results_df['p_value'] < 0.05]
significant_proteins_df


# In[22]:


significant_proteins_df.shape


# In[23]:


significant_proteins_df = significant_proteins_df.reset_index()
significant_proteins_df


# In[24]:


data


# #### Kaplan–Meier estimator 

# In[25]:


protein_medians = data.median()
protein_medians


# In[27]:


kmf = KaplanMeierFitter()
num_proteins = len(MIN_FEATURES)
num_cols = 4
num_rows = (num_proteins + num_cols - 1) // num_cols  # Calculate the number of rows needed

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 6, num_rows * 6), sharey=True)
axes = axes.flatten()  # Flatten the axes array for easy iteration

for i, protein in enumerate(MIN_FEATURES):
    ax = axes[i]
    
    # Create high and low expression groups based on the median
    median_threshold = data[protein].median()
    
    # Only consider the high and low groups
    high_group = data[data[protein] > median_threshold]
    low_group = data[data[protein] <= median_threshold]
    
    # Calculate the log-rank test p-value
    result = statistics.logrank_test(high_group['Survival_from_onset (months)'], 
                                     low_group['Survival_from_onset (months)'], 
                                     event_observed_A=high_group['Status dead=1'], 
                                     event_observed_B=low_group['Status dead=1'])
    p_value = result.p_value
    
    # Plot high expression group
    kmf.fit(durations=high_group['Survival_from_onset (months)'], 
            event_observed=high_group['Status dead=1'], 
            label='High Expression (> median)')
    kmf.plot_survival_function(ax=ax)
    
    # Plot low expression group
    kmf.fit(durations=low_group['Survival_from_onset (months)'], 
            event_observed=low_group['Status dead=1'], 
            label='Low Expression (≤ median)')
    kmf.plot_survival_function(ax=ax)
    
    ax.set_title(f'{protein}\nLog-rank test p-value: {p_value:.5f}')
    ax.set_xlabel('Time (months)')
    if i % num_cols == 0:
        ax.set_ylabel('Survival Probability')
    ax.legend()

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

