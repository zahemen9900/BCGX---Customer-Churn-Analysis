#!/usr/bin/env python3
"""Counterpart script for feature_engineering.ipynb."""

# %% Cell 1
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

sys.path.append('../src')
from preprocessing import clean_data
from feat_engineering import FeatureEngineering

# %% Cell 2
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(repo_root, "data")
df = pd.read_csv(os.path.join(data_dir, "clean_data_after_eda.csv"))

# %% Cell 3
df = clean_data(df)
df.head()

# %% Cell 4
df.info(memory_usage= 'deep')

# %% Cell 5
price_df0 = pd.read_csv(os.path.join(data_dir, "raw", "price_data.csv"))
price_df0 = clean_data(price_df0)
price_df = price_df0.copy()

price_df0.head()

# %% Cell 6
price_df.groupby(['id', 'price_date']).agg({'price_off_peak_var': 'mean', 'price_off_peak_fix': 'mean'})

# %% Cell 7
# # Group off-peak prices by companies and month
# monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg(
#     {'price_off_peak_var': 'mean', 'price_off_peak_fix': 'mean'}
#     ).reset_index()

# # Get january and december prices
# jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
# dec_prices = monthly_price_by_id.groupby('id').last().reset_index()

# # Calculate the difference
# diff = pd.merge(dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}), jan_prices.drop(columns='price_date'), on='id')
# diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
# diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
# diff = diff[['id', 'offpeak_diff_dec_january_energy','offpeak_diff_dec_january_power']]
# diff.head()

# %% Cell 8
monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg({
    'price_off_peak_var': 'mean',
    'price_off_peak_fix': 'mean'
}).reset_index()

jan_prices = monthly_price_by_id.groupby('id').first()
dec_prices = monthly_price_by_id.groupby('id').last()


diff = dec_prices.drop('price_date', axis = 1).join(jan_prices.drop('price_date', axis = 1), on = 'id', how = 'inner', lsuffix = '_d', rsuffix = '_j')
diff.head()

# %% Cell 9
diff['offpeak_diff_dec_january_energy'] = diff['price_off_peak_var_d'] - diff['price_off_peak_var_j']
diff['offpeak_diff_dec_january_power'] = diff['price_off_peak_fix_d'] - diff['price_off_peak_fix_j']

# diff = diff[['id', 'offpeak_diff_dec_january_energy', 'offpeak_diff_dec_january_power']]
diff.loc[:, ['offpeak_diff_dec_january_energy', 'offpeak_diff_dec_january_power']].head()

# %% Cell 10
# price_df['month'] = price_df['price_date'].dt.month
# price_df['day_of_week'] = price_df['price_date'].dt.dayofweek
# price_df['year'] = price_df['price_date'].dt.year
# price_df['day_of_year'] = price_df['price_date'].dt.dayofyear

# %% Cell 11
diff['offpeak_ratio_dec_january_energy'] = (diff['offpeak_diff_dec_january_energy'] + 1) / diff['offpeak_diff_dec_january_energy']
diff['offpeak_ratio_dec_january_power'] = (diff['offpeak_diff_dec_january_power'] + 1) / diff['offpeak_diff_dec_january_power']

features = diff.loc[:, ['offpeak_diff_dec_january_energy', 'offpeak_diff_dec_january_power', 'offpeak_ratio_dec_january_energy', 'offpeak_ratio_dec_january_power']]
features

# %% Cell 12
annual_avg = price_df.groupby('id').agg({
    'price_off_peak_var': 'mean',
    'price_off_peak_fix': 'mean'
})

annual_avg.rename(columns={
    'price_off_peak_var': 'avg_price_off_peak_var',
    'price_off_peak_fix': 'avg_price_off_peak_fix'
}, inplace=True)

features = features.join(annual_avg, on = 'id', how = 'left' )
print(features.shape)
features

# %% Cell 13
summer_prices = price_df[(price_df['price_date'].dt.month == 7) | (price_df['price_date'].dt.month == 8)]
winter_prices = price_df[(price_df['price_date'].dt.month == 12) | (price_df['price_date'].dt.month == 1)]

summer_prices.shape, winter_prices.shape

summer_avg = summer_prices.groupby('id').agg({
    'price_off_peak_var': 'mean',
    'price_off_peak_fix': 'mean'
})

winter_avg = winter_prices.groupby('id').agg({
    'price_off_peak_var': 'mean',
    'price_off_peak_fix': 'mean'
})

summer_avg.rename({
    'price_off_peak_var': 'price_off_peak_var_summer',
    'price_off_peak_fix': 'price_off_peak_fix_summer'
}, axis=1, inplace = True)

winter_avg.rename({
    'price_off_peak_var': 'price_off_peak_var_winter',
    'price_off_peak_fix': 'price_off_peak_fix_winter'
}, axis=1, inplace = True)


print(summer_avg.shape, winter_avg.shape)

# %% Cell 14
seasonal_diff = summer_avg.join(winter_avg, on='id', how = 'right', lsuffix = '_summer', rsuffix = '_winter')

seasonal_diff['summer_winter_diff_energy'] = seasonal_diff['price_off_peak_var_summer'] - seasonal_diff['price_off_peak_var_winter']
seasonal_diff['summer_winter_diff_power'] = seasonal_diff['price_off_peak_fix_summer'] - seasonal_diff['price_off_peak_fix_winter']

features = features.join(seasonal_diff[['summer_winter_diff_energy', 'summer_winter_diff_power']], on = 'id', how = 'left')
print(features.shape)
features

# %% Cell 15
price_df['price_off_peak_var_rolling'] = price_df.groupby('id')['price_off_peak_var'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
price_df['price_off_peak_fix_rolling'] = price_df.groupby('id')['price_off_peak_fix'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)

rolling_avg = price_df.groupby('id').agg({
    'price_off_peak_var_rolling': 'last',
    'price_off_peak_fix_rolling': 'last'
})

rolling_avg.rename(columns={
    'price_off_peak_var_rolling': 'rolling_3m_avg_price_off_peak_var',
    'price_off_peak_fix_rolling': 'rolling_3m_avg_price_off_peak_fix'
}, inplace=True)

features = features.join(rolling_avg, on = 'id', how = 'left')
print(features.shape)
features

# %% Cell 16
price_variability = price_df.groupby('id').agg({
    'price_off_peak_var': ['std', 'max', 'min'],
    'price_off_peak_fix': ['std', 'max', 'min']
})

price_variability.columns = ['std_price_off_peak_var', 'max_price_off_peak_var', 'min_price_off_peak_var',
                                'std_price_off_peak_fix', 'max_price_off_peak_fix', 'min_price_off_peak_fix']

features = features.join(price_variability, on = 'id', how = 'left')
features

# %% Cell 17
# price_df.set_index('id', inplace = True)
price_df = price_df.join(features, on = 'id', how = 'left')
price_df

# %% Cell 18
for feature in price_df.columns:
    print(feature, '>>>', price_df[feature].nunique())

# %% Cell 19
df

# %% Cell 20
price_df = price_df.drop('price_date', axis = 1)

# %% Cell 21
df['tenure'] = ((df['date_end'] - df['date_activ']).dt.days / 365.25).astype(int)
df['tenure'].value_counts()

# %% Cell 22
df.groupby('tenure').agg({'churn': 'mean'}).sort_values(by = 'churn', ascending = False)

# %% Cell 23
def convert_months(reference_date, df, column):
    """
    Input a column with timedeltas and return months.
    """
    # Calculate the time delta in days
    time_delta = (reference_date - df[column]).dt.days

    # Convert days to months (average month length is approximately 30.44 days)
    months = (time_delta / 30.44).astype(int)
    return months

# %% Cell 24
# Create reference date
reference_date = datetime(2016, 1, 1)

# Create columns
df['months_activ'] = convert_months(reference_date, df, 'date_activ')
df['months_to_end'] = -convert_months(reference_date, df, 'date_end')
df['months_modif_prod'] = convert_months(reference_date, df, 'date_modif_prod')
df['months_renewal'] = convert_months(reference_date, df, 'date_renewal')

# %% Cell 25
# We no longer need the datetime columns that we used for feature engineering, so we can drop them
cols_to_remove = [
    'date_activ',
    'date_end',
    'date_modif_prod',
    'date_renewal'
]

df = df.drop(columns=cols_to_remove)
df.head()

# %% Cell 26
df['has_gas'] = df['has_gas'].map({'t': 1, 'f': 0})
df.groupby(['has_gas']).agg({'churn': 'mean'})

# %% Cell 27
# Transform into categorical type
df['channel_sales'] = df['channel_sales'].astype('category')

# Let's see how many categories are within this column
df['channel_sales'].value_counts()

# %% Cell 28
df = pd.get_dummies(df, columns=['channel_sales'], prefix='channel')
df = df.drop(['channel_sddiedcslfslkckwlfkdpoeeailfpeds', 'channel_epumfxlbckeskwekxbiuasklxalciiuu', 'channel_fixdbufsefwooaasfcxdxadsiekoceaa'], axis = 1)
df.head()

# %% Cell 29
# Transform into categorical type
df['origin_up'] = df['origin_up'].astype('category')

# Let's see how many categories are within this column
df['origin_up'].value_counts()

# %% Cell 30
df = pd.get_dummies(df, columns = ['origin_up'], prefix = 'origin_up')
df = df.drop(columns=['origin_up_MISSING', 'origin_up_usapbepcfoloekilkwsdiboslwaxobdp', 'origin_up_ewxeelcelemmiwuafmddpobolfuxioce'])
df.head()

# %% Cell 31
skewed = [
    'cons_12m', 
    'cons_gas_12m', 
    'cons_last_month',
    'forecast_cons_12m', 
    'forecast_cons_year', 
    'forecast_discount_energy',
    'forecast_meter_rent_12m', 
    'forecast_price_energy_off_peak',
    'forecast_price_energy_peak', 
    'forecast_price_pow_off_peak'
]

df[skewed].describe()

# %% Cell 32
# Apply log10 transformation
df["cons_12m"] = np.log10(df["cons_12m"] + 1)
df["cons_gas_12m"] = np.log10(df["cons_gas_12m"] + 1)
df["cons_last_month"] = np.log10(df["cons_last_month"] + 1)
df["forecast_cons_12m"] = np.log10(df["forecast_cons_12m"] + 1)
df["forecast_cons_year"] = np.log10(df["forecast_cons_year"] + 1)
df["forecast_meter_rent_12m"] = np.log10(df["forecast_meter_rent_12m"] + 1)
df["imp_cons"] = np.log10(df["imp_cons"] + 1)

# %% Cell 33
df[skewed].describe()

# %% Cell 34

fig, axs = plt.subplots(nrows=3, figsize=(18, 20))
# Plot histograms
sns.histplot((df["cons_12m"].dropna()), ax=axs[0])
sns.histplot((df[df["has_gas"]==1]["cons_gas_12m"].dropna()), ax=axs[1])
sns.histplot((df["cons_last_month"].dropna()), ax=axs[2])
plt.show()

# %% Cell 35
df.info(memory_usage='deep')

# %% Cell 36
df_corr = df.corr(numeric_only=True)

df_corr

# %% Cell 37
plt.figure(figsize=(45, 45))
sns.heatmap(
    df_corr, 
    xticklabels=df_corr.columns.values,
    yticklabels=df_corr.columns.values, 
    annot=True, 
    annot_kws={'size': 12}
)
# Axis ticks size
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# %% Cell 38
# Dropping cols with high correlation
df = df.drop(['num_years_antig', 'forecast_cons_year', 'months_activ','has_gas'], axis = 1)
df

# %% Cell 39

print(df.isnull().any().sum())

# %% Cell 40
price_corr = price_df.corr(numeric_only=True)

plt.figure(figsize=(45, 45))
sns.heatmap(
    price_corr, 
    xticklabels=price_corr.columns.values,
    yticklabels=price_corr.columns.values, 
    annot=True, 
    annot_kws={'size': 12}
)
# Axis ticks size
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# %% Cell 41
price_df.columns

# %% Cell 42
# drop columns with high correlations

price_df = price_df.drop(
    ['rolling_3m_avg_price_off_peak_var', 'price_off_peak_var_rolling', 'avg_price_off_peak_var', 'min_price_off_peak_var', 'max_price_off_peak_var', \
     'price_off_peak_fix', 'price_off_peak_fix_rolling', 'avg_price_off_peak_fix', 'rolling_3m_avg_price_off_peak_fix']
    , axis = 1)

price_df

# %% Cell 43
# Assuming price_df is already defined
df_e = pd.read_csv(os.path.join(data_dir, "clean_data_after_eda.csv"))
price_df_e = pd.read_csv(os.path.join(data_dir, "raw", "price_data.csv"))

df_e = clean_data(df_e)
price_df_e = clean_data(price_df_e)


pipeline = FeatureEngineering(price_df_e)
enriched_price_df = pipeline.run_feature_engineering()
df_e, enriched_price_df = pipeline.wrangle_final(df_e, enriched_price_df)

# final_features now contains all the engineered features

# %% Cell 44
print(df_e)
print(enriched_price_df)

# %% Cell 45
df_enriched = enriched_price_df.merge(df_e, on = 'id', how = 'inner')
df_enriched

# %% Cell 46
df_enriched.info(memory_usage='deep')

# %% Cell 47
df_enriched[df_enriched.select_dtypes(include = np.float64).columns] = df_enriched[df_enriched.select_dtypes(include = np.float64).columns].astype(np.float32)
df_enriched.info(memory_usage='deep')

# %% Cell 48
df_enriched.to_csv(os.path.join(data_dir, "final_merged_features.csv"), index=False)
