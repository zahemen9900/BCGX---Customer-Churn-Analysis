#!/usr/bin/env python3
"""Counterpart script for EDA.ipynb."""

# %% Cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
try:
    from src.utils import save_plot 
except ImportError:
    sys.path.append('../src')
    from utils import save_plot
# Shows plots in jupyter notebook
# %matplotlib inline

# Set plot style
sns.set_color_codes('deep')
sns.set_theme(style='whitegrid', context='notebook')

#plot presets
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
# %config InlineBackend.figure_format = 'retina'

# %% Cell 2
print(os.getcwd())

# %% Cell 3
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
data_path = os.path.join(repo_root, "data", "raw")
client_df = pd.read_csv(os.path.join(data_path,'client_data.csv'))
price_df = pd.read_csv(os.path.join(data_path, 'price_data.csv'))

# %% Cell 4
client_df.head(3)

# %% Cell 5
price_df.head(3)

# %% Cell 6
client_df.info(memory_usage = 'deep')

# %% Cell 7
price_df.info(memory_usage='deep')

# %% Cell 8
for col in client_df.columns:
    if col.startswith('date'):
        client_df[col] = pd.to_datetime(client_df[col], errors= 'coerce')

client_df[client_df.select_dtypes('object').columns] = client_df[client_df.select_dtypes('object').columns].astype('str')

# %% Cell 9
client_df.describe()

# %% Cell 10
price_df.describe()

# %% Cell 11
def plot_stacked_bars(dataframe, title_, size_=(18, 10), rot_=0, legend_="upper right", labels=["Retention", "Churn"], save: bool = False):
    """
    Plot stacked bars with annotations, improved styling
    """
    # Set Seaborn style for better aesthetics
    sns.set_theme(style="whitegrid")
    
    # Create the stacked bar plot
    ax = dataframe.plot(
        kind="bar",
        stacked=True,
        figsize=size_,
        rot=rot_,
        title=title_,
        color=sns.color_palette("dark", len(labels))  # Use richer colors
    )
    
    # Annotate bars
    annotate_stacked_bars(ax, textsize=14)
    
    # Style adjustments
    plt.legend(labels, loc=legend_, fontsize=12, frameon=True, shadow=True)
    plt.ylabel("Company base (%)", fontsize=12)
    plt.xlabel("Categories", fontsize=12)
    plt.title(title_, fontsize=16, weight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    if save:
        save_plot()



def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=8):
    """
    Add value annotations to the bars with improved styling
    """
    for p in ax.patches:
        # Calculate annotation
        value = str(round(p.get_height(), 1))
        if value == "0.0":
            continue
        ax.annotate(
            value,
            ((p.get_x() + p.get_width() / 2) * pad, (p.get_y() + p.get_height() / 2) * pad),
            color=colour,
            size=textsize,
            ha="center",  # Horizontal alignment
            va="center",  # Vertical alignment
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="black", alpha=0.7)  # Background box for text
        )


def plot_distribution(dataframe, column, ax, bins_=50, save: bool = False):
    """
    Plot variable distribution in a stacked histogram of churned or retained companies with enhanced visuals
    """
    # Create a temporal dataframe with the data to be plotted
    temp = pd.DataFrame({
        "Retention": dataframe[dataframe["churn"] == 0][column],
        "Churn": dataframe[dataframe["churn"] == 1][column]
    })
    
    # Plot the histogram
    temp[["Retention", "Churn"]].plot(
        kind="hist",
        bins=bins_,
        ax=ax,
        stacked=True,
        color=sns.color_palette("colorblind", 2),  # Use richer colors
        alpha=0.85
    )
    
    # Style adjustments
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Distribution of {column}", fontsize=14, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.ticklabel_format(style="plain", axis="x")
    ax.legend(fontsize=10, frameon=True, shadow=True)

    if save:
        save_plot()

# %% Cell 12
churn = client_df[['id', 'churn']]
churn.columns = ['Companies', 'churn']
churn_total = churn.groupby(churn['churn']).count()
churn_percentage = churn_total / churn_total.sum() * 100
plot_stacked_bars(churn_percentage.transpose(), "Churning status", (5, 5), legend_="lower right")

# %% Cell 13
consumption = client_df[['id', 'cons_12m', 'cons_gas_12m', 'cons_last_month', 'imp_cons', 'has_gas', 'churn']]

fig, axs = plt.subplots(nrows=1, figsize=(18, 5))

plot_distribution(consumption, 'cons_12m', axs)

# %% Cell 14
channel = client_df[['id', 'channel_sales', 'churn']]
channel = channel.groupby(['channel_sales', 'churn']).count()
channel

# %% Cell 15
channel = channel.unstack(level = 1)
channel = channel.fillna(0)
channel

# %% Cell 16
channel.div(channel.sum(axis=1), axis=0) * 100

# %% Cell 17
channel = client_df[['id', 'channel_sales', 'churn']]
channel = channel.groupby(['channel_sales', 'churn'])['id'].count().unstack(level=1).fillna(0)
channel_churn = (channel.div(channel.sum(axis=1), axis=0) * 100).sort_values(by=[1], ascending=False)

# %% Cell 18
plot_stacked_bars(channel_churn, 'Sales channel', rot_=30, save = True)

# %% Cell 19
consumption = client_df[['id', 'cons_12m', 'cons_gas_12m', 'cons_last_month', 'imp_cons', 'has_gas', 'churn']]

# %% Cell 20
def plot_distribution(dataframe, column, ax, bins_=50, save: bool = False):
    """
    Plot variable distribution in a stacked histogram of churned or retained company
    """
    # Create a temporal dataframe with the data to be plot
    temp = pd.DataFrame({"Retention": dataframe[dataframe["churn"]==0][column],
    "Churn":dataframe[dataframe["churn"]==1][column]})
    # Plot the histogram
    temp[["Retention","Churn"]].plot(kind='hist', bins=bins_, ax=ax, stacked=True)
    # X-axis label
    ax.set_xlabel(column)
    # Change the x-axis to plain style
    ax.ticklabel_format(style='plain', axis='x')
    if save:
        save_plot()

# %% Cell 21
fig, axs = plt.subplots(nrows=4, figsize=(18, 25))

plot_distribution(consumption, 'cons_12m', axs[0])
plot_distribution(consumption[consumption['has_gas'] == 't'], 'cons_gas_12m', axs[1])
plot_distribution(consumption, 'cons_last_month', axs[2])
plot_distribution(consumption, 'imp_cons', axs[3])

# %% Cell 22
fig, axs = plt.subplots(nrows=4, figsize=(18,25))

# Plot histogram
sns.boxplot(consumption["cons_12m"], ax=axs[0])
sns.boxplot(consumption[consumption["has_gas"] == "t"]["cons_gas_12m"], ax=axs[1])
sns.boxplot(consumption["cons_last_month"], ax=axs[2])
sns.boxplot(consumption["imp_cons"], ax=axs[3])

# Remove scientific notation
for ax in axs:
    ax.ticklabel_format(style='plain', axis='x')
    # Set x-axis limit
    axs[0].set_xlim(-200000, 2000000)
    axs[1].set_xlim(-200000, 2000000)
    axs[2].set_xlim(-20000, 100000)
plt.show()

# %% Cell 23
forecast = client_df[
    ["id", "forecast_cons_12m",
    "forecast_cons_year","forecast_discount_energy","forecast_meter_rent_12m",
    "forecast_price_energy_off_peak","forecast_price_energy_peak",
    "forecast_price_pow_off_peak","churn"
    ]
]

# %% Cell 24
fig, axs = plt.subplots(nrows=7, figsize=(18,50))

# Plot histogram
plot_distribution(client_df, "forecast_cons_12m", axs[0])
plot_distribution(client_df, "forecast_cons_year", axs[1])
plot_distribution(client_df, "forecast_discount_energy", axs[2])
plot_distribution(client_df, "forecast_meter_rent_12m", axs[3])
plot_distribution(client_df, "forecast_price_energy_off_peak", axs[4])
plot_distribution(client_df, "forecast_price_energy_peak", axs[5])
plot_distribution(client_df, "forecast_price_pow_off_peak", axs[6])

# %% Cell 25
contract_type = client_df[['id', 'has_gas', 'churn']]
contract = contract_type.groupby(['churn', 'has_gas'])['id'].count().unstack(level=0)
contract_percentage = (contract.div(contract.sum(axis=1), axis=0) * 100).sort_values(by=[1], ascending=False)

# %% Cell 26
plot_stacked_bars(contract_percentage, 'Contract type (with gas)')

# %% Cell 27
margin = client_df[['id', 'margin_gross_pow_ele', 'margin_net_pow_ele', 'net_margin']]

# %% Cell 28
fig, axs = plt.subplots(nrows=3, figsize=(18,20))
# Plot histogram
sns.boxplot(margin["margin_gross_pow_ele"], ax=axs[0], orient = 'h')
sns.boxplot(margin["margin_net_pow_ele"],ax=axs[1], orient = 'h')
sns.boxplot(margin["net_margin"], ax=axs[2], orient = 'h')
# Remove scientific notation
axs[0].ticklabel_format(style='plain', axis='x')
axs[1].ticklabel_format(style='plain', axis='x')
axs[2].ticklabel_format(style='plain', axis='x')
plt.show()

# %% Cell 29
power = client_df[['id', 'pow_max', 'churn']]

# %% Cell 30
fig, axs = plt.subplots(nrows=1, figsize=(18, 10))
plot_distribution(power, 'pow_max', axs)

# %% Cell 31
others = client_df[['id', 'nb_prod_act', 'num_years_antig', 'origin_up', 'churn']]
products = others.groupby([others["nb_prod_act"],others["churn"]])["id"].count().unstack(level=1)
products_percentage = (products.div(products.sum(axis=1), axis=0)*100).sort_values(by=[1], ascending=False)

# %% Cell 32
plot_stacked_bars(products_percentage, "Number of products", save = True)

# %% Cell 33
years_antig = others.groupby([others["num_years_antig"],others["churn"]])["id"].count().unstack(level=1)
years_antig_percentage = (years_antig.div(years_antig.sum(axis=1), axis=0)*100)
plot_stacked_bars(years_antig_percentage, "Number years", save=True)

# %% Cell 34
origin = others.groupby([others["origin_up"],others["churn"]])["id"].count().unstack(level=1)
origin_percentage = (origin.div(origin.sum(axis=1), axis=0)*100)
plot_stacked_bars(origin_percentage, "Origin contract/offer", rot_= 30)

# %% Cell 35
fig, ax = plt.subplots()

# Create a pie chart
client_df.has_gas.value_counts().plot(
    kind='pie', 
    autopct='%1.1f%%',  # Add percentage labels
    startangle=90,  # Start at 90° for better readability
    ax=ax, 
    labels=["No Gas", "Has Gas"],  # Custom labels if `has_gas` is binary
    colors=["indianred", "steelblue"],
    shadow = True,
    explode = (0.01, 0.1)
)

# Set title and aspect ratio
ax.set_title("Gas Subscription Distribution")
ax.set_ylabel('')  # Remove default y-axis label for a cleaner look
ax.axis('equal')  # Equal aspect ratio to make the pie chart circular

save_plot()
plt.show();

# %% Cell 36
numeric_cols = client_df.select_dtypes(include = np.number).columns.tolist()
grouped_means = client_df.groupby('churn')[numeric_cols].mean()
grouped_means

# %% Cell 37
grouped_means = grouped_means.drop('churn', axis = 1).T.fillna(0.0, axis = 0)
grouped_means

# %% Cell 38
grouped_means_percentage = grouped_means.div(grouped_means.sum(axis=1), axis = 0) * 100
grouped_means_percentage

# %% Cell 39
plot_stacked_bars(grouped_means_percentage, 'Feature Averages by Churn', rot_ = 60, save=True)

# %% Cell 40
plt.figure(figsize=(12, 8))
sns.heatmap(grouped_means, annot=True, fmt=".2f", cmap="rocket", cbar_kws={'label': 'Mean Value'})

# Customize plot
plt.title("Mean Values of Numeric Features by Churn Status", fontsize=16, fontweight='bold')
plt.xlabel("Churn Status", fontsize=12)
plt.ylabel("Numeric Features", fontsize=12)

# Rotate y-axis labels for readability
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.show()

# %% Cell 41
# Calculate differences (residuals) and percentage error
client_df_c = client_df.copy(deep = True)

client_df_c['electricity_residual'] = client_df_c['forecast_cons_12m'] - client_df_c['cons_12m']
client_df_c['electricity_percentage_error'] = (
    (client_df_c['forecast_cons_12m'] - client_df_c['cons_12m']) / client_df_c['cons_12m']
) * 100

# For gas (if applicable)
if 'cons_gas_12m' in client_df_c.columns and 'forecast_cons_12m' in client_df_c.columns:
    client_df_c['gas_residual'] = client_df_c['forecast_cons_12m'] - client_df_c['cons_gas_12m']
    client_df_c['gas_percentage_error'] = (
        (client_df_c['forecast_cons_12m'] - client_df_c['cons_gas_12m']) / client_df_c['cons_gas_12m']
    ) * 100

# %% Cell 42

plt.figure(figsize=(10, 6))
sns.lineplot(client_df_c['electricity_residual'], color="steelblue", label="Electricity Residuals")
if 'gas_residual' in client_df_c.columns:
    sns.lineplot(client_df_c['gas_residual'], color="indianred", label="Gas Residuals")

plt.title("Residual Distribution: Forecasted vs Real Consumption", fontsize=14, fontweight='bold')
plt.xlabel("Residual (Forecasted - Real)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
save_plot()
plt.show()

# %% Cell 43
# Extract year and month for temporal analysis
client_df['activation_year'] = client_df['date_activ'].dt.year
client_df['activation_month'] = client_df['date_activ'].dt.month

# %% Cell 44
# Aggregate yearly electricity and gas consumption
yearly_consumption = client_df.groupby('activation_year').agg(
    electricity_consumption=('cons_12m', 'sum'),
    gas_consumption=('cons_gas_12m', 'sum')
).reset_index()

# Plot yearly trends
plt.figure(figsize=(10, 6))
plt.plot(yearly_consumption['activation_year'], yearly_consumption['electricity_consumption'], label='Electricity Consumption', marker='o', color = 'steelblue')
plt.plot(yearly_consumption['activation_year'], yearly_consumption['gas_consumption'], label='Gas Consumption', marker='o', linestyle='--', color = 'indianred')
plt.title("Yearly Energy Consumption Trends", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Total Consumption", fontsize=12)
plt.legend()
plt.grid(True)
save_plot()
plt.show()

# %% Cell 45
# Aggregate monthly consumption
monthly_consumption = client_df.groupby('activation_month').agg(
    electricity_consumption=('cons_last_month', 'mean'),
    gas_consumption=('cons_gas_12m', 'mean')  # Use cons_gas_12m for gas, if available
).reset_index()

monthly_consumption

# %% Cell 46
plot_stacked_bars(monthly_consumption[['electricity_consumption', 'gas_consumption']], "Average Monthly Consumption", labels=['electricity', 'gas'])
plt.xticks(ticks=range(0, 12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation = 0);

# %% Cell 47
fig, ax = plt.subplots(figsize=(12, 6))
custom_colors = ["steelblue", "indianred"]

monthly_consumption[['electricity_consumption', 'gas_consumption']].plot(
    kind='bar', 
    ax=ax, 
    color=custom_colors
)
ax.set_title("Seasonality in Energy Consumption (Monthly Averages)", fontsize=14, fontweight='bold')
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Average Monthly Consumption", fontsize=12)
ax.set_xticks(ticks=range(0, 12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation = 0)
ax.legend()
for container in ax.containers:
    ax.bar_label(
        container, 
        fmt='%.1f',  
        fontsize=9, 
        color='black', 
        padding=3, 
        fontweight='bold'
    )
    
save_plot()
plt.show()

# %% Cell 48
# Extract year and month for product modifications and renewals
client_df['modif_year_month'] = client_df['date_modif_prod'].dt.to_period('M')
client_df['renewal_year_month'] = client_df['date_renewal'].dt.to_period('M')

# Count modifications and renewals over time
modif_trend = client_df['modif_year_month'].value_counts().sort_index()
renewal_trend = client_df['renewal_year_month'].value_counts().sort_index()

# Plot trends
plt.figure(figsize=(18, 6))
plt.plot(modif_trend.index.astype(str), modif_trend.values, label='Product Modifications', marker='o', color = 'steelblue')
plt.plot(renewal_trend.index.astype(str), renewal_trend.values, label='Contract Renewals', marker='o', linestyle='--', color = 'indianred')

plt.title("Trends in Product Modifications and Renewals", fontsize=14, fontweight='bold')
plt.xlabel("Year-Month", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 49
plt_presets = dict(fontsize=15, fontweight='bold', color="darkblue")

# Create a figure for boxplots
fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(18, 24))
axs = axs.flatten()

# Improved styling for boxplots
for i, col in enumerate(numeric_cols):
    sns.boxplot(
        data=client_df, 
        x=col, 
        ax=axs[i], 
        color=sns.color_palette("pastel")[1],  # Use a more appealing pastel color
        fliersize=4,  # Adjust outlier size
        linewidth=1.5  # Add slightly thicker lines for better clarity
    )
    # Titles and labels
    axs[i].set_title(f"Boxplot of {col.capitalize()}", **plt_presets)
    axs[i].set_xlabel(col.capitalize(), fontsize=12, fontweight="bold")
    axs[i].grid(axis="y", linestyle="--", alpha=0.7)  # Add subtle gridlines for clarity

# Hide unused subplots
for j in range(len(numeric_cols), len(axs)):
    fig.delaxes(axs[j])

# Add an overall title
plt.tight_layout()
fig.subplots_adjust(top=0.95)  # Adjust space for the title
plt.suptitle(
    "Outlier Detection Using Boxplots",
    fontsize=18,
    fontweight='bold',
    color="darkblue"
)

# Display the plot
plt.show()

# %% Cell 50
# Calculate correlation with churn (binary column)
correlation_with_churn = client_df.corr(numeric_only=True)['churn'].sort_values(ascending=False)
print(correlation_with_churn.head(10))


# Visualize correlations
fig, ax = plt.subplots(figsize=(12, 8))

correlation_with_churn.drop('churn').plot(
    kind='barh', 
    color=sns.color_palette("Blues", len(correlation_with_churn) - 1),  # Gradient of blues
    edgecolor='black', 
    ax=ax
)

ax.set_title('Correlation of Features with Churn', fontsize=18, fontweight='bold', color="darkblue")
ax.set_xlabel('Correlation Coefficient', fontsize=14, fontweight='bold')
ax.set_ylabel('Features', fontsize=14, fontweight='bold')

ax.set_xlim(-0.075, 0.5)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2g", label_type="edge", fontsize=10, padding=3)

plt.xticks(rotation=45, ha='right')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()

# %% Cell 51
# Calculate proximity to end or renewal date
client_df['days_to_end'] = (client_df['date_end'] - pd.to_datetime('today')).dt.days
client_df['days_to_renewal'] = (client_df['date_renewal'] - pd.to_datetime('today')).dt.days

# Churn rate by proximity buckets
client_df['end_bucket'] = pd.cut(client_df['days_to_end'], bins=[-float('inf'), 30, 90, 180, 365, float('inf')],
                                 labels=['<1 Month', '1-3 Months', '3-6 Months', '6-12 Months', '>1 Year'])

churn_rate_end = client_df.groupby('end_bucket')['churn'].mean()

churn_rate_end.fillna(0, inplace=True)
churn_rate_end

# %% Cell 52
price_df.info(memory_usage = 'deep')

# %% Cell 53
price_df['id'] = price_df['id'].astype('str')
price_df['price_date'] = pd.to_datetime(price_df['price_date'], errors = 'coerce')

# %% Cell 54
price_cols = ['price_off_peak_var', 'price_peak_var', 'price_mid_peak_var',
              'price_off_peak_fix', 'price_peak_fix', 'price_mid_peak_fix']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
sns.set_style("white")
plt_presets = dict(fontsize=14, fontweight='bold', color="darkblue")

# Plot each price trend
for i, ax in enumerate(axes.flatten()):
    sns.lineplot(data=price_df, x='price_date', y=price_cols[i], ax=ax, marker='o', linewidth=2)
    ax.set_title(f'Trend of {price_cols[i].replace("_", " ").capitalize()} Over Time', fontsize=16, fontweight='bold', color='black')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))  # Format date for better readability
    ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=30))  # Set interval for date labels
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.suptitle("Price Trends Over Time", fontsize=18, fontweight='bold', color = 'black', y=1.02)
save_plot()

# %% Cell 55
