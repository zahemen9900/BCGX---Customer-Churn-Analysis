import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, price_df):
        """
        Initializes the FeatureEngineering class with the input price data.
        
        Parameters:
        price_df (pd.DataFrame): The DataFrame containing the price data.
        """
        self.price_df = price_df.copy(deep=True)
        self.features = pd.DataFrame()

    def calculate_monthly_price_differences(self):
        """
        Calculates the differences in energy prices between December and January.
        """
        monthly_price_by_id = self.price_df.groupby(['id', 'price_date']).agg({
            'price_off_peak_var': 'mean',
            'price_off_peak_fix': 'mean'
        }).reset_index()

        jan_prices = monthly_price_by_id.groupby('id').first()
        dec_prices = monthly_price_by_id.groupby('id').last()

        diff = dec_prices.drop('price_date', axis=1).join(
            jan_prices.drop('price_date', axis=1),
            on='id',
            how='inner',
            lsuffix='_d',
            rsuffix='_j'
        )
        diff['offpeak_diff_dec_january_energy'] = diff['price_off_peak_var_d'] - diff['price_off_peak_var_j']
        diff['offpeak_diff_dec_january_power'] = diff['price_off_peak_fix_d'] - diff['price_off_peak_fix_j']

        self.features = pd.concat([self.features, diff[['offpeak_diff_dec_january_energy', 'offpeak_diff_dec_january_power']]], axis=1)

    def calculate_price_ratios(self):
        """
        Calculates price ratios for the differences between December and January.
        """
        self.features['offpeak_ratio_dec_january_energy'] = (self.features['offpeak_diff_dec_january_energy'] + 1) / self.features['offpeak_diff_dec_january_energy']
        self.features['offpeak_ratio_dec_january_power'] = (self.features['offpeak_diff_dec_january_power'] + 1) / self.features['offpeak_diff_dec_january_power']

    def calculate_annual_averages(self):
        """
        Calculates the annual average prices for energy and power.
        """
        annual_avg = self.price_df.groupby('id').agg({
            'price_off_peak_var': 'mean',
            'price_off_peak_fix': 'mean'
        })
        annual_avg.rename(columns={
            'price_off_peak_var': 'avg_price_off_peak_var',
            'price_off_peak_fix': 'avg_price_off_peak_fix'
        }, inplace=True)
        
        self.features = self.features.join(annual_avg, on='id', how='left')

    def calculate_seasonal_averages(self):
        """
        Calculates the average prices for summer and winter.
        """
        summer_prices = self.price_df[(self.price_df['price_date'].dt.month == 7) | (self.price_df['price_date'].dt.month == 8)]
        winter_prices = self.price_df[(self.price_df['price_date'].dt.month == 12) | (self.price_df['price_date'].dt.month == 1)]

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
        }, axis=1, inplace=True)

        winter_avg.rename({
            'price_off_peak_var': 'price_off_peak_var_winter',
            'price_off_peak_fix': 'price_off_peak_fix_winter'
        }, axis=1, inplace=True)

        seasonal_diff = summer_avg.join(winter_avg, on='id', how='right', lsuffix='_summer', rsuffix='_winter')
        seasonal_diff['summer_winter_diff_energy'] = seasonal_diff['price_off_peak_var_summer'] - seasonal_diff['price_off_peak_var_winter']
        seasonal_diff['summer_winter_diff_power'] = seasonal_diff['price_off_peak_fix_summer'] - seasonal_diff['price_off_peak_fix_winter']

        self.features = self.features.join(seasonal_diff[['summer_winter_diff_energy', 'summer_winter_diff_power']], on='id', how='left')

    def calculate_rolling_averages(self):
        """
        Calculates rolling 3-month averages for energy and power prices.
        """
        self.price_df['price_off_peak_var_rolling'] = self.price_df.groupby('id')['price_off_peak_var'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
        self.price_df['price_off_peak_fix_rolling'] = self.price_df.groupby('id')['price_off_peak_fix'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)

        rolling_avg = self.price_df.groupby('id').agg({
            'price_off_peak_var_rolling': 'last',
            'price_off_peak_fix_rolling': 'last'
        })
        rolling_avg.rename(columns={
            'price_off_peak_var_rolling': 'rolling_3m_avg_price_off_peak_var',
            'price_off_peak_fix_rolling': 'rolling_3m_avg_price_off_peak_fix'
        }, inplace=True)

        self.features = self.features.join(rolling_avg, on='id', how='left')

    def calculate_price_variability(self):
        """
        Calculates the standard deviation, max, and min for energy and power prices.
        """
        price_variability = self.price_df.groupby('id').agg({
            'price_off_peak_var': ['std', 'max', 'min'],
            'price_off_peak_fix': ['std', 'max', 'min']
        })
        price_variability.columns = [
            'std_price_off_peak_var', 'max_price_off_peak_var', 'min_price_off_peak_var',
            'std_price_off_peak_fix', 'max_price_off_peak_fix', 'min_price_off_peak_fix'
        ]
        
        self.features = self.features.join(price_variability, on='id', how='left')

    
    def run_feature_engineering(self):
        """
        Runs the entire feature engineering process by calling all the methods.
        """
        self.calculate_monthly_price_differences()
        self.calculate_price_ratios()
        self.calculate_annual_averages()
        self.calculate_seasonal_averages()
        self.calculate_rolling_averages()
        self.calculate_price_variability()
        # Merge final features back with the original price DataFrame
        self.price_df = self.price_df.join(self.features, on='id', how='left')
        return self.price_df
    
# Usage example:
# price_df = pd.read_csv('path/to/price_data.csv')  # Ensure price_df is already loaded
# pipeline = FeatureEngineeringPipeline(price_df)
# enriched_price_df = pipeline.run_feature_engineering()
