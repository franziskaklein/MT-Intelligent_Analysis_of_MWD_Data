# -*- coding: utf-8 -*-
"""
Interpolation methods

@author: klein
"""


import pandas as pd
from pathlib import Path 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score


def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)


import matplotlib.colors as mcolors
def lighten_color(color, amount=0.5):
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        c = mcolors.CSS4_COLORS[color]
    return tuple(1 - (1 - channel) * amount for channel in c)

# =============================================================================
# load and prepare data
# =============================================================================

#adapt path
path = Path(r'C:\Users\Data\00_created_Data\05_fourty_row_dataframes')

mwd_list = []

for file in sorted(path.glob('*.csv')): 
    df = pd.read_csv(file)
    mwd_list.append(df)  
    
# Columns to be predicted
cols = ["Feed Pressure [Bar]", "Flushing Flow [l/min]", "Flushing Pressure [Bar]",
        "Penetration Rate [m/min]", "Percussion Pressure [Bar]", "Rotation Pressure [Bar]"]

# Ensure each series is of the same length and has no NaNs
filtered_list = []
for df in mwd_list:
    if len(df) == 40:  # Ensure 40 rows per DataFrame
        if not df[cols].isnull().values.any():  # Check for NaN values
            filtered_list.append(df)
mwd_list = filtered_list


#interpolate data to increase data points to use
interpolated_data = []
for df in mwd_list:
    interpolated_df = df.copy()

    # Include Total Depth [m] in columns to interpolate
    columns_to_interpolate = cols + ['Depth [m]']

    # Create interpolated rows for each column
    for col in columns_to_interpolate:
        x = df.index  # Original indices
        y = df[col]  # Original values
        
        # Ensure no NaNs in the original data
        if y.isnull().any():
            print(f"NaNs found in column {col}, skipping interpolation for this column.")
            continue
        
        f = interp1d(x, y, kind='linear', fill_value="extrapolate")  # Linear interpolation
        
        # Generate new indices for interpolation
        new_index = sorted(list(range(len(x))) + [i + 0.5 for i in range(len(x) - 1)])
        
        # Apply interpolation
        interpolated_values = f(new_index)
        
        # Update the interpolated DataFrame
        interpolated_df = interpolated_df.reindex(new_index)  # Create rows for new indices
        interpolated_df[col] = interpolated_values  # Assign interpolated values

    # Reset index and append to the result list
    interpolated_df = interpolated_df.reset_index(drop=True)
    interpolated_data.append(interpolated_df)
    

# =============================================================================
# Prepare Test Data
# =============================================================================

# Train-test split
train_dfs, test_dfs = train_test_split(interpolated_data, test_size=0.2, random_state=42)

# Prepare scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# # Collect all data to fit the scalers
# all_train_data = []
# all_train_targets = []

# # Prepare training data for scaler fitting
# for df in train_dfs:
#     X = df.iloc[:-10][cols].values  # Inputs are all except last 10 rows
#     y = df.iloc[-20:][cols].values  # Targets are last 10 and 20-30 rows for all columns
#     all_train_data.append(X)
#     all_train_targets.append(y)

# # Fit scalers
# scaler_X.fit(np.vstack(all_train_data))
# scaler_y.fit(np.vstack(all_train_targets))

#save scalers for later
scaler_X_path = r'C:\Users\Models\scaler_multi_X-40.pkl'
scaler_y_path = r'C:\Users\Models\scaler_multi_y-40.pkl'
# joblib.dump(scaler_X, scaler_X_path)
# joblib.dump(scaler_y, scaler_y_path)

# #load scaler
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

test_mwd_holes = []
for df in test_dfs:
    df_copy = df.copy()
    test_mwd_holes.append(df_copy)
        
for df in test_mwd_holes:
    df['Total Depth [m]'] = df['Depth [m]']
    

# =============================================================================
# Prepare long boreholes for comparison
# =============================================================================


# Load raw data
path_boreholes_raw = Path(r'C:\Users\Data\TestHoles')
holes_mwd_raw = []
for file in path_boreholes_raw.glob('*_face779_780_raw.csv'):
    df = pd.read_csv(file)
    sorted_df = df.sort_values(by="Total Depth [m]").reset_index(drop=True)
    holes_mwd_raw.append(sorted_df) 
 
# Delete overlap of raw data and fill up with NaNs
holes_mwd_without_overlap = []
for df in holes_mwd_raw:
    df_copy = df.copy()
    df_copy.loc[df_copy["Depth [m]"] > 1.5, df_copy.columns[3:9]] = np.nan
    holes_mwd_without_overlap.append(df_copy)
    
#plot long boreholes
def plot_holes_with_depth(holes_mwd_raw_LSTM, depth_min, depth_max, 
                          colors=['b', 'g', 'r'], indices_to_plot=[0, 1, 2]):
    # Check that files start from the same point
    first_filenames = [holes_mwd_raw_LSTM[idx]["FileName"].iloc[0] for idx in indices_to_plot]
    if len(set(first_filenames)) > 1:
        print("Error: Files do not start at the same face")
        return

    fig, axes = plt.subplots(nrows=len(holes_mwd_raw_LSTM[0].columns[3:9]), ncols=1, figsize=(15, 18))
    fig.suptitle(f"Comparison of holes (with LSTM prediction) from {depth_min}-{depth_max}", fontsize=16)
    all_handles, all_labels = [], []

    for idx, color in zip(indices_to_plot, colors):
        df = holes_mwd_raw_LSTM[idx]
        hole_id = df["Hole ID"].iloc[0]
        
        # Filter by specified depth range
        combined_df = df[(df['Total Depth [m]'] >= depth_min) & (df['Total Depth [m]'] <= depth_max)]
        legend_label = f'Hole ID: {hole_id}'

        columns = combined_df.columns[3:9]

        for i, column in enumerate(columns):
            ax = axes[i]
            x_original, y_original = [], []  # Original values
            x_predicted, y_predicted = [], []  # Predicted values

            for _, row in combined_df.iterrows():
                if pd.isna(row[f"Original {column}"]):  # NaN in "Original {column}" -> predicted
                    x_predicted.append(row['Total Depth [m]'])
                    y_predicted.append(row[column])
                else:  # Original value is present
                    x_original.append(row['Total Depth [m]'])
                    y_original.append(row[column])

            # Plot original values as points
            orig_handle, = ax.plot(x_original, y_original, marker='o', linestyle='', 
                                   color=color, label=f'{legend_label} (original)' if i == 0 else "", markersize=4)
            
            # Plot predicted values as stars
            pred_handle, = ax.plot(x_predicted, y_predicted, marker='*', linestyle='', 
                                   color=color, label=f'{legend_label} (predicted)' if i == 0 else "", markersize=10)

            # Collect handles and labels for the legend (only if not already added)
            if (f'{legend_label} (original)' not in all_labels) and (f'{legend_label} (predicted)' not in all_labels):
                all_handles.extend([orig_handle, pred_handle])
                all_labels.extend([f'{legend_label} (original)', f'{legend_label} (predicted)'])

            ax.set_xlabel('Total Depth [m]')
            ax.set_ylabel(column)
            ax.set_title(f'{column}')

    # Add the legend outside the plot area
    fig.legend(all_handles, all_labels, bbox_to_anchor=(0.5, 1.05), loc='upper center', ncol=len(all_labels) // 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# =============================================================================
# Linear Interpolation as comparison
# =============================================================================

#make linear interpolation in test data

#copy test data    
results_lin = [df.copy() for df in test_dfs]

for df in results_lin:
    for col in cols:
        original_col = f"Original {col}"
        if original_col not in df.columns: #copy original values
            df[original_col] = df[col]
            #set to predict values to nan (same values as in LSTM)
        df.loc[20:40, col] = np.nan

#interpolate
for df in results_lin:
    for col in cols:
        df[col] = df[col].interpolate(method='linear')    
       
all_actuals_lin = {col: [] for col in cols}
all_predictions_lin = {col: [] for col in cols}
for df in results_lin:
    for col in cols:
        actual = df[f"Original {col}"][20:40]
        predicted = df[col][20:40]
        all_actuals_lin[col].extend(actual)
        all_predictions_lin[col].extend(predicted)
        
for col in cols:
    all_actuals_lin[col] = np.array(all_actuals_lin[col])
    all_predictions_lin[col] = np.array(all_predictions_lin[col])   

  
#calc metrics
result_data_linear = {}

for col in cols:
    actual = all_actuals_lin[col]
    predicted = all_predictions_lin[col]

    mse_lin = mean_squared_error(actual, predicted)
    rmse_lin = mse_lin ** 0.5
    r2_lin = r2_score(actual, predicted)
    sym_mape_lin = smape(actual, predicted)
    mape_lin = np.mean(np.abs((actual - predicted) / actual)) * 100
    accuracy_lin = 100 - sym_mape_lin
    max_error_lin = np.max(np.abs(actual - predicted))
    
    result_data_linear[col] = {
        "MSE": mse_lin,
        "RMSE": rmse_lin,
        "R2": r2_lin,
        "sMAPE": sym_mape_lin,
        "accuracy": accuracy_lin,
        "MAPE": mape_lin,
        "Max Error": max_error_lin}  

output_path_lin = Path(r"C:\Users\Models\Vergleich linear")
results_path_lin = output_path_lin/ "results-linear-new.csv"

results_df_lin = pd.DataFrame.from_dict(result_data_linear, orient="index").reset_index()
results_df_lin.rename(columns={"index": "Column"}, inplace=True)

results_df_lin.to_csv(results_path_lin, index=False)




#try on test data + plot
test_mwd_holes_linear = [df.copy() for df in test_mwd_holes]

for df in test_mwd_holes_linear:
    for col in cols:
        original_col = f"Original {col}"
        if original_col not in df.columns:
            df[original_col] = df[col] 
        df.loc[20:40, col] = np.nan 

for df in test_mwd_holes_linear:
    for col in cols:
        df[col] = df[col].interpolate(method='linear')
        
        
def plot_holes_by_depth_linear(holes_mwd_raw_linear, depth_min, depth_max, 
                               colors=['b', 'g', 'r'], indices_to_plot=[0, 1, 2]):
    lighter_colors = [lighten_color(c, amount=0.5) for c in colors]

    for idx, (color, lighter_color) in zip(indices_to_plot, zip(colors, lighter_colors)):
        df = holes_mwd_raw_linear[idx]
        hole_id = df["Hole ID"].iloc[0]

        combined_df = df[(df['Total Depth [m]'] >= depth_min) & (df['Total Depth [m]'] <= depth_max)]

        columns = combined_df.columns[3:9]
        fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(10, 15))
        fig.suptitle(f"Testdata Hole ID {hole_id} (Linear Interpolation)", fontsize=16)
        all_handles, all_labels = [], []

        for i, column in enumerate(columns):
            ax = axes[i]
            x_original, y_original = [], [] 
            x_predicted, y_predicted = [], [] 

            for row_idx, row in combined_df.iterrows():
                if 20 <= row_idx < 40:  # Bedingung für Zeilen 20 bis 40
                    x_depth = row['Depth [m]']
                    y_pred = row[column]
                    if pd.notna(y_pred): 
                        x_predicted.append(x_depth)
                        y_predicted.append(y_pred)

                if pd.notna(row[f"Original {column}"]):  
                    x_original.append(row['Depth [m]'])
                    y_original.append(row[f"Original {column}"])

            pred_handle, = ax.plot(x_predicted, y_predicted, marker='*', linestyle='', 
                                   color=lighter_color, alpha=0.7, label='Interpolated', 
                                   markersize=10)

            orig_handle, = ax.plot(x_original, y_original, marker='o', linestyle='', 
                                   color=color, label='Original', markersize=4)

            if 'Original' not in all_labels and 'Interpolated' not in all_labels:
                all_handles.extend([orig_handle, pred_handle])
                all_labels.extend(['Original', 'Interpolated'])

            ax.set_xlabel('Total Depth [m]')
            ax.set_ylabel(column)
            ax.set_title(f'{column}')

        fig.legend(all_handles, all_labels, bbox_to_anchor=(0.5, 1.05), loc='upper center', ncol=len(all_labels))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
plot_holes_by_depth_linear(test_mwd_holes_linear, depth_min=0.6, depth_max=1.2, indices_to_plot=[5, 8, 9])
plot_holes_by_depth_linear(test_mwd_holes_linear, depth_min=0.6, depth_max=1.2, indices_to_plot=[50, 105, 568])       



#try on long boreholes
holes_mwd_linear =[df.copy() for df in holes_mwd_without_overlap]

for df in holes_mwd_linear:
    for col in cols:
        df[col] = df[col].interpolate(method='linear') 
        
# =============================================================================
# Spline Interpolation
# =============================================================================

#copy test data    
results_spline = [df.copy() for df in test_dfs]

for df in results_spline:
    for col in cols:
        original_col = f"Original {col}"
        if original_col not in df.columns: #copy original values
            df[original_col] = df[col]
        df.loc[20:40, col] = np.nan

#interpolate
from scipy.interpolate import UnivariateSpline
for df in results_spline:
    for column in cols:
        not_nan = ~np.isnan(df[column])
        x = df["Depth [m]"][not_nan]
        y = df[column][not_nan]
        if len(x) > 3:
            spline = UnivariateSpline(x, y, s=0) 
            nan_indices = df.index[df[column].isna()] 
            df.loc[nan_indices, column] = spline(df["Depth [m]"].iloc[nan_indices])
            
all_actuals = {col: [] for col in cols}
all_predictions = {col: [] for col in cols}
for df in results_spline:
    for col in cols:
        actual = df[f"Original {col}"][20:40]
        predicted = df[col][20:40]
        all_actuals[col].extend(actual)
        all_predictions[col].extend(predicted)
        
for col in cols:
    all_actuals[col] = np.array(all_actuals[col])
    all_predictions[col] = np.array(all_predictions[col])   

#calc metrics
result_data_spline = {}


for col in cols:
    actual = all_actuals[col]
    predicted = all_predictions[col]
    
    mse_spline = mean_squared_error(actual, predicted)
    rmse_spline = mse_spline ** 0.5
    r2_spline = r2_score(actual, predicted)
    sym_mape_spline = smape(actual, predicted)
    mape_spline = np.mean(np.abs((actual - predicted) / actual)) * 100
    accuracy_spline = 100 - sym_mape_spline
    max_error_spline = np.max(np.abs(actual - predicted))
    
    result_data_spline[col] = {
        "MSE": mse_spline,
        "RMSE": rmse_spline,
        "R2": r2_spline,
        "sMAPE": sym_mape_spline,
        "accuracy": accuracy_spline,
        "MAPE": mape_spline,
        "Max Error": max_error_spline}  

output_path_spline = Path(r"C:\Users\Models\Vergleich Spline")
results_path_spline = output_path_spline/ "results-spline.csv"

results_df_spline = pd.DataFrame.from_dict(result_data_spline, orient="index").reset_index()
results_df_spline.rename(columns={"index": "Column"}, inplace=True)

results_df_spline.to_csv(results_path_spline, index=False)


        
def plot_holes_by_depth_spline(holes_mwd_raw_spline, depth_min, depth_max, 
                               colors=['b', 'g', 'r'], indices_to_plot=[0, 1, 2]):
    lighter_colors = [lighten_color(c, amount=0.5) for c in colors]

    for idx, (color, lighter_color) in zip(indices_to_plot, zip(colors, lighter_colors)):
        df = holes_mwd_raw_spline[idx]
        hole_id = df["Hole ID"].iloc[0]

        combined_df = df[(df['Depth [m]'] >= depth_min) & (df['Depth [m]'] <= depth_max)]

        columns = combined_df.columns[3:9]
        fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(10, 15))
        fig.suptitle(f"Testdata Hole ID {hole_id} (Spline Interpolation)", fontsize=16)
        all_handles, all_labels = [], []

        for i, column in enumerate(columns):
            ax = axes[i]
            x_original, y_original = [], [] 
            x_predicted, y_predicted = [], [] 

            for row_idx, row in combined_df.iterrows():
                if 20 <= row_idx < 40:  # Bedingung für Zeilen 20 bis 40
                    x_depth = row['Depth [m]']
                    y_pred = row[column]
                    if pd.notna(y_pred): 
                        x_predicted.append(x_depth)
                        y_predicted.append(y_pred)

                if pd.notna(row[f"Original {column}"]):  
                    x_original.append(row['Depth [m]'])
                    y_original.append(row[f"Original {column}"])

            pred_handle, = ax.plot(x_predicted, y_predicted, marker='*', linestyle='', 
                                   color=lighter_color, alpha=0.7, label='Interpolated', 
                                   markersize=10)

            orig_handle, = ax.plot(x_original, y_original, marker='o', linestyle='', 
                                   color=color, label='Original', markersize=4)

            if 'Original' not in all_labels and 'Interpolated' not in all_labels:
                all_handles.extend([orig_handle, pred_handle])
                all_labels.extend(['Original', 'Interpolated'])

            ax.set_xlabel('Depth [m]')
            ax.set_ylabel(column)
            ax.set_title(f'{column}')

        fig.legend(all_handles, all_labels, bbox_to_anchor=(0.5, 1.05), loc='upper center', ncol=len(all_labels))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
plot_holes_by_depth_spline(results_spline, depth_min=0.6, depth_max=1.2, indices_to_plot=[5, 8, 9])
plot_holes_by_depth_spline(results_spline, depth_min=0.6, depth_max=1.2, indices_to_plot=[50, 105, 568])       
        
        


# =============================================================================
# Kriging Interpolation
# =============================================================================

#copy test data    
results_krig = [df.copy() for df in test_dfs]

for df in results_krig:
    for col in cols:
        original_col = f"Original {col}"
        if original_col not in df.columns: #copy original values
            df[original_col] = df[col]
        df.loc[20:40, col] = np.nan

#interpolate
from pykrige.ok import OrdinaryKriging

for df in results_krig:
    for column in cols:
        not_nan = ~np.isnan(df[column])
        if not_nan.sum() > 1:  # At least two points needed for Kriging
            x = df["Depth [m]"][not_nan].values
            y = df[column][not_nan].values
            krige = OrdinaryKriging(x, np.zeros(len(x)), y, variogram_model='linear', verbose=False, enable_plotting=False)

            # Missing values only
            missing_indices = df.index[df[column].isna()]
            if len(missing_indices) > 0:
                missing_depths = df.loc[missing_indices, "Depth [m]"].values
                interpolated_values, _ = krige.execute('points', missing_depths, np.zeros(len(missing_depths)))
                df.loc[missing_indices, column] = interpolated_values
            
       
all_actuals_krig = {col: [] for col in cols}
all_predictions_krig = {col: [] for col in cols}
for df in results_krig:
    for col in cols:
        actual = df[f"Original {col}"][20:40]
        predicted = df[col][20:40]
        all_actuals_krig[col].extend(actual)
        all_predictions_krig[col].extend(predicted)
        
for col in cols:
    all_actuals_krig[col] = np.array(all_actuals_krig[col])
    all_predictions_krig[col] = np.array(all_predictions_krig[col])   

#calc metrics
result_data_krig = {}


for col in cols:
    actual = all_actuals_krig[col]
    predicted = all_predictions_krig[col]
    
    mse_krig = mean_squared_error(actual, predicted)
    rmse_krig = mse_krig ** 0.5
    r2_krig = r2_score(actual, predicted)
    sym_mape_krig = smape(actual, predicted)
    mape_krig = np.mean(np.abs((actual - predicted) / actual)) * 100
    accuracy_krig = 100 - sym_mape_krig
    max_error_krig = np.max(np.abs(actual - predicted))
    
    result_data_krig[col] = {
        "MSE": mse_krig,
        "RMSE": rmse_krig,
        "R2": r2_krig,
        "sMAPE": sym_mape_krig,
        "accuracy": accuracy_krig,
        "MAPE": mape_krig,
        "Max Error": max_error_krig}  

output_path_krig = Path(r"C:\Users\Models\Vergleich Kriging")
results_path_krig = output_path_krig/ "results-krig.csv"

results_df_krig = pd.DataFrame.from_dict(result_data_krig, orient="index").reset_index()
results_df_krig.rename(columns={"index": "Column"}, inplace=True)

results_df_krig.to_csv(results_path_krig, index=False)


        
def plot_holes_by_depth_krig(holes_mwd_raw_krig, depth_min, depth_max, 
                               colors=['b', 'g', 'r'], indices_to_plot=[0, 1, 2]):
    lighter_colors = [lighten_color(c, amount=0.5) for c in colors]

    for idx, (color, lighter_color) in zip(indices_to_plot, zip(colors, lighter_colors)):
        df = holes_mwd_raw_krig[idx]
        hole_id = df["Hole ID"].iloc[0]

        combined_df = df[(df['Depth [m]'] >= depth_min) & (df['Depth [m]'] <= depth_max)]

        columns = combined_df.columns[3:9]
        fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(10, 15))
        fig.suptitle(f"Testdata Hole ID {hole_id} (Kriging Interpolation)", fontsize=16)
        all_handles, all_labels = [], []

        for i, column in enumerate(columns):
            ax = axes[i]
            x_original, y_original = [], [] 
            x_predicted, y_predicted = [], [] 

            for row_idx, row in combined_df.iterrows():
                if 20 <= row_idx < 40:  # Bedingung für Zeilen 20 bis 40
                    x_depth = row['Depth [m]']
                    y_pred = row[column]
                    if pd.notna(y_pred): 
                        x_predicted.append(x_depth)
                        y_predicted.append(y_pred)

                if pd.notna(row[f"Original {column}"]):  
                    x_original.append(row['Depth [m]'])
                    y_original.append(row[f"Original {column}"])

            pred_handle, = ax.plot(x_predicted, y_predicted, marker='*', linestyle='', 
                                   color=lighter_color, alpha=0.7, label='Interpolated', 
                                   markersize=10)

            orig_handle, = ax.plot(x_original, y_original, marker='o', linestyle='', 
                                   color=color, label='Original', markersize=4)

            if 'Original' not in all_labels and 'Interpolated' not in all_labels:
                all_handles.extend([orig_handle, pred_handle])
                all_labels.extend(['Original', 'Interpolated'])

            ax.set_xlabel('Depth [m]')
            ax.set_ylabel(column)
            ax.set_title(f'{column}')

        fig.legend(all_handles, all_labels, bbox_to_anchor=(0.5, 1.05), loc='upper center', ncol=len(all_labels))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
plot_holes_by_depth_krig(results_krig, depth_min=0.6, depth_max=1.2, indices_to_plot=[5, 8, 9])
plot_holes_by_depth_krig(results_krig, depth_min=0.6, depth_max=1.2, indices_to_plot=[50, 105, 568])       
        

