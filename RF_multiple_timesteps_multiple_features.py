# -*- coding: utf-8 -*-
"""

@author: klein
predict multiple features - autoregressive Random Forest

"""

import pandas as pd
from pathlib import Path 
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib


def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)


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


# =============================================================================
# Prepare RF Data
# =============================================================================

# Train-test split
train_dfs, test_dfs = train_test_split(mwd_list, test_size=0.2, random_state=42)

# Prepare scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

#save scalers for later
scaler_X_path = r'C:\Users\Models\scaler_multi_X-40.pkl'
scaler_y_path = r'C:\Users\Models\scaler_multi_y-40.pkl'
# joblib.dump(scaler_X, scaler_X_path)
# joblib.dump(scaler_y, scaler_y_path)

# #load scaler
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)


# Initialize lists for storing data
X_train_list, y_train_list, depth_train_list = [], [], []
X_test_list, y_test_list, depth_test_list = [], [], []
X_train_rf, y_train_rf = [], []
X_test_rf, y_test_rf = [], []

# Convert training data
for df in train_dfs:
    data = df[cols].values
    
    #initial input
    initial_input = data[0:20]
    target_1 = data[20:30]
    
    #input with prediction included
    autoreg_input = data[10:20]
    target_2 = data[30:40]
    
    # Scale
    initial_input_scaled = scaler_X.transform(initial_input)
    target_1_scaled = scaler_y.transform(target_1)
    autoreg_input_scaled = scaler_X.transform(autoreg_input)
    target_2_scaled = scaler_y.transform(target_2)
    
    #use this for RF
    rf_input = np.hstack((initial_input_scaled.flatten(), autoreg_input_scaled.flatten()))  # Combine both inputs
    rf_target = np.hstack((target_1_scaled.flatten(), target_2_scaled.flatten()))  # Combine both targets
    X_train_rf.append(rf_input)
    y_train_rf.append(rf_target)
    
    # Convert to tensors
    X_train_list.append({
        "initial_input": torch.tensor(initial_input_scaled, dtype=torch.float32).unsqueeze(0),
        "autoreg_input": torch.tensor(autoreg_input_scaled, dtype=torch.float32).unsqueeze(0)})
    y_train_list.append({
        "target_1": torch.tensor(target_1_scaled, dtype=torch.float32),
        "target_2": torch.tensor(target_2_scaled, dtype=torch.float32)})
    
# Convert lists to numpy arrays for RandomForest
X_train_rf = np.vstack(X_train_rf)  # Stack all X_train for RF
y_train_rf = np.vstack(y_train_rf)  # Stack all y_train for RF

# Prepare test data
for df in test_dfs:
    data = df[cols].values
    
    #initial input
    initial_input = data[0:20]
    target_1 = data[20:30]
    
    #input with prediction included
    autoreg_input = data[10:20]
    target_2 = data[30:40]
    
    # Scale
    initial_input_scaled = scaler_X.transform(initial_input)
    target_1_scaled = scaler_y.transform(target_1)
    autoreg_input_scaled = scaler_X.transform(autoreg_input)
    target_2_scaled = scaler_y.transform(target_2)
    
    #use this for RF
    rf_input = np.hstack((initial_input_scaled.flatten(), autoreg_input_scaled.flatten()))  # Combine both inputs
    rf_target = np.hstack((target_1_scaled.flatten(), target_2_scaled.flatten()))  # Combine both targets
    X_test_rf.append(rf_input)
    y_test_rf.append(rf_target)

    # Convert to tensors for LSTM
    X_test_list.append({
        "initial_input": torch.tensor(initial_input_scaled, dtype=torch.float32).unsqueeze(0),
        "autoreg_input": torch.tensor(autoreg_input_scaled, dtype=torch.float32).unsqueeze(0)})
    y_test_list.append({
        "target_1": torch.tensor(target_1_scaled, dtype=torch.float32),
        "target_2": torch.tensor(target_2_scaled, dtype=torch.float32)})

# convert to numpy array for RF
X_test_rf = np.vstack(X_test_rf)
y_test_rf = np.vstack(y_test_rf)



#prepare backwards data for second LSTM
X_train_list_backwards, y_train_list_backwards = [], []
X_test_list_backwards, y_test_list_backwards = [], []
X_train_rf_backwards, y_train_rf_backwards = [], []
X_test_rf_backwards, y_test_rf_backwards = [], []

#train
for df in train_dfs:
    data = df[cols].values[::-1]  # Reverse the data

    initial_input = data[0:20]  
    target_1 = data[20:30]      
    autoreg_input = data[10:20] 
    target_2 = data[30:40]      

    #scale data
    initial_input_scaled = scaler_X.transform(initial_input)
    target_1_scaled = scaler_y.transform(target_1)
    autoreg_input_scaled = scaler_X.transform(autoreg_input)
    target_2_scaled = scaler_y.transform(target_2)
    
    #RF data
    rf_input = np.hstack((initial_input_scaled.flatten(), autoreg_input_scaled.flatten())) 
    rf_target = np.hstack((target_1_scaled.flatten(), target_2_scaled.flatten()))  
    X_train_rf_backwards.append(rf_input)
    y_train_rf_backwards.append(rf_target)
    
    #convert to tensors
    X_train_list_backwards.append({
        "initial_input": torch.tensor(initial_input_scaled, dtype=torch.float32).unsqueeze(0),
        "autoreg_input": torch.tensor(autoreg_input_scaled, dtype=torch.float32).unsqueeze(0)})
    y_train_list_backwards.append({
        "target_1": torch.tensor(target_1_scaled, dtype=torch.float32),
        "target_2": torch.tensor(target_2_scaled, dtype=torch.float32)})

# Convert lists to numpy arrays for RandomForest
X_train_rf_backwards = np.vstack(X_train_rf_backwards)  # Stack all X_train for RF
y_train_rf_backwards = np.vstack(y_train_rf_backwards)  # Stack all y_train for RF

#test
for df in test_dfs:
    data = df[cols].values[::-1]  # Reverse the data

    initial_input = data[0:20]  
    target_1 = data[20:30]      
    autoreg_input = data[10:20] 
    target_2 = data[30:40]      

    #scale data
    initial_input_scaled = scaler_X.transform(initial_input)
    target_1_scaled = scaler_y.transform(target_1)
    autoreg_input_scaled = scaler_X.transform(autoreg_input)
    target_2_scaled = scaler_y.transform(target_2)
    
    #RF data
    rf_input = np.hstack((initial_input_scaled.flatten(), autoreg_input_scaled.flatten())) 
    rf_target = np.hstack((target_1_scaled.flatten(), target_2_scaled.flatten()))  
    X_test_rf_backwards.append(rf_input)
    y_test_rf_backwards.append(rf_target)
    
    #convert to tensors
    X_test_list_backwards.append({
        "initial_input": torch.tensor(initial_input_scaled, dtype=torch.float32).unsqueeze(0),
        "autoreg_input": torch.tensor(autoreg_input_scaled, dtype=torch.float32).unsqueeze(0)})
    y_test_list_backwards.append({
        "target_1": torch.tensor(target_1_scaled, dtype=torch.float32),
        "target_2": torch.tensor(target_2_scaled, dtype=torch.float32)})

# Convert lists to numpy arrays for RandomForest
X_test_rf_backwards = np.vstack(X_test_rf_backwards)  
y_test_rf_backwards = np.vstack(y_test_rf_backwards) 


# =============================================================================
# get long boreholes
# =============================================================================


# Load raw data
path_boreholes_raw = Path(r'C:\Users\klein\Desktop\Geotechnical and Hydraulic Engineering\MP_MA\Masterarbeit\Coding\Abbildungen\TestHoles')
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



# =============================================================================
# Basemodel - Random Forest (autoregressive)
# =============================================================================

#use a multi-output regressor with RF to generate 6 outputs per timestep
model_rf_step1 = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=800,
        random_state=42,
        max_depth=14,
        min_samples_leaf=3,
        max_features=6))


model_rf_step2 = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=800,
        random_state=42,
        max_depth=14,
        min_samples_leaf=3,
        max_features=6))

#train first step: initial inputwith only real data
X_train_step1 = [x["initial_input"].numpy().flatten() for x in X_train_list]
y_train_step1 = [y["target_1"].numpy().flatten() for y in y_train_list]
model_rf_step1.fit(np.array(X_train_step1), np.array(y_train_step1))

#train step 2
X_train_step2 = []
y_train_step2 = []

#gather data here
for x_train, y_train in zip(X_train_list, y_train_list):
    initial_input = x_train["initial_input"].numpy()  # Shape: (1, 20, len(cols))
    autoreg_input = x_train["autoreg_input"].numpy()  # Shape: (1, 10, len(cols))
    target_2 = y_train["target_2"].numpy()  # Shape: (10, len(cols))

    #predict step one
    predicted_step1 = model_rf_step1.predict(initial_input.flatten().reshape(1, -1))  # Shape: (1, 10 * len(cols))
    predicted_step1 = predicted_step1.reshape(10, len(cols))  # Shape: (10, len(cols))

    predicted_step1_expanded = np.expand_dims(predicted_step1, axis=0)  # Shape: (1, 10, len(cols))
    combined_input = np.concatenate((autoreg_input[:, :-10, :], predicted_step1_expanded), axis=1).flatten()
    X_train_step2.append(combined_input)
    y_train_step2.append(target_2.flatten())

#train model
model_rf_step2.fit(np.array(X_train_step2), np.array(y_train_step2))

#save
base_model_path = Path(r'C:/Users/klein/Desktop/')
model_path_step1 = base_model_path / 'RF_multi_outputautoreg_step1.pkl'
model_path_step2 = base_model_path / 'RF_multi_outputautoreg_step2.pkl'
# joblib.dump(model_rf_step1, model_path_step1)
# joblib.dump(model_rf_step2, model_path_step2)

#load model
model_rf_step1 = joblib.load(model_path_step1)
model_rf_step2 = joblib.load(model_path_step2)


# Make predictions
rf_predictions_step1 = []
rf_predictions_step2 = []

#first prediction
initial_inputs = [x_test["initial_input"].numpy().flatten() for x_test in X_test_list]  # Flache Eingaben
rf_predictions_step1 = model_rf_step1.predict(np.array(initial_inputs))  # Batch-Vorhersage
rf_predictions_step1 = rf_predictions_step1.reshape(-1, 10, len(cols))  # Reshape für Step 1

#second prediction
rf_predictions_step2 = []
for i, x_test in enumerate(X_test_list):
    autoreg_input = x_test["autoreg_input"].numpy()  # Shape: (1, 10, len(cols))

    # Combine autoreg_input and predicted_step1 for Step 2
    predicted_step1 = rf_predictions_step1[i]  # Shape: (10, len(cols))
    predicted_step1_expanded = np.expand_dims(predicted_step1, axis=0)  # Add batch dimension
    combined_input = np.concatenate((autoreg_input[:, :-10, :], predicted_step1_expanded), axis=1).flatten()

    rf_predictions_step2.append(combined_input)

rf_predictions_step2 = model_rf_step2.predict(np.array(rf_predictions_step2))  # Batch-Vorhersage
rf_predictions_step2 = rf_predictions_step2.reshape(-1, 10, len(cols))  # Reshape für Step 2

#get actual values for comparison
actual_reshaped = y_test_rf.reshape(-1, 20, len(cols))  # Shape: (n_samples, 20, len(cols))
actual_step_1 = actual_reshaped[:, :10, :]  # Step 1
actual_step_2 = actual_reshaped[:, 10:, :]  # Step 2

# Flatten for metrics
predicted_step1_flat = rf_predictions_step1.reshape(-1, len(cols))
predicted_step2_flat = rf_predictions_step2.reshape(-1, len(cols))
actual_step1_flat = actual_step_1.reshape(-1, len(cols))
actual_step2_flat = actual_step_2.reshape(-1, len(cols))



#calculate metrics
result_data_list_rf = {"Step 1": {}, "Step 2": {}, "Combined": {}}

# Step 1 und Step 2 separat berechnen und Scatterplots erstellen
for step, (actual_flat, predicted_flat) in zip(
    ["Step 1", "Step 2"],[(actual_step1_flat, predicted_step1_flat), 
                          (actual_step2_flat, predicted_step2_flat)]):
    for i, col in enumerate(cols):
        actual_col = actual_flat[:, i]
        predicted_col = predicted_flat[:, i]

        #calculate metrics
        mse_rf = mean_squared_error(actual_col, predicted_col)
        rmse_rf = mse_rf ** 0.5
        sym_mape_rf = smape(actual_col, predicted_col)
        accuracy_rf = 100 - sym_mape_rf

        result_data_list_rf[step][col] = {"MSE": mse_rf, "RMSE": rmse_rf, "Accuracy": accuracy_rf}

        #scatterplot
        plt.figure(figsize=(8, 8))
        plt.scatter(actual_col, predicted_col, alpha=0.7)
        plt.title(f'Scatter Plot RF ({step}) for {col}')
        plt.xlabel(f'Actual {col} ({step})')
        plt.ylabel(f'Predicted {col} ({step})')
        plt.xlim(actual_col.min(), actual_col.max())
        plt.ylim(predicted_col.min(), predicted_col.max())
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

#get metrics for both steps combined to compare to LSTM
actual_combined = np.vstack([actual_step1_flat, actual_step2_flat])
predicted_combined = np.vstack([predicted_step1_flat, predicted_step2_flat])

for i, col in enumerate(cols):
    actual_col = actual_combined[:, i]
    predicted_col = predicted_combined[:, i]

    mse_rf = mean_squared_error(actual_col, predicted_col)
    rmse_rf = mse_rf ** 0.5
    sym_mape_rf = smape(actual_col, predicted_col)
    accuracy_rf = 100 - sym_mape_rf

    result_data_list_rf["Combined"][col] = {"MSE": mse_rf, "RMSE": rmse_rf, "Accuracy": accuracy_rf}

    # Scatterplot for both steps 
    plt.figure(figsize=(8, 8))
    plt.scatter(actual_col, predicted_col, alpha=0.7)
    plt.title(f'Scatter Plot RF (Combined) for {col}')
    plt.xlabel(f'Actual {col} (Combined)')
    plt.ylabel(f'Predicted {col} (Combined)')
    plt.xlim(actual_col.min(), actual_col.max())
    plt.ylim(predicted_col.min(), predicted_col.max())
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    



# =============================================================================
# predict long boreholes
# =============================================================================

def predict_and_fill_nans_rf(model_rf_step1, model_rf_step2, data, scaler_X, scaler_y, cols):
    output_data = []

    for df in data:
        # Save original values for each column
        for col in cols:
            original_col = f"Original {col}"
            if original_col not in df.columns:
                df[original_col] = df[col]

        # Sort by "Total Depth [m]" and reset the index
        df = df.sort_values(by="Total Depth [m]").reset_index(drop=True)

        # Find NaN segments
        nan_segments = df[cols[0]].isna()

        i = 20  # Start at 20 since input requires 20 rows
        while i < len(df):
            if nan_segments[i]:
                nan_start = i
                #length of the NaN segment
                while i < len(df) and nan_segments[i]:
                    i += 1
                nan_end = i
                nan_length = nan_end - nan_start

                # Reset to Step 1 for the new NaN segment
                step = 1

                #check if 20 rows in front are complete
                while nan_length > 0:
                    input_data = df[cols].iloc[nan_start - 20:nan_start].values
                    input_data = scaler_X.transform(input_data)  # Scale input data

                    #for first 10 rows --> use model 1
                    if step == 1:
                        input_data_flat = input_data.flatten().reshape(1, -1)  # Flatten to (1, 20 * n_features)
                        predicted_output = model_rf_step1.predict(input_data_flat)
                        step = 2  #model 1 just for first 10, then model 2
                    else:  #model_rf_step2
                        autoreg_input = input_data[-10:]  # Last 10 rows
                        autoreg_input_scaled = scaler_X.transform(autoreg_input)  # Scale autoregressive input
                        predicted_step1 = predicted_output.reshape(10, len(cols))
                        combined_input = np.concatenate((autoreg_input_scaled, predicted_step1), axis=0).flatten()
                        combined_input = combined_input[-60:]  # Use only the last 10 rows (10 * len(cols))
                        predicted_output = model_rf_step2.predict(combined_input.reshape(1, -1))

                    #Reshape to (10, len(cols))
                    predicted_output = scaler_y.inverse_transform(
                        predicted_output.reshape(10, len(cols)))

                    #Fill NaN values
                    num_to_fill = min(nan_length, 10)
                    for j, col in enumerate(cols):
                        df.iloc[
                            nan_start:nan_start + num_to_fill, df.columns.get_loc(col)
                        ] = predicted_output[:num_to_fill, j]

                    # Update variables
                    nan_start += num_to_fill
                    nan_length -= num_to_fill
            else:
                i += 1

        output_data.append(df)

    return output_data

holes_mwd_raw_RF = predict_and_fill_nans_rf(model_rf_step1, model_rf_step2,
    holes_mwd_without_overlap, scaler_X, scaler_y, cols)

# Plotting

def plot_holes_with_depth_RF(data, depth_min, depth_max, 
                          colors=['b', 'g', 'r'], indices_to_plot=[0, 1, 2]):
    # Check that files start from the same point
    first_filenames = [data[idx]["FileName"].iloc[0] for idx in indices_to_plot]
    if len(set(first_filenames)) > 1:
        print("Error: Files do not start at the same face")
        return

    fig, axes = plt.subplots(nrows=len(data[0].columns[3:9]), ncols=1, figsize=(15, 18))
    fig.suptitle(f"Comparison of holes (with Random Forest prediction) from {depth_min}-{depth_max}", fontsize=16)
    all_handles, all_labels = [], []

    for idx, color in zip(indices_to_plot, colors):
        df = data[idx]
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
plot_holes_with_depth_RF(holes_mwd_raw_RF, depth_min=2, depth_max=4.5)


idx = [0,1,2]

for i in idx:
    df = holes_mwd_raw_RF[i]
    df_right = df[df["FileName"].isin(["780_drilling.csv"])]

    cols = df.iloc[:, 3:9].columns.tolist()  
    df_selected = df_right[cols].apply(pd.to_numeric, errors='coerce')  # 
    correlation_matrix = df_selected.corr()
    hole_id = df_right['Hole ID'].iloc[0]
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix Random Forest data (autoreg.) for Hole {hole_id} in face 780')
    plt.tight_layout()
    plt.show()       


# =============================================================================
# backward prediction RF
# =============================================================================

#Rf does not work with time-series --> use same model as before and just flip input

rf_pred_backward_step1= []
rf_pred_backward_step2 = []

#flip testdata
X_test_list_backward = list(reversed(X_test_list))

#initial prediction
initial_inputs_backward = [x_test["initial_input"].numpy().flatten() for x_test in X_test_list_backward]  # Flatten inputs
rf_pred_backward_step1= model_rf_step1.predict(np.array(initial_inputs_backward))  # Batch prediction
rf_pred_backward_step1= rf_pred_backward_step1.reshape(-1, 10, len(cols))  
#results back to original order
rf_pred_backward_step1= np.flip(rf_pred_backward_step1, axis=0)  

#step 2
rf_pred_backward_step2 = []
for i, x_test in enumerate(X_test_list_backward):
    autoreg_input = x_test["autoreg_input"].numpy()  # Shape: (1, 10, len(cols))

    # Combine autoreg_input and predicted_step1 for Step 2
    predicted_step1 = rf_pred_backward_step1[i]  # Shape: (10, len(cols))
    predicted_step1_expanded = np.expand_dims(predicted_step1, axis=0)  # Add batch dimension
    combined_input = np.concatenate((autoreg_input[:, :-10, :], predicted_step1_expanded), axis=1).flatten()

    rf_pred_backward_step2.append(combined_input)

rf_pred_backward_step2 = model_rf_step2.predict(np.array(rf_pred_backward_step2))  # Batch prediction
rf_pred_backward_step2 = rf_pred_backward_step2.reshape(-1, 10, len(cols))  
#get back original order
rf_pred_backward_step2 = np.flip(rf_pred_backward_step2, axis=0)


# =============================================================================
# forward + backward prediction RF
# =============================================================================

rf_pred_combined_step1 = (rf_predictions_step1 + rf_pred_backward_step1)/2
rf_pred_combined_step2 = (rf_predictions_step2 + rf_pred_backward_step2)/2

#calculate metrics
result_data_list_rf_combined = {"Combined": {}}

for i, col in enumerate(cols):
    actual_col = np.vstack([actual_step1_flat, actual_step2_flat])[:, i]
    predicted_col = np.vstack([rf_pred_combined_step1, rf_pred_combined_step2])[:, :, i].flatten()

    # Calculate Metrics
    mse_rf = mean_squared_error(actual_col, predicted_col)
    rmse_rf = mse_rf ** 0.5
    sym_mape_rf = smape(actual_col, predicted_col)
    accuracy_rf = 100 - sym_mape_rf

    result_data_list_rf_combined["Combined"][col] = {"MSE": mse_rf, "RMSE": rmse_rf, "Accuracy": accuracy_rf}

    # Scatterplot
    plt.figure(figsize=(8, 8))
    plt.scatter(actual_col, predicted_col, alpha=0.7)
    plt.title(f'Scatter Plot RF (Combined) for {col}')
    plt.xlabel(f'Actual {col} (Combined)')
    plt.ylabel(f'Predicted {col} (Combined)')
    plt.xlim(actual_col.min(), actual_col.max())
    plt.ylim(predicted_col.min(), predicted_col.max())
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

#irgendwas passt hier nicht!! werden nicht richtig kombiniert??

def predict_and_fill_nans_rf_combined(model_rf_step1, model_rf_step2, data, scaler_X, scaler_y, cols):
    output_data = []
    all_predictions = [] #list to check actual algorithm
    
    for df in data:
        # Save original values for each column
        for col in cols:
            original_col = f"Original {col}"
            if original_col not in df.columns:
                df[original_col] = df[col]

        # Sort by "Total Depth [m]" and reset the index
        df = df.sort_values(by="Total Depth [m]").reset_index(drop=True)
        df_backward = df.sort_values(by="Total Depth [m]", ascending=False).reset_index(drop=True)

        # Precompute NaN segments
        nan_segments_forward = df[cols[0]].isna()
        nan_segments_backward = df_backward[cols[0]].isna()

        #collect predictions here
        predictions_forward = df[cols].copy()
        predictions_backward = df_backward[cols].copy()

        # Forward prediction
        if nan_segments_forward.any():
            forward_nan_indices = np.where(nan_segments_forward)[0]
            forward_start_indices = forward_nan_indices[forward_nan_indices >= 20]  # Only valid segments

            for nan_start in forward_start_indices:
                step = 1
                nan_length = np.sum(nan_segments_forward[nan_start:])
                while nan_length > 0:
                    # Prepare input data
                    input_start = max(nan_start - 20, 0)
                    input_data = df[cols].iloc[input_start:nan_start].values

                    # Skip iteration if NaN found in input
                    if np.isnan(input_data).any():
                        break

                    input_data = scaler_X.transform(input_data)  # Scale input data

                    # Step 1 or Step 2 predictions
                    if step == 1:
                        input_data_flat = input_data.flatten().reshape(1, -1)
                        predicted_output = model_rf_step1.predict(input_data_flat)
                        step = 2
                    else:
                        autoreg_input = input_data[-10:]
                        combined_input = np.concatenate((autoreg_input, predicted_output.reshape(10, len(cols))), axis=0).flatten()
                        combined_input = combined_input[-60:]
                        predicted_output = model_rf_step2.predict(combined_input.reshape(1, -1))

                    # Scale back predictions and update forward predictions
                    predicted_output = scaler_y.inverse_transform(predicted_output.reshape(10, len(cols)))
                    num_to_fill = min(nan_length, 10)
                    for j, col in enumerate(cols):
                        predictions_forward.iloc[
                            nan_start:nan_start + num_to_fill, predictions_forward.columns.get_loc(col)] = \
                            predicted_output[:num_to_fill, j]

                    nan_start += num_to_fill
                    nan_length -= num_to_fill

        # Backward prediction
        if nan_segments_backward.any():
            backward_nan_indices = np.where(nan_segments_backward)[0]
            backward_start_indices = backward_nan_indices[backward_nan_indices >= 20]

            for nan_start in backward_start_indices:
                step = 1
                nan_length = np.sum(nan_segments_backward[nan_start:])
                while nan_length > 0:
                    # Prepare input data
                    input_start = max(nan_start - 20, 0)
                    input_data = df_backward[cols].iloc[input_start:nan_start].values

                    # Skip iteration if NaN found in input
                    if np.isnan(input_data).any():
                        break

                    input_data = scaler_X.transform(input_data)

                    if step == 1:
                        input_data_flat = input_data.flatten().reshape(1, -1)
                        predicted_output = model_rf_step1.predict(input_data_flat)
                        step = 2
                    else:
                        autoreg_input = input_data[-10:]
                        combined_input = np.concatenate((autoreg_input, predicted_output.reshape(10, len(cols))), axis=0).flatten()
                        combined_input = combined_input[-60:]
                        predicted_output = model_rf_step2.predict(combined_input.reshape(1, -1))

                    # Scale back predictions and update backward predictions
                    predicted_output = scaler_y.inverse_transform(predicted_output.reshape(10, len(cols)))
                    num_to_fill = min(nan_length, 10)
                    for j, col in enumerate(cols):
                        predictions_backward.iloc[
                            nan_start:nan_start + num_to_fill, predictions_backward.columns.get_loc(col)] = \
                            predicted_output[:num_to_fill, j]

                    nan_start += num_to_fill
                    nan_length -= num_to_fill
                    
        predictions_backward.iloc[::-1]
        # Combine forward and backward predictions
        combined_predictions = predictions_forward.copy()
        for col in cols:
            forward_values = predictions_forward[col]
            backward_values = predictions_backward[col]

            # Combine only if both are available; otherwise use the one available
            combined_predictions[col] = np.where(
                forward_values.notna() & backward_values.notna(),
                (forward_values + backward_values) / 2,
                forward_values.fillna(backward_values))
        
        
        pred_df = df[['Total Depth [m]']].copy()
        for col in cols:
            pred_df[f"Original {col}"] = df[f"Original {col}"]
            pred_df[f"Forward Predicted {col}"] = predictions_forward[col]
            pred_df[f"Backward Predicted {col}"] = predictions_backward[col].iloc[::-1]
            pred_df[f"Combined Predicted {col}"] = combined_predictions[col]
        
        all_predictions.append(pred_df)
        
        # Fill combined predictions into the original DataFrame
        for col in cols:
            nan_rows = df[col].isna()
            df.loc[nan_rows, col] = combined_predictions.loc[nan_rows, col]

        output_data.append(df)
        
    return output_data, all_predictions


holes_mwd_raw_RF_combined, pred_check = predict_and_fill_nans_rf_combined(model_rf_step1, 
model_rf_step2, holes_mwd_without_overlap, scaler_X, scaler_y, cols)

plot_holes_with_depth_RF(holes_mwd_raw_RF_combined, depth_min=2, depth_max=4.5)
