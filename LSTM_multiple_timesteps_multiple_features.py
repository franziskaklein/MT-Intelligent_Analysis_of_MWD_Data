# -*- coding: utf-8 -*-
"""


@author: klein

try backward + forward LSTM with larger dataset (interpolated points) and Random Forest

"""


import pandas as pd
from pathlib import Path 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import os
import joblib
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score


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
# Prepare LSTM Data
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


# Initialize lists for Forward model
X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

# Convert training data
for df in train_dfs:
    data = df[cols].values
    
    # Split data into inputs and targets
    initial_input = data[0:20]
    target_1 = data[20:30]
    autoreg_input = data[10:20]
    target_2 = data[30:40]
    
    # Scale inputs and targets
    initial_input_scaled = scaler_X.transform(initial_input)
    target_1_scaled = scaler_y.transform(target_1)
    autoreg_input_scaled = scaler_X.transform(autoreg_input)
    target_2_scaled = scaler_y.transform(target_2)
    
    # Convert to tensors for LSTM
    X_train_list.append({
        "initial_input": torch.tensor(initial_input_scaled, dtype=torch.float32).unsqueeze(0),
        "autoreg_input": torch.tensor(autoreg_input_scaled, dtype=torch.float32).unsqueeze(0)})
    y_train_list.append({
        "target_1": torch.tensor(target_1_scaled, dtype=torch.float32),
        "target_2": torch.tensor(target_2_scaled, dtype=torch.float32)})

# Prepare test data
for df in test_dfs:
    data = df[cols].values
    
    # Split data into inputs and targets
    initial_input = data[0:20]
    target_1 = data[20:30]
    autoreg_input = data[10:20]
    target_2 = data[30:40]
    
    # Scale inputs and targets
    initial_input_scaled = scaler_X.transform(initial_input)
    target_1_scaled = scaler_y.transform(target_1)
    autoreg_input_scaled = scaler_X.transform(autoreg_input)
    target_2_scaled = scaler_y.transform(target_2)

    # Convert to tensors for LSTM
    X_test_list.append({
        "initial_input": torch.tensor(initial_input_scaled, dtype=torch.float32).unsqueeze(0),
        "autoreg_input": torch.tensor(autoreg_input_scaled, dtype=torch.float32).unsqueeze(0)})
    y_test_list.append({
        "target_1": torch.tensor(target_1_scaled, dtype=torch.float32),
        "target_2": torch.tensor(target_2_scaled, dtype=torch.float32)})




#backward model
X_train_list_backwards, y_train_list_backwards = [], []
X_test_list_backwards, y_test_list_backwards = [], []

# Define the correct lengths for interpolated data
num_original_rows = 40
num_interpolated_rows = 40 + 39  # Original rows + interpolated rows
interpolated_indices = list(range(num_interpolated_rows))

# Prepare for training
for df in train_dfs:
    data = df[cols].iloc[::-1].values  # Reverse the rows for backward prediction

    # Define correct indices for backward LSTM after interpolation
    initial_input_indices = interpolated_indices[-20:]  # Last 20 rows (reverse order)
    autoreg_input_indices = interpolated_indices[-30:-10]  # Rows 30-10 from the back
    target_1_indices = interpolated_indices[-30:-20]  # Rows 30-20 from the back
    target_2_indices = interpolated_indices[-40:-30]  # Rows 40-30 from the back

    # Get the corresponding data
    initial_input = data[initial_input_indices]
    autoreg_input = data[autoreg_input_indices]
    target_1 = data[target_1_indices]
    target_2 = data[target_2_indices]

    # Scale the inputs and targets
    initial_input_scaled = scaler_X.transform(initial_input)
    autoreg_input_scaled = scaler_X.transform(autoreg_input)
    target_1_scaled = scaler_y.transform(target_1)
    target_2_scaled = scaler_y.transform(target_2)

    # Convert to tensors for backward LSTM
    X_train_list_backwards.append({
        "initial_input": torch.tensor(initial_input_scaled, dtype=torch.float32).unsqueeze(0),
        "autoreg_input": torch.tensor(autoreg_input_scaled, dtype=torch.float32).unsqueeze(0)
    })
    y_train_list_backwards.append({
        "target_1": torch.tensor(target_1_scaled, dtype=torch.float32),
        "target_2": torch.tensor(target_2_scaled, dtype=torch.float32)
    })

# Prepare for testing
for df in test_dfs:
    data = df[cols].iloc[::-1].values  # Reverse the rows for backward prediction

    # Define correct indices for backward LSTM after interpolation
    initial_input_indices = interpolated_indices[-20:]  # Last 20 rows (reverse order)
    autoreg_input_indices = interpolated_indices[-30:-10]  # Rows 30-10 from the back
    target_1_indices = interpolated_indices[-30:-20]  # Rows 30-20 from the back
    target_2_indices = interpolated_indices[-40:-30]  # Rows 40-30 from the back

    # Get the corresponding data
    initial_input = data[initial_input_indices]
    autoreg_input = data[autoreg_input_indices]
    target_1 = data[target_1_indices]
    target_2 = data[target_2_indices]

    # Scale the inputs and targets
    initial_input_scaled = scaler_X.transform(initial_input)
    autoreg_input_scaled = scaler_X.transform(autoreg_input)
    target_1_scaled = scaler_y.transform(target_1)
    target_2_scaled = scaler_y.transform(target_2)

    # Convert to tensors for backward LSTM
    X_test_list_backwards.append({
        "initial_input": torch.tensor(initial_input_scaled, dtype=torch.float32).unsqueeze(0),
        "autoreg_input": torch.tensor(autoreg_input_scaled, dtype=torch.float32).unsqueeze(0)})
    y_test_list_backwards.append({
        "target_1": torch.tensor(target_1_scaled, dtype=torch.float32),
        "target_2": torch.tensor(target_2_scaled, dtype=torch.float32)})
    
    
    
    
# =============================================================================
# LSTM model forward
# =============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, len(cols)) 

    def forward(self, x, h_0, c_0):
        # LSTM forward pass with previous hidden and cell states
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.dropout(out) #dropout after LSTM
        out = self.fc(out[:, -10:, :])  #Use last 10 timesteps and predict all target features
        return out, h_n, c_n  #Return updated hidden and cell states as well as output

# Hyperparameters
input_size = len(cols)  
hidden_size = 50  # Adaptable, number neurons in LSTM
output_size = len(cols)*10  # Predict all 6 features for 10 timesteps

# Initialize model and optimizer
model = LSTMModel(input_size, hidden_size, output_size, dropout = 0.3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#scheduler 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=3, 
                                                       verbose=True)

# Train
epochs = 50
best_val_loss = float('inf')
patience = 5
patience_counter = 0
for epoch in range(epochs):
    model.train()
    epoch_loss = 0  # Initialize loss per epoch

    # Initializing hidden and cell states (outside of loop for state retention)
    h_0 = torch.zeros(1, X_train_list[0]["initial_input"].size(0), hidden_size)  # (num_layers, batch_size, hidden_size)
    c_0 = torch.zeros(1, X_train_list[0]["initial_input"].size(0), hidden_size)  # (num_layers, batch_size, hidden_size)

    for X_train, y_train in zip(X_train_list, y_train_list):
        optimizer.zero_grad()
        
        #predict with target 1 and initial input
        initial_input = X_train["initial_input"]
        outputs_1, h_0, c_0 = model(initial_input, h_0.detach(), c_0.detach())
        target_1 = y_train["target_1"]
        loss_1 = criterion(outputs_1, target_1.unsqueeze(0))  # Add batch dimension
        
        #add output to input for next prediction
        autoreg_input = torch.cat((X_train["autoreg_input"][:, :-10, :], outputs_1[:, -10:, :]), dim=1)
        outputs_2, h_0, c_0 = model(autoreg_input, h_0, c_0)
        target_2 = y_train["target_2"]
        loss_2 = criterion(outputs_2, target_2.unsqueeze(0))  # Add batch dimension
        
        #combine losses from both steps
        total_loss = loss_1 + loss_2

        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
    
    #validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        h_0_val = torch.zeros(1, X_test_list[0]["initial_input"].size(0), hidden_size)
        c_0_val = torch.zeros(1, X_test_list[0]["initial_input"].size(0), hidden_size)

        for X_val, y_val in zip(X_test_list, y_test_list):
            #for initial input
            initial_input = X_val["initial_input"]
            outputs_1, h_0_val, c_0_val = model(initial_input, h_0_val, c_0_val)
            target_1 = y_val["target_1"]
            loss_1 = criterion(outputs_1, target_1.unsqueeze(0))
            
            #for second autoreg. input
            autoreg_input = torch.cat((X_val["autoreg_input"][:, :-10, :],
                                       outputs_1[:, -10:, :]), dim=1)
            outputs_2, h_0_val, c_0_val = model(autoreg_input, h_0_val, c_0_val)
            target_2 = y_val["target_2"]
            loss_2 = criterion(outputs_2, target_2.unsqueeze(0))
            
            val_loss += (loss_1 + loss_2).item()
            
    print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}')

    # Scheduler Schritt
    scheduler.step(val_loss)

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


#test model 
prediction_lstm = []  
model.eval()
test_loss = 0
with torch.no_grad():
    h_0 = torch.zeros(1, X_test_list[0]["initial_input"].size(0), hidden_size)  
    c_0 = torch.zeros(1, X_test_list[0]["initial_input"].size(0), hidden_size)  
    
    for X_test, y_test in zip(X_test_list, y_test_list):
        #initial input
        initial_input = X_test["initial_input"]
        outputs_1, h_0, c_0 = model(initial_input, h_0, c_0)

        target_1 = y_test["target_1"]
        predicted_1 = scaler_y.inverse_transform(outputs_1.cpu().numpy().reshape(-1, len(cols)))
        actual_1 = scaler_y.inverse_transform(target_1.cpu().numpy().reshape(-1, len(cols)))
        loss_1 = criterion(torch.tensor(predicted_1, dtype=torch.float32), torch.tensor(actual_1, dtype=torch.float32))
        test_loss += loss_1.item()

        #autoregressive input
        autoreg_input = torch.cat((X_test["autoreg_input"][:, :-10, :], outputs_1[:, -10:, :]), dim=1)
        outputs_2, h_0, c_0 = model(autoreg_input, h_0, c_0)

        target_2 = y_test["target_2"]
        predicted_2 = scaler_y.inverse_transform(outputs_2.cpu().numpy().reshape(-1, len(cols)))
        actual_2 = scaler_y.inverse_transform(target_2.cpu().numpy().reshape(-1, len(cols)))
        loss_2 = criterion(torch.tensor(predicted_2, dtype=torch.float32), torch.tensor(actual_2, dtype=torch.float32))
        test_loss += loss_2.item()

        # collect predictions
        for i in range(len(predicted_1)):
            prediction = {
                **{f'Predicted {col} (Step 1)': predicted_1[i][j] for j, col in enumerate(cols)},
                **{f'Predicted {col} (Step 2)': predicted_2[i][j] for j, col in enumerate(cols)}
            }
            actual_values = {
                **{f'Actual {col} (Step 1)': actual_1[i][j] for j, col in enumerate(cols)},
                **{f'Actual {col} (Step 2)': actual_2[i][j] for j, col in enumerate(cols)}
            }
            prediction_lstm.append({**prediction, **actual_values})

print(f"Total Test Loss: {test_loss:.4f}")
results_df = pd.DataFrame(prediction_lstm)
 
 
# # save model
# save_path = os.path.join(r'C:\Users\Models', 
#                           'LSTM-tensteps-allfeatures-autoreg-intdata.pth')
# torch.save(model.state_dict(), save_path)


# #load model:
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
save_path = os.path.join('C:/Users/Models', 
                          'LSTM-tensteps-allfeatures-autoreg-intdata.pth')
model.load_state_dict(torch.load(save_path))
model.eval()    



# =============================================================================
# LSTM Model backwards
# =============================================================================

class LSTMModelBackwards(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super(LSTMModelBackwards, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, len(cols)) 

    def forward(self, x, h_0, c_0):
        # LSTM forward pass with previous hidden and cell states
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.dropout(out) #dropout after LSTM
        out = self.fc(out[:, -10:, :])  #Use last 10 timesteps and predict all target features
        return out, h_n, c_n  #Return updated hidden and cell states as well as output

# Hyperparameters
input_size = len(cols)  
hidden_size = 50  # Adaptable, number neurons in LSTM
output_size = len(cols)*10  # Predict all 6 features for 10 timesteps

# Initialize model and optimizer
model_backwards = LSTMModelBackwards(input_size, hidden_size, output_size, dropout=0.3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_backwards.parameters(), lr=0.001)

#scheduler 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=3, 
                                                       verbose=True)

# Train
epochs = 50
best_val_loss = float('inf')
patience = 5
patience_counter = 0
for epoch in range(epochs):
    model_backwards.train()
    epoch_loss = 0  # Initialize loss per epoch

    # Initializing hidden and cell states (outside of loop for state retention)
    h_0 = torch.zeros(1, X_train_list_backwards[0]["initial_input"].size(0), hidden_size)  # (num_layers, batch_size, hidden_size)
    c_0 = torch.zeros(1, X_train_list_backwards[0]["initial_input"].size(0), hidden_size)  # (num_layers, batch_size, hidden_size)

    for X_train, y_train in zip(X_train_list_backwards, y_train_list_backwards):
        optimizer.zero_grad()
        
        #predict with target 1 and initial input
        initial_input = X_train["initial_input"]
        outputs_1, h_0, c_0 = model_backwards(initial_input, h_0.detach(), c_0.detach())
        target_1 = y_train["target_1"]
        loss_1 = criterion(outputs_1, target_1.unsqueeze(0))  # Add batch dimension
        
        #add output to input for next prediction
        autoreg_input = torch.cat((X_train["autoreg_input"][:, :-10, :], outputs_1[:, -10:, :]), dim=1)
        outputs_2, h_0, c_0 = model_backwards(autoreg_input, h_0, c_0)
        target_2 = y_train["target_2"]
        loss_2 = criterion(outputs_2, target_2.unsqueeze(0))  # Add batch dimension
        
        #combine losses from both steps
        total_loss = loss_1 + loss_2

        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
    
    #validation
    model_backwards.eval()
    val_loss = 0
    with torch.no_grad():
        h_0_val = torch.zeros(1, X_test_list_backwards[0]["initial_input"].size(0), hidden_size)
        c_0_val = torch.zeros(1, X_test_list_backwards[0]["initial_input"].size(0), hidden_size)

        for X_val, y_val in zip(X_test_list_backwards, y_test_list_backwards):
            #for initial input
            initial_input = X_val["initial_input"]
            outputs_1, h_0_val, c_0_val = model_backwards(initial_input, h_0_val, c_0_val)
            target_1 = y_val["target_1"]
            loss_1 = criterion(outputs_1, target_1.unsqueeze(0))
            
            #for second autoreg. input
            autoreg_input = torch.cat((X_val["autoreg_input"][:, :-10, :],
                                       outputs_1[:, -10:, :]), dim=1)
            outputs_2, h_0_val, c_0_val = model_backwards(autoreg_input, h_0_val, c_0_val)
            target_2 = y_val["target_2"]
            loss_2 = criterion(outputs_2, target_2.unsqueeze(0))
            
            val_loss += (loss_1 + loss_2).item()
            
    print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}')

    # Scheduler Schritt
    scheduler.step(val_loss)

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


#test model 
prediction_lstm_backwards = []  
model_backwards.eval()
test_loss = 0
with torch.no_grad():
    h_0 = torch.zeros(1, X_test_list_backwards[0]["initial_input"].size(0), hidden_size)  
    c_0 = torch.zeros(1, X_test_list_backwards[0]["initial_input"].size(0), hidden_size)  
    
    for X_test, y_test in zip(X_test_list_backwards, y_test_list_backwards):
        #initial input
        initial_input = X_test["initial_input"]
        outputs_1, h_0, c_0 = model_backwards(initial_input, h_0, c_0)

        target_1 = y_test["target_1"]
        predicted_1 = scaler_y.inverse_transform(outputs_1.cpu().numpy().reshape(-1, len(cols)))
        actual_1 = scaler_y.inverse_transform(target_1.cpu().numpy().reshape(-1, len(cols)))
        loss_1 = criterion(torch.tensor(predicted_1, dtype=torch.float32), torch.tensor(actual_1, dtype=torch.float32))
        test_loss += loss_1.item()

        #autoregressive input
        autoreg_input = torch.cat((X_test["autoreg_input"][:, :-10, :], outputs_1[:, -10:, :]), dim=1)
        outputs_2, h_0, c_0 = model_backwards(autoreg_input, h_0, c_0)

        target_2 = y_test["target_2"]
        predicted_2 = scaler_y.inverse_transform(outputs_2.cpu().numpy().reshape(-1, len(cols)))
        actual_2 = scaler_y.inverse_transform(target_2.cpu().numpy().reshape(-1, len(cols)))
        loss_2 = criterion(torch.tensor(predicted_2, dtype=torch.float32), torch.tensor(actual_2, dtype=torch.float32))
        test_loss += loss_2.item()

        # collect predictions
        for i in range(len(predicted_1)):
            prediction = {
                **{f'Predicted {col} (Step 1)': predicted_1[i][j] for j, col in enumerate(cols)},
                **{f'Predicted {col} (Step 2)': predicted_2[i][j] for j, col in enumerate(cols)}
            }
            actual_values = {
                **{f'Actual {col} (Step 1)': actual_1[i][j] for j, col in enumerate(cols)},
                **{f'Actual {col} (Step 2)': actual_2[i][j] for j, col in enumerate(cols)}
            }
            prediction_lstm_backwards.append({**prediction, **actual_values})

print(f"Total Test Loss: {test_loss:.4f}")
results_df_backwards = pd.DataFrame(prediction_lstm_backwards)


 
# # save model
# save_path = os.path.join(r'C:\Users\Models', 
#                           'LSTM-tensteps-allfeatures-autoreg-backwards-int.pth')
# torch.save(model_backwards.state_dict(), save_path)

model_backwards = LSTMModelBackwards(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
save_path = os.path.join('C:/Users/Models', 
                          'LSTM-tensteps-allfeatures-autoreg-backwards-int.pth')
model.load_state_dict(torch.load(save_path))
model.eval()    






# =============================================================================
# try on long boreholes
# =============================================================================


# Load raw data
path_boreholes_raw = Path(r'C:\Users\TestHoles')
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

# Function to fill up NaN values using the LSTM model for all 6 columns
def predict_and_fill_nans(model, data, hidden_size, scaler_y, scaler_X, cols):
    output_data = []

    for df in data:
        # Copy original values for each column into new columns "Original {col}"
        for col in cols:
            original_col = f"Original {col}"
            if original_col not in df.columns:
                df[original_col] = df[col]

        # Sort by "Total Depth [m]" and reset the index
        df = df.sort_values(by="Total Depth [m]").reset_index(drop=True)

        # Find NaN segments based on one of the target columns (assuming NaNs are synchronized)
        nan_segments = df[cols[0]].isna()  # Assuming NaNs are synchronized across all target columns

        i = 20  # Start point, since at least 20 rows are needed for input
        while i < len(df):
            if nan_segments[i]:  # If a NaN segment is found
                nan_start = i

                # Determine the length of the NaN segment
                while i < len(df) and nan_segments[i]:
                    i += 1
                nan_end = i
                nan_length = nan_end - nan_start

                # Prediction is only possible if the previous 20 rows have no NaNs
                while nan_length > 0:
                    # Prepare input data
                    input_data = df[cols].iloc[nan_start - 20:nan_start].values
                    input_data = scaler_X.transform(input_data)  # Scale input data
                    input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

                    # Initialize hidden states
                    h_0 = torch.zeros(1, input_data.size(0), hidden_size)
                    c_0 = torch.zeros(1, input_data.size(0), hidden_size)

                    # Predict the next 10 values
                    with torch.no_grad():
                        predicted_output, h_0, c_0 = model(input_data, h_0, c_0)

                    # Inverse scale the predictions
                    predicted_output = scaler_y.inverse_transform(predicted_output.cpu().numpy().reshape(-1, len(cols)))

                    # Fill NaN values in the DataFrame for all target columns
                    num_to_fill = min(nan_length, 10)
                    for j, col in enumerate(cols):
                        df.iloc[nan_start:nan_start + num_to_fill, df.columns.get_loc(col)] = predicted_output[:num_to_fill, j]

                    # Update iteration variables
                    nan_start += num_to_fill
                    nan_length -= num_to_fill

            else:
                i += 1  # Move to the next row if no NaN is present

        # Save the updated DataFrame
        output_data.append(df)

    return output_data

# function to fill up with backwards model
def predict_and_fill_nans_backwards(model, data, hidden_size, scaler_y, scaler_X, cols):
    output_data = []
    for df in data:
        # Copy original values for each column into new columns "Original {col}"
        for col in cols:
            original_col = f"Original {col}"
            if original_col not in df.columns:
                df[original_col] = df[col]

        # Sort by "Total Depth [m]" and reset the index
        df = df.sort_values(by="Total Depth [m]", ascending=False).reset_index(drop=True)  # Reverse order for backwards prediction

        #find nans
        nan_segments = df[cols[0]].isna()
        i = 20  # Start point, since at least 20 rows are needed for input
        while i < len(df):
            if nan_segments[i]:  # If a NaN segment is found
                nan_start = i

                # Determine the length of the NaN segment
                while i < len(df) and nan_segments[i]:
                    i += 1
                nan_end = i
                nan_length = nan_end - nan_start

                # Prediction is only possible if the previous 20 rows have no NaNs
                while nan_length > 0:
                    # Prepare input data
                    input_data = df[cols].iloc[nan_start - 20:nan_start].values
                    input_data = scaler_X.transform(input_data)
                    input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

                    # Initialize hidden states
                    h_0 = torch.zeros(1, input_data.size(0), hidden_size)
                    c_0 = torch.zeros(1, input_data.size(0), hidden_size)

                    # Predict the next 10 values
                    with torch.no_grad():
                        predicted_output, h_0, c_0 = model(input_data, h_0, c_0)

                    # Inverse scale the predictions
                    predicted_output = scaler_y.inverse_transform(predicted_output.cpu().numpy().reshape(-1, len(cols)))

                    # Fill NaN values in the DataFrame for all target columns
                    num_to_fill = min(nan_length, 10)
                    for j, col in enumerate(cols):
                        df.iloc[nan_start:nan_start + num_to_fill, df.columns.get_loc(col)] = predicted_output[:num_to_fill, j]

                    # Update iteration variables
                    nan_start += num_to_fill
                    nan_length -= num_to_fill

            else:
                i += 1

        # Sort the DataFrame back to the original order
        df = df.sort_values(by="Total Depth [m]").reset_index(drop=True)
        output_data.append(df)

    return output_data

#combine by taking average values
def combine_predictions(forward_data, backward_data, cols):
    combined_data = []

    for forward_df, backward_df in zip(forward_data, backward_data):
        #sort both by total depth
        forward_df = forward_df.sort_values(by="Total Depth [m]").reset_index(drop=True)
        backward_df = backward_df.sort_values(by="Total Depth [m]").reset_index(drop=True)

        combined_df = forward_df.copy()

        for col in cols:
            forward_values = forward_df[col].values  
            backward_values = backward_df[col].values  
            original_values = forward_df[f"Original {col}"].values 

            combined_column = []

            for orig, forward, backward in zip(original_values, forward_values, backward_values):
                if pd.isna(orig):  #if original value is nan --> combine
                    combined_column.append((forward + backward) / 2)
                else: #else keep original value
                    combined_column.append(orig)

            combined_df[col] = combined_column

        combined_data.append(combined_df)

    return combined_data

#plot boreholes
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


# Applying the function to clean data
holes_mwd_raw_LSTM = predict_and_fill_nans(model, holes_mwd_without_overlap, 
                                           hidden_size, scaler_y, scaler_X, cols)
holes_mwd_LSTM_backwards = predict_and_fill_nans_backwards(model_backwards, holes_mwd_without_overlap, 
                                           hidden_size, scaler_y, scaler_X, cols)
holes_mwd_LSTM_combined = combine_predictions(holes_mwd_raw_LSTM, 
                                              holes_mwd_LSTM_backwards, cols)

plot_holes_with_depth(holes_mwd_LSTM_combined, depth_min=2, depth_max=4.5)
plot_holes_with_depth(holes_mwd_raw_LSTM, depth_min=2, depth_max=4.5)
plot_holes_with_depth(holes_mwd_LSTM_backwards, depth_min=2, depth_max=4.5)



# =============================================================================
# check trainings data
# =============================================================================

#FORWARD

def predict_from_depth_multi_forward(model, data, hidden_size, 
                             scaler_y, scaler_X, cols):
    output_data = []

    for df in data:
        # Keep real values
        for col in cols:
            original_col = f"Original {col}"
            if original_col not in df.columns:
                df[original_col] = df[col]
        df = df.sort_values(by="Total Depth [m]").reset_index(drop=True)

        # Save predictions separately
        for col in cols:
            predicted_col = f"{col} predicted"
            if predicted_col not in df.columns:
                df[predicted_col] = np.nan
                
        input_data = df[cols].iloc[0:20].values
        input_data_scaled = scaler_X.transform(input_data)
        input_data_scaled = torch.tensor(input_data_scaled, dtype=torch.float32).unsqueeze(0)

        h_0 = torch.zeros(1, input_data_scaled.size(0), hidden_size)
        c_0 = torch.zeros(1, input_data_scaled.size(0), hidden_size)

        #first prediction
        with torch.no_grad():
            predicted_output, h_0, c_0 = model(input_data_scaled, h_0, c_0)

        predicted_output = scaler_y.inverse_transform(predicted_output.cpu().numpy().reshape(-1, len(cols)))

        for j, col in enumerate(cols):
            predicted_col = f"{col} predicted"
            df.iloc[20:30, df.columns.get_loc(predicted_col)] = predicted_output[:10, j]

         # second prediction
        input_data = np.concatenate([
            df[cols].iloc[10:20].values,  
            predicted_output[:10, :]  ])
        input_data_scaled = scaler_X.transform(input_data)
        input_data_scaled = torch.tensor(input_data_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            predicted_output, h_0, c_0 = model(input_data_scaled, h_0, c_0)

        predicted_output = scaler_y.inverse_transform(predicted_output.cpu().numpy().reshape(-1, len(cols)))

        for j, col in enumerate(cols):
            predicted_col = f"{col} predicted"
            df.iloc[30:40, df.columns.get_loc(predicted_col)] = predicted_output[:10, j]

        output_data.append(df)

    return output_data



import matplotlib.colors as mcolors
def lighten_color(color, amount=0.5):
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        c = mcolors.CSS4_COLORS[color]
    return tuple(1 - (1 - channel) * amount for channel in c)

def plot_holes_by_depth(holes_mwd_raw_LSTM, depth_min, depth_max, 
                        colors=['b', 'g', 'r'], indices_to_plot=[0, 1, 2]):

    # Generate lighter colors for predictions
    lighter_colors = [lighten_color(c, amount=0.5) for c in colors]

    for idx, (color, lighter_color) in zip(indices_to_plot, zip(colors, lighter_colors)):
        df = holes_mwd_raw_LSTM[idx]
        hole_id = df["Hole ID"].iloc[0]
        
        # Filter by specified depth range
        combined_df = df[(df['Total Depth [m]'] >= depth_min) & (df['Total Depth [m]'] <= depth_max)]

        columns = combined_df.columns[3:9]
        fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(10, 15))
        fig.suptitle(f"Testdata Hole ID {hole_id} (with target row + flushing flow) Forward Prediction", fontsize=16)
        all_handles, all_labels = [], []

        for i, column in enumerate(columns):
            ax = axes[i]
            x_original, y_original = [], []  # Original values
            x_predicted, y_predicted = [], []  # Predicted values

            for _, row in combined_df.iterrows():
                x_depth = row['Total Depth [m]']
                y_pred = row[f"{column} predicted"]
                y_orig = row[f"Original {column}"]

                if pd.notna(y_orig):  # Original value exists
                    x_original.append(x_depth)
                    y_original.append(y_orig)
                if pd.notna(y_pred):  # Predicted value exists
                    x_predicted.append(x_depth)
                    y_predicted.append(y_pred)

            # Plot predicted values as lighter stars in the background
            pred_handle, = ax.plot(x_predicted, y_predicted, marker='*', linestyle='', 
                                   color=lighter_color, alpha= 0.7, label='Predicted', 
                                   markersize=10)

            # Plot original values as points in the foreground
            orig_handle, = ax.plot(x_original, y_original, marker='o', linestyle='', 
                                   color=color, label='Original', markersize=4)

            # Collect handles and labels for the legend (only if not already added)
            if 'Original' not in all_labels and 'Predicted' not in all_labels:
                all_handles.extend([orig_handle, pred_handle])
                all_labels.extend(['Original', 'Predicted'])

            ax.set_xlabel('Total Depth [m]')
            ax.set_ylabel(column)
            ax.set_title(f'{column}')

        # Add the legend outside the plot area
        fig.legend(all_handles, all_labels, bbox_to_anchor=(0.5, 1.05), loc='upper center', ncol=len(all_labels))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

test_mwd_holes = []
for df in test_dfs:
    df_copy = df.copy()
    test_mwd_holes.append(df_copy)
        
for df in test_mwd_holes:
    df['Total Depth [m]'] = df['Depth [m]']

test_holes_LSTM_forward = predict_from_depth_multi_forward(model=model, data= test_mwd_holes, 
                                           hidden_size=hidden_size, scaler_y=scaler_y, 
                                           scaler_X=scaler_X, cols=cols)
plot_holes_by_depth(test_holes_LSTM_forward, depth_min=0.6, depth_max=1.2,indices_to_plot=[5,8,9])
plot_holes_by_depth(test_holes_LSTM_forward, depth_min=0.6, depth_max=1.2,indices_to_plot=[50,105,568])



#BACKWARD
def predict_from_depth_multi_backward(model, data, hidden_size, scaler_y, scaler_X, cols):
    output_data = []

    for df in data:
        for col in cols:
            original_col = f"Original {col}"
            if original_col not in df.columns:
                df[original_col] = df[col]

        df = df.iloc[::-1].reset_index(drop=True)

        for col in cols:
            predicted_col = f"{col} predicted"
            if predicted_col not in df.columns:
                df[predicted_col] = np.nan

        initial_input = df[cols].iloc[-20:].values
        initial_input_scaled = scaler_X.transform(initial_input)
        initial_input_scaled = torch.tensor(initial_input_scaled, dtype=torch.float32).unsqueeze(0)

        h_0 = torch.zeros(1, initial_input_scaled.size(0), hidden_size)
        c_0 = torch.zeros(1, initial_input_scaled.size(0), hidden_size)

        with torch.no_grad():
            predicted_output, h_0, c_0 = model(initial_input_scaled, h_0, c_0)

        predicted_output = scaler_y.inverse_transform(predicted_output.cpu().numpy().reshape(-1, len(cols)))

        for j, col in enumerate(cols):
            predicted_col = f"{col} predicted"
            df.iloc[-30:-20, df.columns.get_loc(predicted_col)] = predicted_output[:10, j]

            nan_indices = df.iloc[-30:-20][col].isna()
            if nan_indices.any():
                indices_to_fill = nan_indices[nan_indices].index
                values_to_fill = predicted_output[:10, j][nan_indices.values]
                df.loc[indices_to_fill, col] = values_to_fill

        autoreg_input = np.concatenate([
            df[cols].iloc[-30:-20].values, 
            predicted_output[:10, :]])
        autoreg_input_scaled = scaler_X.transform(autoreg_input)
        autoreg_input_scaled = torch.tensor(autoreg_input_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            predicted_output, h_0, c_0 = model(autoreg_input_scaled, h_0, c_0)

        predicted_output = scaler_y.inverse_transform(predicted_output.cpu().numpy().reshape(-1, len(cols)))

        for j, col in enumerate(cols):
            predicted_col = f"{col} predicted"
            df.iloc[-40:-30, df.columns.get_loc(predicted_col)] = predicted_output[:10, j]

            nan_indices = df.iloc[-40:-30][col].isna()
            if nan_indices.any():
                indices_to_fill = nan_indices[nan_indices].index
                values_to_fill = predicted_output[:10, j][nan_indices.values]
                df.loc[indices_to_fill, col] = values_to_fill
        df = df.iloc[::-1].reset_index(drop=True)
        output_data.append(df)
    return output_data

def plot_holes_by_depth_back(holes_mwd_raw_LSTM, depth_min, depth_max, 
                        colors=['b', 'g', 'r'], indices_to_plot=[0, 1, 2]):

    # Generate lighter colors for predictions
    lighter_colors = [lighten_color(c, amount=0.5) for c in colors]

    for idx, (color, lighter_color) in zip(indices_to_plot, zip(colors, lighter_colors)):
        df = holes_mwd_raw_LSTM[idx]
        hole_id = df["Hole ID"].iloc[0]
        
        # Filter by specified depth range
        combined_df = df[(df['Total Depth [m]'] >= depth_min) & (df['Total Depth [m]'] <= depth_max)]

        columns = combined_df.columns[3:9]
        fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(10, 15))
        fig.suptitle(f"Testdata Hole ID {hole_id} (with target row + flushing flow) Backward Prediction", fontsize=16)
        all_handles, all_labels = [], []

        for i, column in enumerate(columns):
            ax = axes[i]
            x_original, y_original = [], []  # Original values
            x_predicted, y_predicted = [], []  # Predicted values

            for _, row in combined_df.iterrows():
                x_depth = row['Total Depth [m]']
                y_pred = row[f"{column} predicted"]
                y_orig = row[f"Original {column}"]

                if pd.notna(y_orig):  # Original value exists
                    x_original.append(x_depth)
                    y_original.append(y_orig)
                if pd.notna(y_pred):  # Predicted value exists
                    x_predicted.append(x_depth)
                    y_predicted.append(y_pred)

            # Plot predicted values as lighter stars in the background
            pred_handle, = ax.plot(x_predicted, y_predicted, marker='*', linestyle='', 
                                   color=lighter_color, alpha= 0.7, label='Predicted', 
                                   markersize=10)

            # Plot original values as points in the foreground
            orig_handle, = ax.plot(x_original, y_original, marker='o', linestyle='', 
                                   color=color, label='Original', markersize=4)

            # Collect handles and labels for the legend (only if not already added)
            if 'Original' not in all_labels and 'Predicted' not in all_labels:
                all_handles.extend([orig_handle, pred_handle])
                all_labels.extend(['Original', 'Predicted'])

            ax.set_xlabel('Total Depth [m]')
            ax.set_ylabel(column)
            ax.set_title(f'{column}')

        # Add the legend outside the plot area
        fig.legend(all_handles, all_labels, bbox_to_anchor=(0.5, 1.05), loc='upper center', ncol=len(all_labels))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


test_holes_LSTM_backward = predict_from_depth_multi_backward(model=model_backwards, data= test_mwd_holes, 
                                           hidden_size=hidden_size, scaler_y=scaler_y, 
                                           scaler_X=scaler_X, cols=cols)
plot_holes_by_depth_back(test_holes_LSTM_backward, depth_min=0.6, depth_max=1.2,indices_to_plot=[5,8,9])
plot_holes_by_depth_back(test_holes_LSTM_backward, depth_min=0.6, depth_max=1.2,indices_to_plot=[50,105,568])



# =============================================================================
# Analysis of model training
# =============================================================================

#forward
output_path = Path(r'C:\Users\Models\With Flushing Flow\forward-LSTM')

result_data_list_lstm_forward = {"Step 1": {}, "Step 2": {}, "Combined": {}}
max_errors = {}

for step in ["Step 1", "Step 2"]:
    for col in cols:
        actual_col = f'Actual {col} ({step})'
        predicted_col = f'Predicted {col} ({step})'

        # Metrics calculation
        mse_lstm = mean_squared_error(results_df[actual_col], results_df[predicted_col])
        rmse_lstm = mse_lstm ** 0.5
        r2_lstm = r2_score(results_df[actual_col], results_df[predicted_col])
        sym_mape_lstm = smape(results_df[actual_col], results_df[predicted_col])
        accuracy_lstm = 100 - sym_mape_lstm
        max_error = np.max(np.abs(results_df[actual_col] - results_df[predicted_col]))
        mape_lstm = np.mean(np.abs((results_df[actual_col] - results_df[predicted_col]) / results_df[actual_col])) * 100

        result_data_list_lstm_forward[step][col] = {
            "MSE": mse_lstm,
            "RMSE": rmse_lstm,
            "R2": r2_lstm,
            "Accuracy": accuracy_lstm,
            "Max Error": max_error,
            "MAPE": mape_lstm}

        #scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(results_df[actual_col], results_df[predicted_col], alpha=0.6)
        plt.title(f'Scatter Plot LSTM ({step}) for predicting {col}')
        plt.xlabel(f'Actual {col} ({step})')
        plt.ylabel(f'Predicted {col} ({step})')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        scatter_plot_path = output_path / f"scatter_plot_{col.replace('[', '').replace(']', '').replace('/', '_').strip()}_{step}.png"
        plt.savefig(scatter_plot_path)
        plt.close()

results_path = os.path.join(output_path, "results-forward_new.csv")
all_results = []
for step, step_results in result_data_list_lstm_forward.items():
    for col, metrics in step_results.items():
        row = {"Step": step, "Column": col, **metrics}
        all_results.append(row)
pd.DataFrame(all_results).to_csv(results_path, index=False)



#for backward
output_path_backward = Path(r'C:\Users\Models\With Flushing Flow\backward-LSTM')


result_data_list_lstm_backward = {"Step 1": {}, "Step 2": {}, "Combined": {}}
max_errors_backward = {}

for step in ["Step 1", "Step 2"]:
    for col in cols:
        actual_col = f'Actual {col} ({step})'
        predicted_col = f'Predicted {col} ({step})'

        # Metrics calculation
        mse_lstm = mean_squared_error(results_df_backwards[actual_col], results_df_backwards[predicted_col])
        rmse_lstm = mse_lstm ** 0.5
        r2_lstm = r2_score(results_df_backwards[actual_col], results_df_backwards[predicted_col])
        sym_mape_lstm = smape(results_df_backwards[actual_col], results_df_backwards[predicted_col])
        accuracy_lstm = 100 - sym_mape_lstm
        max_error = np.max(np.abs(results_df_backwards[actual_col] - results_df_backwards[predicted_col]))
        mape_lstm = np.mean(np.abs((results_df_backwards[actual_col] - results_df_backwards[predicted_col]) / results_df_backwards[actual_col])) * 100
        
        
        result_data_list_lstm_backward[step][col] = {
            "MSE": mse_lstm,
            "RMSE": rmse_lstm,
            "R2": r2_lstm,
            "Accuracy": accuracy_lstm,
            "Max Error": max_error,
            "MAPE":mape_lstm}

        # Scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(results_df_backwards[actual_col], results_df_backwards[predicted_col], alpha=0.6)
        plt.title(f'Scatter Plot LSTM Backward ({step}) for predicting {col}')
        plt.xlabel(f'Actual {col} ({step})')
        plt.ylabel(f'Predicted {col} ({step})')

        # Add diagonal line for reference
        min_val = min(results_df_backwards[actual_col].min(), results_df_backwards[predicted_col].min())
        max_val = max(results_df_backwards[actual_col].max(), results_df_backwards[predicted_col].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.tight_layout()
        scatter_plot_path = output_path_backward / f"scatter_plot_{col.replace('[', '').replace(']', '').replace('/', '_').strip()}_{step}.png"
        plt.savefig(scatter_plot_path)
        plt.close()

results_path_backward = output_path_backward / "results-backward_new.csv"
all_results_backward = []
for step, step_results in result_data_list_lstm_backward.items():
    for col, metrics in step_results.items():
        row = {"Step": step, "Column": col, **metrics}
        all_results_backward.append(row)
pd.DataFrame(all_results_backward).to_csv(results_path_backward, index=False)



# =============================================================================
# Combine trainingsdata for models
# =============================================================================
def combine_predictions(forward_data, backward_data, cols):
    combined_data = []

    for forward_df, backward_df in zip(forward_data, backward_data):
        combined_df = forward_df.copy()

        for col in cols:
            forward_col = f"{col} predicted"
            backward_col = f"{col} predicted"

            # Combine predictions: Forward for 20:30, Backward for 30:40
            combined_df.loc[20:30, forward_col] = forward_df.loc[20:30, forward_col]
            combined_df.loc[30:40, forward_col] = backward_df.loc[30:40, backward_col]

        combined_data.append(combined_df)

    return combined_data

def plot_combined_predictions(data, depth_min, depth_max, cols, colors=['b', 'g', 'r'], indices_to_plot=[0, 1, 2]):
    lighter_colors = [lighten_color(c, amount=0.5) for c in colors]

    for idx, (color, lighter_color) in zip(indices_to_plot, zip(colors, lighter_colors)):
        df = data[idx]
        hole_id = df["Hole ID"].iloc[0]

        combined_df = df[(df['Total Depth [m]'] >= depth_min) & (df['Total Depth [m]'] <= depth_max)]

        columns = combined_df.columns[3:9]
        fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(10, 15))
        fig.suptitle(f"Combined Predictions - Hole ID {hole_id}", fontsize=16)
        all_handles, all_labels = [], []

        for i, column in enumerate(columns):
            ax = axes[i]
            x_original, y_original = [], []  # Original values
            x_predicted, y_predicted = [], []  # Predicted values

            for _, row in combined_df.iterrows():
                x_depth = row['Total Depth [m]']
                y_pred = row[f"{column} predicted"]
                y_orig = row[f"Original {column}"]

                if pd.notna(y_orig):  # Original value exists
                    x_original.append(x_depth)
                    y_original.append(y_orig)
                if pd.notna(y_pred):  # Predicted value exists
                    x_predicted.append(x_depth)
                    y_predicted.append(y_pred)

            # Plot predicted values
            pred_handle, = ax.plot(x_predicted, y_predicted, marker='*', linestyle='', 
                                   color=lighter_color, alpha=0.7, label='Predicted', markersize=10)

            # Plot original values
            orig_handle, = ax.plot(x_original, y_original, marker='o', linestyle='', 
                                   color=color, label='Original', markersize=4)

            if 'Original' not in all_labels and 'Predicted' not in all_labels:
                all_handles.extend([orig_handle, pred_handle])
                all_labels.extend(['Original', 'Predicted'])

            ax.set_xlabel('Total Depth [m]')
            ax.set_ylabel(column)
            ax.set_title(f'{column}')

        fig.legend(all_handles, all_labels, bbox_to_anchor=(0.5, 1.05), loc='upper center', ncol=len(all_labels))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

combined_predictions = combine_predictions(test_holes_LSTM_forward, test_holes_LSTM_backward, cols)


output_path_combined = Path(r'C:\Users\Models\forward-backward-LSTM')

result_data_list_lstm_combined = {}

for col in cols:
    actual_col = f'Original {col}'
    predicted_col = f'{col} predicted'

    # Metrics calculation
    mse_lstm = mean_squared_error(combined_predictions[0][actual_col][20:40], combined_predictions[0][predicted_col][20:40])
    rmse_lstm = mse_lstm ** 0.5
    r2_lstm = r2_score(combined_predictions[0][actual_col][20:40], combined_predictions[0][predicted_col][20:40])
    sym_mape_lstm = smape(combined_predictions[0][actual_col][20:40], combined_predictions[0][predicted_col][20:40])
    accuracy_lstm = 100 - sym_mape_lstm
    max_error = np.max(np.abs(combined_predictions[0][actual_col][20:40] - combined_predictions[0][predicted_col][20:40]))
    mape_lstm = np.mean(np.abs((combined_predictions[0][actual_col][20:40] - combined_predictions[0][predicted_col][20:40]) / combined_predictions[0][actual_col][20:40])) * 100

    result_data_list_lstm_combined[col] = {
        "MSE": mse_lstm,
        "RMSE": rmse_lstm,
        "R2": r2_lstm,
        "Accuracy": accuracy_lstm,
        "Max Error": max_error,
        "MAPE": mape_lstm
    }


results_path_combined = output_path_combined / "results-combined-new2.csv"
all_results_combined = []
for col, metrics in result_data_list_lstm_combined.items():
    row = {"Column": col, **metrics}
    all_results_combined.append(row)

pd.DataFrame(all_results_combined).to_csv(results_path_combined, index=False)


plot_combined_predictions(combined_predictions, depth_min=0.6, depth_max=1.2,cols=cols,colors=['b', 'g', 'r'], indices_to_plot=[0,1,2])
plot_combined_predictions(combined_predictions, depth_min=0.6, depth_max=1.2,cols=cols,colors=['b', 'g', 'r'], indices_to_plot=[3,4,5])       






# =============================================================================
# threshold to flag results
# =============================================================================
#forward
#use standard deviation
residuals_summary = []

for step in ["Step 1", "Step 2"]:
    for col in cols:
        actual_col = f'Actual {col} ({step})'
        predicted_col = f'Predicted {col} ({step})'

        residuals = results_df[actual_col] - results_df[predicted_col]

        mean_residual = residuals.mean()
        std_residual = residuals.std()

        #flag outliers
        threshold_upper = mean_residual + 2 * std_residual
        threshold_lower = mean_residual - 2 * std_residual
        outlier_flags = ~residuals.between(threshold_lower, threshold_upper)

        residuals_summary.append({
            "Step": step,
            "Column": col,
            "Mean Residual": mean_residual,
            "Std Dev": std_residual,
            "Upper Threshold": threshold_upper,
            "Lower Threshold": threshold_lower,
            "Outliers Count": outlier_flags.sum(),
            "Outlier Percentage": 100 * outlier_flags.sum() / len(residuals)})

        # #write in original if it is flagged
        # results_df[f'Residual {col} ({step})'] = residuals
        # results_df[f'Outlier Flag {col} ({step})'] = outlier_flags

residuals_summary_df = pd.DataFrame(residuals_summary)


#backward flagging
residuals_summary_backward = []

for step in ["Step 1", "Step 2"]:
    for col in cols:
        actual_col = f'Actual {col} ({step})'
        predicted_col = f'Predicted {col} ({step})'

        residuals = results_df_backwards[actual_col] - results_df_backwards[predicted_col]

        mean_residual = residuals.mean()
        std_residual = residuals.std()

        #flag outliers
        threshold_upper = mean_residual + 2 * std_residual
        threshold_lower = mean_residual - 2 * std_residual
        outlier_flags = ~residuals.between(threshold_lower, threshold_upper)

        residuals_summary_backward.append({
            "Step": step,
            "Column": col,
            "Mean Residual": mean_residual,
            "Std Dev": std_residual,
            "Upper Threshold": threshold_upper,
            "Lower Threshold": threshold_lower,
            "Outliers Count": outlier_flags.sum(),
            "Outlier Percentage": 100 * outlier_flags.sum() / len(residuals)})

        # #write in original if it is flagged
        # results_df_backwards[f'Residual {col} ({step})'] = residuals
        # results_df_backwards[f'Outlier Flag {col} ({step})'] = outlier_flags

residuals_summary_backward_df = pd.DataFrame(residuals_summary_backward)



















