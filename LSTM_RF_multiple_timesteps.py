# -*- coding: utf-8 -*-
"""

@author: klein

prediction goal: next 10 time steps for just one feature
cleaned code, just for LSTM and basemodel (RF)

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor
import os
import joblib
import random


def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)


# =============================================================================
# load and prepare data
# =============================================================================

#path
path = Path(r'C:\Users\Data\00_created_Data\04_thirty_row_dataframes')

mwd_list = []

for file in sorted(path.glob('*.csv')): 
    df = pd.read_csv(file)
    mwd_list.append(df)  
 

#ensure data is equally spaced --> spaced by depth (only necessary for LSTM!
#transformers do not need that)

cols = ["Feed Pressure [Bar]", "Flushing Flow [l/min]", "Flushing Pressure [Bar]",
    "Penetration Rate [m/min]", "Percussion Pressure [Bar]", "Rotation Pressure [Bar]"]


#ensure each series is of the same length
filtered_list = []
for df in mwd_list:
    if len(df) == 30: #as 30 rows was the goal
        if not df[cols].isnull().values.any(): #ensure there are no nan values
            filtered_list.append(df)

mwd_list = filtered_list

# =============================================================================
# prepare LSTM Data / RF Data
# =============================================================================

cols = ["Feed Pressure [Bar]", "Flushing Flow [l/min]", "Flushing Pressure [Bar]",
    "Penetration Rate [m/min]", "Percussion Pressure [Bar]", "Rotation Pressure [Bar]"]

# train-test split
train_dfs, test_dfs = train_test_split(mwd_list, test_size=0.2, random_state=42)

# prepare scaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()



# collect all data to fit the scalers
all_train_data = []
all_train_targets = []

# prepare train data --> prediction goal: last three time steps
for df in train_dfs:
    X = df.iloc[:-10][cols].values  # everything except last ten rows is input
    y = df.iloc[-10:]['Penetration Rate [m/min]'].values  # last ten rows are target as numpy array
    all_train_data.append(X)
    all_train_targets.append(y)

# # fit scaler for X
# scaler_X.fit(np.vstack(all_train_data))

# # fit scaler for y
# scaler_y.fit(np.vstack(all_train_targets).reshape(-1, 1))

#save scaler for later
scaler_X_path = r'C:\Users\scaler_multi_X_X.pkl'
scaler_y_path = r'C:\Users\scaler_multi_EX_y.pkl'

# joblib.dump(scaler_X, scaler_X_path)
# joblib.dump(scaler_y, scaler_y_path)

#load scaler
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

# Initialize lists for LSTM and RandomForest data
X_train_list, y_train_list, depth_train_list = [], [], []
X_test_list, y_test_list, depth_test_list = [], [], []
X_train_rf, y_train_rf_steps = [], [[] for _ in range(10)]
X_test_rf, y_test_rf_steps = [], [[] for _ in range(10)]

# Convert training data
for df in train_dfs:
    # Prepare input (first 20 rows) and target (last 10 rows)
    X = df.iloc[:20][cols].values  # Use only first 20 rows as input
    y = df.iloc[-10:]['Penetration Rate [m/min]'].values  # Last 10 rows as target
    depths = df.iloc[:20]['Depth [m]'].values

    # Scale inputs and targets
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()

    # Use data for RandomForest
    X_train_rf.append(X_scaled.flatten())  # Flatten 20-row segment into single row for RF
    for i in range(10):  # Append each step of y to the correct list in y_train_rf_steps
        y_train_rf_steps[i].append(y_scaled[i])

    # Convert data to tensors for LSTM
    X_train_list.append(torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0))  # Unsqueeze for batch dimension
    y_train_list.append(torch.tensor(y_scaled, dtype=torch.float32))
    depth_train_list.append(torch.tensor(depths, dtype=torch.float32).unsqueeze(0))

# Convert lists to numpy arrays for RandomForest
X_train_rf = np.vstack(X_train_rf)  # Stack all X_train for RF
y_train_rf_steps = [np.array(y_step) for y_step in y_train_rf_steps]  # Convert each target step to an array


# prepare test data
for df in test_dfs:
    X = df.iloc[:-10][cols].values
    y = df.iloc[-10:]['Penetration Rate [m/min]'].values  
    depths = df.iloc[:-10]['Depth [m]'].values

    # scale
    X = scaler_X.transform(X)
    y = scaler_y.transform(y.reshape(-1, 1)).flatten() 
    
    # Use data for RandomForest
    X_test_rf.append(X.flatten())  # Flatten 20-row segment into single row for RF
    for i in range(10):  # Append each step of y to the correct list in y_train_rf_steps
        y_test_rf_steps[i].append(y[i])

    # convert to tensors
    X_test_list.append(torch.tensor(X, dtype=torch.float32).unsqueeze(0))
    y_test_list.append(torch.tensor(y, dtype=torch.float32))
    depth_test_list.append(torch.tensor(depths, dtype=torch.float32).unsqueeze(0))
# Convert lists to numpy arrays for RandomForest
X_test_rf = np.vstack(X_test_rf)
y_test_rf_steps = [np.array(y_step) for y_step in y_test_rf_steps]


#check scaler
original_and_rescaled = []
df = test_dfs[0]
y_original = df['Penetration Rate [m/min]'].values
y_scaled = scaler_y.transform(y_original.reshape(-1, 1)).flatten()
y_tensor = torch.tensor(y_scaled, dtype=torch.float32) #convert to pytorch just like input

y_rescaled = scaler_y.inverse_transform(y_tensor.cpu().numpy().reshape(-1, 1)).flatten()
for orig, rescaled in zip(y_original, y_rescaled):
    original_and_rescaled.append({'Original': orig, 'Rescaled': rescaled})



# =============================================================================
# LSTM model
# =============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)  # Add dropout layer to reduce overfitting
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0, c_0):
        # LSTM forward pass with previous hidden and cell states
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.dropout(out)  # Apply dropout after LSTM layer
        out = self.fc(out[:, -10:, :])  # Use last 10 timesteps
        out = out[:, :, 0]  # Output shape: (batch_size, 10)
        return out, h_n, c_n  # Return updated hidden and cell states as well as output

# Hyperparameters
input_size = len(cols)  # Number of input features
hidden_size = 50  # Can be tuned
output_size = 10  # Predicting 10 timesteps for a single feature

# Initialize model and optimizer
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=3, 
                                                       verbose=True)  # Reduce learning rate by half if val_loss does not improve for 3 epochs

#time the training process
import time
total_start_time = time.time()
# Training loop with early stopping
epochs = 50
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0  # Initialize loss per epoch

    # Initialize hidden and cell states for training
    h_0 = torch.zeros(1, X_train_list[0].size(0), hidden_size)
    c_0 = torch.zeros(1, X_train_list[0].size(0), hidden_size)

    for X_train, y_train in zip(X_train_list, y_train_list):
        optimizer.zero_grad()

        # Forward pass
        outputs, h_0, c_0 = model(X_train, h_0.detach(), c_0.detach())

        # Reshape y_train to match the output shape (batch_size, 10)
        y_train = y_train.view(X_train.size(0), 10)

        # Calculate loss and update weights
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.4f}')

    # Validation loop for early stopping
    model.eval()
    val_loss = 0
    with torch.no_grad():
        h_0_val = torch.zeros(1, X_test_list[0].size(0), hidden_size)
        c_0_val = torch.zeros(1, X_test_list[0].size(0), hidden_size)

        for X_val, y_val in zip(X_test_list, y_test_list):
            outputs, h_0_val, c_0_val = model(X_val, h_0_val, c_0_val)
            y_val = y_val.view(X_val.size(0), 10)
            val_loss += criterion(outputs, y_val).item()

    print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}')

    # Step the learning rate scheduler with the validation loss
    scheduler.step(val_loss)  # Adjusts learning rate if validation loss plateaus

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  # Reset counter, as we found a better model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
total_end_time = time.time()
print(f'Total Training Time: {total_end_time - total_start_time:.2f} seconds')





# test model
prediction_lstm = []  # predictions all in one

model.eval()
test_loss = 0
with torch.no_grad():
    h_0 = torch.zeros(1, X_test_list[0].size(0), hidden_size)  # Initialize test hidden state
    c_0 = torch.zeros(1, X_test_list[0].size(0), hidden_size)  # Initialize test cell state
    
    for X_test, y_test in zip(X_test_list, y_test_list):
        # Forward pass: Predict with previous hidden states
        test_outputs, h_0, c_0 = model(X_test, h_0, c_0)

        # Inverse scaling the predictions and actual values
        predicted = scaler_y.inverse_transform(test_outputs.cpu().numpy().reshape(-1, 1))
        actual = scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
        
        # Calculate loss between predicted and actual values (in unscaled space)
        loss = criterion(torch.tensor(predicted, dtype=torch.float32), torch.tensor(actual, dtype=torch.float32))
        test_loss += loss.item()
        
        # store predictions and actual values
        for i in range(len(predicted)):
            prediction_lstm.append({
                'Predicted Penetration Rate [m/min]': predicted[i][0], 
                'Actual Penetration Rate [m/min]': actual[i][0]
            })
            
            print(f"Prediction {i+1}: {predicted[i][0]}, Actual {i+1}: {actual[i][0]}")

print(f"Total Test Loss: {test_loss:.4f}")
results_df = pd.DataFrame(prediction_lstm)


# # save model
# save_path = os.path.join('C:/Users/Models', 
#                           'LSTM-tensteps-PR.pth')
# torch.save(model.state_dict(), save_path)


#load model:
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
save_path = os.path.join('C:/Users/Models', 
                          'LSTM-tensteps-PR.pth')
model.load_state_dict(torch.load(save_path))
model.eval()



# determine performance of model
mse_lstm = mean_squared_error(results_df['Actual Penetration Rate [m/min]'], 
                              results_df['Predicted Penetration Rate [m/min]'])
rmse_lstm = mse_lstm**.5
sym_mape_lstm = smape(results_df['Actual Penetration Rate [m/min]'], 
                      results_df['Predicted Penetration Rate [m/min]'])
accuracy_lstm = 100 - sym_mape_lstm

# results
result_data_list_lstm_pr = [
        "MSE {:.2f}".format(mse_lstm),
        "RMSE {:.2f}".format(rmse_lstm),
        "Acc {:.2f}".format(accuracy_lstm)]

#scatterplot
plt.figure(figsize=(8, 8))
plt.scatter(results_df['Actual Penetration Rate [m/min]'], 
            results_df['Predicted Penetration Rate [m/min]'])
plt.title('Scatter Plot LSTM for predicting Penetration Rate [m/min]')
plt.xlabel('Actual Penetration Rate [m/min]')
plt.ylabel('Predicted Penetration Rate [m/min]')
# plt.xlim(0.3, 4.2)  
# plt.ylim(0.3, 4.2)
plt.gca().set_aspect('equal', adjustable='box')




#confusion matrix
n_classes = 10  
#calculate classes for both values combined
combined_values = pd.concat([results_df['Actual Penetration Rate [m/min]'], 
                             results_df['Predicted Penetration Rate [m/min]']])
quantiles = np.quantile(combined_values, q=np.linspace(0, 1, n_classes + 1))

#classify results and acutal values
results_df['Actual Class'] = pd.cut(results_df['Actual Penetration Rate [m/min]'],
                                    bins=quantiles, labels=False, include_lowest=True)
results_df['Predicted Class'] = pd.cut(results_df['Predicted Penetration Rate [m/min]'], 
                                       bins=quantiles, labels=False, include_lowest=True)

conf_matrix = confusion_matrix(results_df['Actual Class'], results_df['Predicted Class'], 
                               labels=np.arange(n_classes))


#create confusion matrix with absolute values displayed
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, 
                              display_labels=[f'Class {i+1}' for i in range(n_classes)])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix LSTM for multiple time-steps Penetration Rate')
plt.show()


#create matrix with prozentual values 
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_percent, 
                              display_labels=[f'Class {i+1}' for i in range(n_classes)])
disp.plot(cmap=plt.cm.Blues, values_format=".2f")  
plt.title('Confusion Matrix (in %) LSTM for multiple time-steps Penetration Rate')
plt.show()




#plot along borehole

#create a list with original data + predictions
mwd_list_withLSTM = [df.copy() for df in test_dfs]
start_idx = 0 

for i, df in enumerate(mwd_list_withLSTM):
    num_predictions = 10  
    #get predicted values AND actual (to chceck if it works)
    predicted_values = [pred['Predicted Penetration Rate [m/min]'] for pred in prediction_lstm[start_idx:start_idx + num_predictions]]
    actual_values = [pred['Actual Penetration Rate [m/min]'] for pred in prediction_lstm[start_idx:start_idx + num_predictions]]
    
    #fill up first 20 values with None, rest with results
    df['Predicted Penetration Rate [m/min]'] = [None] * (len(df) - num_predictions) + predicted_values
    df['Actual Penetration Rate [m/min]'] = [None] * (len(df) - num_predictions) + actual_values
    
    start_idx += num_predictions

#check if values are matched correctly
for df in mwd_list_withLSTM:
    if 'Actual Penetration Rate [m/min]' in df.columns and 'Penetration Rate [m/min]' in df.columns:
        #round values for check
        df['Difference'] = (df['Actual Penetration Rate [m/min]'] - df['Penetration Rate [m/min]']).round(4)
        if not df['Difference'].isin([0, float('nan')]).all(): 
            print("Error")



# =============================================================================
# try LSTM on long created boreholes to fill up overlap
# =============================================================================

#load clean data
path_boreholes = Path(r'C:\Users\TestHoles')
holes_mwd_clean = []
for file in path_boreholes.glob('*_face779_780.csv'):
    df = pd.read_csv(file)
    sorted_df = df.sort_values(by="Total Depth [m]").reset_index(drop=True)
    holes_mwd_clean.append(sorted_df) 

#load raw data
path_boreholes_raw = Path(r'C:\Users\klein\Desktop\Geotechnical and Hydraulic Engineering\MP_MA\Masterarbeit\Coding\Abbildungen\TestHoles')
holes_mwd_raw = []
for file in path_boreholes_raw.glob('*_face779_780_raw.csv'):
    df = pd.read_csv(file)
    sorted_df = df.sort_values(by="Total Depth [m]").reset_index(drop=True)
    holes_mwd_raw.append(sorted_df) 
 
#delete overlap of raw data and fill up with nans
holes_mwd_without_overlap = []
for df in holes_mwd_raw:
    df_copy = df.copy()
    df_copy.loc[df_copy["Depth [m]"] > 1.5, df_copy.columns[3:9]] = np.nan
    holes_mwd_without_overlap.append(df_copy)
    


#plot all in one figure
colors = ['b', 'g', 'r']  
indices_to_plot = [0,1,2]
fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(15, 18))
fig.suptitle("Comparison of holes in face 779 + 780 (Raw Data)", fontsize=16)

for idx, color in zip(indices_to_plot, colors):
    df = holes_mwd_raw[idx]
    df.iloc[:, 3:9] = df.iloc[:, 3:9].apply(pd.to_numeric, errors='coerce')
    filtered_df = df[df["FileName"] == "780_drilling.csv"] #just that face
    filtered_df = filtered_df.dropna(subset=filtered_df.columns[3:9]) #drop nans
    hole_id = filtered_df["Hole ID"].iloc[0]
    file_name = filtered_df["FileName"].iloc[0]
    
    #show borehole before the one analysed
    filtered_df2 = df[df["FileName"] == "779_drilling.csv"]
    filtered_df2 = filtered_df2.dropna(subset=filtered_df.columns[3:9])
    
    combined_df = pd.concat([filtered_df, filtered_df2], ignore_index=True)
    legend_label = f'Hole ID: {hole_id}'

    columns = combined_df.columns[3:9]

    for i, column in enumerate(columns):
        ax = axes[i]
        x = combined_df['Total Depth [m]'].tolist()
        y = combined_df[column].tolist()
        ax.plot(x, y, marker='o', linestyle='', color=color, label=legend_label)
        ax.set_xlabel('Total Depth [m]')
        ax.set_ylabel(column)
        ax.set_title(f'{column}')
        ax.legend(loc='upper right')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
    
 
#function to fill up values in cleaned data
def predict_and_fill_nans(model, data, hidden_size, scaler_y,scaler_X, cols):
    output_data = []
    for df in data:
        df_copy = df.copy()
        df_copy = df_copy.sort_values(by="Total Depth [m]").reset_index(drop=True)
        # save original values to see which values have been predicted
        df_copy['Original PR'] = df_copy['Penetration Rate [m/min]']
        
        nan_segments = df_copy['Penetration Rate [m/min]'].isna()  # find nans
        
        i = 20  # model is trained with input of 20 rows -> start iterating at 20
        while i < len(df_copy):
            if nan_segments[i]:  # check if a nan segment starts there
                nan_start = i
                # detect length of nan segment
                while i < len(df_copy) and nan_segments[i]:
                    i += 1
                nan_end = i
                nan_length = nan_end - nan_start

                # prediction just possible if last 20 rows have no nans
                while nan_length > 0:
                    input_data = df_copy[cols].iloc[nan_start - 20:nan_start].values
                    input_data = scaler_X.transform(input_data)  #scale data
                    input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
                    
                    # initialize hidden states
                    h_0 = torch.zeros(1, input_data.size(0), hidden_size)
                    c_0 = torch.zeros(1, input_data.size(0), hidden_size)
                    
                    # predict next 10 values
                    with torch.no_grad():
                        predicted_output, h_0, c_0 = model(input_data, h_0, c_0)
                    
                    predicted_output = scaler_y.inverse_transform(predicted_output.cpu().numpy().reshape(-1, 1))
                    
                    # calculate how many values are needed to fill up nans + fill up
                    num_to_fill = min(nan_length, 10)
                    df_copy.iloc[nan_start:nan_start + num_to_fill, df_copy.columns.get_loc('Penetration Rate [m/min]')] = predicted_output[:num_to_fill, 0]
                    
                    # update iteration variables
                    nan_start += num_to_fill
                    nan_length -= num_to_fill

            else:
                i += 1  # if no nan -> jump to next row
        output_data.append(df_copy)
    return output_data


#function to predict values in a new column from a fixed staring point
def predict_from_depth(model, data, depth_threshold, hidden_size, scaler_y, scaler_X):
    result = []
    for df in data:
        df = df.sort_values(by="Total Depth [m]").reset_index(drop=True)

        #create column for predcitions
        if 'Penetration Rate predicted [m/min]' not in df.columns:
            df['Penetration Rate predicted [m/min]'] = np.nan

        #start at given depth = threshold
        start_index = df[df['Total Depth [m]'] >= depth_threshold].index[0]
        
        # Only predict if there are at least 20 rows before the depth threshold
        if start_index >= 20:
            i = start_index
            while i < len(df):
                #input = last 20 rows before current index
                input_data = df[cols].iloc[i - 20:i].values  
                input_data = scaler_X.transform(input_data)  
                input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
                # Initialize hidden states
                h_0 = torch.zeros(1, input_data.size(0), hidden_size)
                c_0 = torch.zeros(1, input_data.size(0), hidden_size)
                
                # Predict the next 10 values
                with torch.no_grad():
                    predicted_output, h_0, c_0 = model(input_data, h_0, c_0)
                
                # Inverse scale the predicted values and save in new column
                predicted_output = scaler_y.inverse_transform(predicted_output.cpu().numpy().reshape(-1, 1))
                num_to_fill = min(10, len(df) - i)
                df.iloc[i:i + num_to_fill, 
                        df.columns.get_loc('Penetration Rate predicted [m/min]')] = predicted_output[:num_to_fill, 0]
                
                i += num_to_fill
        result.append(df)

    return result


#fill up nans
holes_mwd_raw_LSTM = predict_and_fill_nans(model, holes_mwd_without_overlap, hidden_size, scaler_y,scaler_X, cols)


#analyze predictions at points where values are given
pred_LSTM_290 = predict_from_depth(model=model, data=holes_mwd_without_overlap, 
                                   depth_threshold=2.90, hidden_size=hidden_size, 
                                   scaler_y=scaler_y)
pred_LSTM_280 = predict_from_depth(model=model, data=holes_mwd_without_overlap, 
                                   depth_threshold=2.80, hidden_size=hidden_size, 
                                   scaler_y=scaler_y)

#plot fill up nans

def plot_holes_with_depth(holes_mwd_raw_LSTM, depth_min, depth_max, 
                          colors=['b', 'g', 'r'], indices_to_plot=[0, 1, 2]):
    # Check that files start from the same point
    first_filenames = [holes_mwd_raw_LSTM[idx]["FileName"].iloc[0] for idx in indices_to_plot]
    if len(set(first_filenames)) > 1:
        print("Error: Files do not start at same face")
        return

    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(15, 18))
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
            x1, y1 = [], []  # Original values
            x2, y2 = [], []  # Predicted values

            for _, row in combined_df.iterrows():
                if pd.isna(row['Original PR']):  # NaN -> predicted
                    x2.append(row['Total Depth [m]'])
                    y2.append(row[column])
                else:
                    x1.append(row['Total Depth [m]'])
                    y1.append(row[column])

            # Plot original values as points
            orig_handle, = ax.plot(x1, y1, marker='o', linestyle='', 
                                   color=color, label=f'{legend_label} (original)' if i == 0 else "", markersize=4)
            
            # Plot predicted values as stars
            pred_handle, = ax.plot(x2, y2, marker='*', linestyle='', 
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

#use function
plot_holes_with_depth(holes_mwd_raw_LSTM, depth_min=2, depth_max=4.5)

            


# =============================================================================
# Test LSTM in three sections
# =============================================================================

# test model

#split predictions
prediction_lstm1to3 = []
prediction_lstm4to6 = []
prediction_lstm7to10 = []

model.eval()
test_loss = 0
with torch.no_grad():
    h_0 = torch.zeros(1, X_test_list[0].size(0), hidden_size)  # Initialize test hidden state
    c_0 = torch.zeros(1, X_test_list[0].size(0), hidden_size)  # Initialize test cell state
    
    for X_test, y_test in zip(X_test_list, y_test_list):
        # Forward pass: Predict with previous hidden states
        test_outputs, h_0, c_0 = model(X_test, h_0, c_0)

        # Inverse scaling the predictions and actual values
        predicted = scaler_y.inverse_transform(test_outputs.cpu().numpy().reshape(-1, 1))
        actual = scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
        
        # Calculate loss between predicted and actual values (in unscaled space)
        loss = criterion(torch.tensor(predicted, dtype=torch.float32), torch.tensor(actual, dtype=torch.float32))
        test_loss += loss.item()
        
        #split predictions in three groups to later analyze them individually
        predicted_groups = [predicted[:3], predicted[3:6], predicted[6:]]
        actual_groups = [actual[:3], actual[3:6], actual[6:]]
        
        for i in range(3):
            for j in range(len(predicted_groups[i])):
                group = i + 1
                pred_value = predicted_groups[i][j][0]
                actual_value = actual_groups[i][j][0]
                if i == 0:
                    prediction_lstm1to3.append({'Predicted Penetration Rate [m/min]': pred_value, 
                                                'Actual Penetration Rate [m/min]': actual_value})
                elif i == 1:
                    prediction_lstm4to6.append({'Predicted Penetration Rate [m/min]': pred_value, 
                                                'Actual Penetration Rate [m/min]': actual_value})
                elif i == 2:
                    prediction_lstm7to10.append({'Predicted Penetration Rate [m/min]': pred_value, 
                                                 'Actual Penetration Rate [m/min]': actual_value})

                # Print each prediction and actual value
                print(f"Group {group} Prediction {j+1}: Predicted = {pred_value}, Actual = {actual_value}")



print(f"Total Test Loss: {test_loss:.4f}")
results_df = pd.DataFrame(prediction_lstm)

#in case it was split into three groups
results_df1to3 = pd.DataFrame(prediction_lstm1to3)
results_df4to6 = pd.DataFrame(prediction_lstm4to6)
results_df7to10 = pd.DataFrame(prediction_lstm7to10)


#analyze results
result_data_list_lstm_pr = []

dfs = [(results_df1to3, "Step 1-3"),
    (results_df4to6, "Step 4-6"),
    (results_df7to10, "Step 7-10")]

for df, group_name in dfs:
    mse = mean_squared_error(df['Actual Penetration Rate [m/min]'], 
                             df['Predicted Penetration Rate [m/min]'])
    rmse = mse ** 0.5
    sym_mape = smape(df['Actual Penetration Rate [m/min]'], 
                     df['Predicted Penetration Rate [m/min]'])
    accuracy = 100 - sym_mape
    
    # Append results for each group to the list
    result_data_list_lstm_pr.append({"Group": group_name, "MSE": mse,"RMSE": rmse,
                                     "Accuracy": accuracy})

    print(f"{group_name} - MSE: {mse:.2f}, RMSE: {rmse:.2f}, Accuracy: {accuracy:.2f}%")
    
    # Scatterplot for each group
    plt.figure(figsize=(8, 8))
    plt.scatter(df['Actual Penetration Rate [m/min]'], df['Predicted Penetration Rate [m/min]'])
    plt.title(f'Scatter Plot LSTM for predicting Penetration Rate [m/min] - {group_name}')
    plt.xlabel('Actual Penetration Rate [m/min]')
    plt.ylabel('Predicted Penetration Rate [m/min]')
    plt.xlim(0.3, 4.2)  
    plt.ylim(0.3, 4.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()






# =============================================================================
# Random Forest
# =============================================================================

models = []
for step in range(10):
    # Initialize a new Random Forest model for each step
    model_rf = RandomForestRegressor(
        n_estimators=800,
        random_state=42 + step,  # seed for each model for variability
        max_depth=14,
        min_samples_leaf=3,
        max_features=6  
    )
    models.append(model_rf)


#train models
for step, model in enumerate(models):
    y_train_rf = y_train_rf_steps[step]  # Get training target for this step
    model.fit(X_train_rf, y_train_rf)

base_model_path = Path(r'C:/Users/Models/')
# Save models
for step, model in enumerate(models):
    model_path = base_model_path / f'RF_step_{step+1}.pth'
    joblib.dump(model, model_path)

# #load modls
# models = [joblib.load(base_model_path / f'RF_step_{step+1}.pth') for step in range(10)]


#collect all predictions to calcualte RMSE/MSE
all_predictions = []
all_actuals = []

for step, model in enumerate(models):
    rf_predictions = model.predict(X_test_rf)
    
    #transform back
    rf_predictions_unscaled = scaler_y.inverse_transform(rf_predictions.reshape(-1, 1)).flatten()
    y_test_rf_unscaled = scaler_y.inverse_transform(y_test_rf_steps[step].reshape(-1, 1)).flatten()
    
    #collect values
    all_predictions.extend(rf_predictions_unscaled)
    all_actuals.extend(y_test_rf_unscaled)


#metrics for all models combined
overall_mse = mean_squared_error(all_actuals, all_predictions)
overall_rmse = overall_mse ** 0.5
overall_smape = smape(np.array(all_actuals), np.array(all_predictions))
overall_accuracy = 100 - overall_smape

result_data_list_rf_pr = [
    "Overall MSE {:.2f}".format(overall_mse),
    "Overall RMSE {:.2f}".format(overall_rmse),
    "Overall Accuracy {:.2f}".format(overall_accuracy)]

#scatterplot
plt.figure(figsize=(8, 8))
plt.scatter(all_actuals, all_predictions, color='#1f77b4')  
plt.title('Scatter Plot Basemodel for predicting Penetration Rate [m/min]')
plt.xlabel('Actual Penetration Rate [m/min]')
plt.ylabel('Predicted Penetration Rate [m/min]')
plt.xlim(0.3, 4.2)
plt.ylim(0.3, 4.2)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


def predict_and_fill_nans_rf(models, df, scaler_X, scaler_y, cols):
    # Create a copy of the DataFrame to keep the original data intact
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by="Total Depth [m]").reset_index(drop=True)
    df_copy['Original PR'] = df_copy['Penetration Rate [m/min]']  # Save original values for reference
    
    # Start after 20 rows to ensure sufficient input data for prediction
    i = 20
    while i < len(df_copy):
        # Check if the current row has a NaN value in the target column
        if pd.isna(df_copy.loc[i, 'Penetration Rate [m/min]']):
            # Identify the start and end of the NaN segment
            nan_start = i
            while i < len(df_copy) and pd.isna(df_copy.loc[i, 'Penetration Rate [m/min]']):
                i += 1
            nan_end = i
            nan_length = nan_end - nan_start

            # Ensure that the last 20 rows before the NaN segment contain complete data
            if not df_copy[cols].iloc[nan_start - 20:nan_start].isna().any().any():
                # Extract the last 20 rows as input data for prediction
                input_data = df_copy[cols].iloc[nan_start - 20:nan_start].values
                input_data_scaled = scaler_X.transform(input_data)  # Scale the input data
                
                # Use each model to predict each of the next 10 steps
                predicted_outputs = []
                for step, model in enumerate(models):
                    # Predict using the model for this specific step
                    pred_scaled = model.predict(input_data_scaled.flatten().reshape(1, -1)).flatten()
                    # Inverse-transform the prediction back to original scale
                    pred_unscaled = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                    predicted_outputs.append(pred_unscaled[0])  # Save the unscaled prediction
                
                # Determine how many values to fill in the NaN segment
                num_to_fill = min(nan_length, 10)
                df_copy.loc[nan_start:nan_start + num_to_fill - 1, 'Penetration Rate [m/min]'] = predicted_outputs[:num_to_fill]
                
                # Move to the next row after the filled segment
                i = nan_start + num_to_fill
            else:
                # If the last 20 rows contain NaNs, skip this NaN segment
                print(f"Skipping segment from index {nan_start} to {nan_end} due to insufficient data.")
        else:
            # If no NaN, move to the next row
            i += 1
    
    return df_copy

# Example of using the function to fill NaNs
holes_mwd_raw_RF = []
for df in holes_mwd_without_overlap:
    prediction = predict_and_fill_nans_rf(models, df, scaler_X, scaler_y, cols)
    holes_mwd_raw_RF.append(prediction)

plot_holes_with_depth(holes_mwd_raw_RF, depth_min=2, depth_max=4.5)


