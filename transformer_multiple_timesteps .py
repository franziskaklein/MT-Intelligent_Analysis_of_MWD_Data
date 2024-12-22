# -*- coding: utf-8 -*-
"""

@author: klein

prediction goal: next 10 time steps for just one feature
cleaned code with only transformer related sections

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
import joblib


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
# prepare Transformer Data
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
scaler_X_path = r'C:\Users\Models\scaler_multi_X_X.pkl'
scaler_y_path = r'C:\Users\Models\scaler_multi_EX_y.pkl'

# joblib.dump(scaler_X, scaler_X_path)
# joblib.dump(scaler_y, scaler_y_path)

#load scaler
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

# Initialize lists for storing data
X_train_list, y_train_list, depth_train_list = [], [], []
X_test_list, y_test_list, depth_test_list = [], [], []

# convert training data
for df in train_dfs:
    X = df.iloc[:-10][cols].values
    y = df.iloc[-10:]['Penetration Rate [m/min]'].values  # shape (10,)
    depths = df.iloc[:-10]['Depth [m]'].values

    # scale
    X = scaler_X.transform(X)
    y = scaler_y.transform(y.reshape(-1, 1)).flatten()  # reshape y to (10, 1) and flatten after scaling

    # convert to tensors
    X_train_list.append(torch.tensor(X, dtype=torch.float32).unsqueeze(0))  # unsqueeze to add batch dimension
    y_train_list.append(torch.tensor(y, dtype=torch.float32))
    depth_train_list.append(torch.tensor(depths, dtype=torch.float32).unsqueeze(0))

# prepare test data
for df in test_dfs:
    X = df.iloc[:-10][cols].values
    y = df.iloc[-10:]['Penetration Rate [m/min]'].values  
    depths = df.iloc[:-10]['Depth [m]'].values

    # scale
    X = scaler_X.transform(X)
    y = scaler_y.transform(y.reshape(-1, 1)).flatten()  

    # convert to tensors
    X_test_list.append(torch.tensor(X, dtype=torch.float32).unsqueeze(0))
    y_test_list.append(torch.tensor(y, dtype=torch.float32))
    depth_test_list.append(torch.tensor(depths, dtype=torch.float32).unsqueeze(0))


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
# transfomer
# =============================================================================

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, hidden_dim, output_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_dim)  # Input embedding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, hidden_dim))  # Positional encoding 
        self.transformer = nn.Transformer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_size)  # Final layer to predict 10 timesteps

    def forward(self, x, pos_enc):
        # Embed input and add positional encoding
        x = self.embedding(x) + pos_enc
        x = self.transformer(x, x)  # Self-attention over the input sequence
        x = self.fc(x[:, -10:, :])  # Take the last 10 timesteps for prediction
        x = x[:, :, 0]  # Reshape to (batch_size, 0)
        return x

# Hyperparameters
input_size = len(cols)  # Number of input features (e.g. features in the time series)
hidden_dim = 64         # Model dimensionality 
num_heads = 2           # Number of attention heads
num_layers = 2          # Number of transformer layers
output_size = 10        # Predicting 10 timesteps

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model2 = TransformerModel(input_size, num_heads, num_layers, hidden_dim, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model2.parameters(), lr=0.001)


#time the training process
import time
total_start_time = time.time()

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f'Starting Epoch {epoch+1}')
    model2.train()
    epoch_loss = 0  # Initialize loss per epoch

    for X_train, y_train, depth_train in zip(X_train_list, y_train_list, depth_train_list):
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        depth_train = depth_train.to(device)

        optimizer.zero_grad()

        # Create positional encoding using depth 
        depth_scaled = (depth_train - depth_train.min()) / (depth_train.max() - depth_train.min())  # Normalize depths
        pos_enc = depth_scaled.unsqueeze(-1).repeat(1, 1, hidden_dim)  # Repeat to match hidden_dim

        # Forward pass: predict
        outputs = model2(X_train, pos_enc)
        loss = criterion(outputs, y_train.view(X_train.size(0), 10))  # Reshape y_train to (batch_size, 10)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

total_end_time = time.time()
print(f'Total Training Time: {total_end_time - total_start_time:.2f} seconds')



# test model
prediction_transformer = []

model2.eval()
test_loss = 0
with torch.no_grad():
    for X_test, y_test, depth_test in zip(X_test_list, y_test_list, depth_test_list):
        # positional encoding
        depth_scaled = (depth_test - depth_test.min()) / (depth_test.max() - depth_test.min())
        pos_enc = depth_scaled.unsqueeze(-1).repeat(1, 1, hidden_dim)

        test_outputs = model2(X_test, pos_enc)  #predictions
        # Transform values back (predicted and actual test data)
        predicted = scaler_y.inverse_transform(test_outputs.numpy().reshape(-1, 1))
        actual = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1))

        loss = criterion(
            torch.tensor(predicted, dtype=torch.float32), 
            torch.tensor(actual, dtype=torch.float32)
        )
        test_loss += loss.item()
        
        for i in range(3):
            prediction_transformer.append({
                'Predicted Penetration Rate [m/min]': predicted[i][0],  
                'Actual Penetration Rate [m/min]': actual[i][0]})

            print(f"Prediction {i+1}: {predicted[i][0]}, Actual {i+1}: {actual[i][0]}")


print(f"Total Test Loss: {test_loss:.4f}")
results_df_transformer = pd.DataFrame(prediction_transformer)

#save model
#save_path = os.path.join('C:/Users/Models', 
#                         'Transformer-tenstep-PR.pth')
#torch.save(model2.state_dict(), save_path)

#load model:
save_path = r'C:\Users\Models\Transformer-tenstep-PR.pth'
model2 = TransformerModel(input_size, num_heads, num_layers, hidden_dim, output_size)

model2.load_state_dict(torch.load(save_path))
model2.eval()



# determine performance of model
mse_transformer = mean_squared_error(results_df_transformer['Actual Penetration Rate [m/min]'], 
                              results_df_transformer['Predicted Penetration Rate [m/min]'])
rmse_transformer = mse_transformer**.5
sym_mape_transformer = smape(results_df_transformer['Actual Penetration Rate [m/min]'], 
                      results_df_transformer['Predicted Penetration Rate [m/min]'])
accuracy_transformer = 100 - sym_mape_transformer

# results
result_data_list_transformer_pr = [
        "MSE {:.2f}".format(mse_transformer),
        "RMSE {:.2f}".format(rmse_transformer),
        "Acc {:.2f}".format(accuracy_transformer)]

plt.figure(figsize=(8, 8))
plt.scatter(results_df_transformer['Actual Penetration Rate [m/min]'], results_df_transformer['Predicted Penetration Rate [m/min]'])
plt.title('Scatter Plot Transformer for predicting Penetration Rate [m/min]')
plt.xlabel('Actual Penetration Rate [m/min]')
plt.ylabel('Predicted Penetration Rate [m/min]')
plt.xlim(0.3, 4.2)  
plt.ylim(0.3, 4.2)
plt.gca().set_aspect('equal', adjustable='box')


#confusion matrix

n_classes = 10  
#calculate classes for both values combined
combined_values = pd.concat([results_df_transformer['Actual Penetration Rate [m/min]'], 
                             results_df_transformer['Predicted Penetration Rate [m/min]']])
quantiles = np.quantile(combined_values, q=np.linspace(0, 1, n_classes + 1))
#classify results and acutal values
results_df_transformer['Actual Class'] = pd.cut(results_df_transformer['Actual Penetration Rate [m/min]'],
                                    bins=quantiles, labels=False, include_lowest=True)
results_df_transformer['Predicted Class'] = pd.cut(results_df_transformer['Predicted Penetration Rate [m/min]'], 
                                       bins=quantiles, labels=False, include_lowest=True)

#create confusion matrix
conf_matrix_transformer = confusion_matrix(results_df_transformer['Actual Class'], 
                               results_df_transformer['Predicted Class'], 
                               labels=np.arange(n_classes))

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_transformer, 
                              display_labels=[f'Class {i+1}' for i in range(n_classes)])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix Transformer for multiple time-steps Penetration Rate')
plt.show()

# create a prozentual confusion matrix
conf_matrix_transformer_percent = conf_matrix_transformer.astype('float') / conf_matrix_transformer.sum(axis=1)[:, np.newaxis] * 100
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_transformer_percent, 
                              display_labels=[f'Class {i+1}' for i in range(n_classes)])
disp.plot(cmap=plt.cm.Blues, values_format=".2f")  
plt.title('Confusion Matrix (in %) Transformer for multiple time-steps Penetration Rate')
plt.show()

# =============================================================================
# test transformer in three sections
# =============================================================================

prediction_transformer1to3 = []
prediction_transformer4to6 = []
prediction_transformer7to10 = []

model2.eval()
test_loss = 0
with torch.no_grad():
    for X_test, y_test, depth_test in zip(X_test_list, y_test_list, depth_test_list):
        # Positional Encoding
        depth_scaled = (depth_test - depth_test.min()) / (depth_test.max() - depth_test.min())
        pos_enc = depth_scaled.unsqueeze(-1).repeat(1, 1, hidden_dim)

        # Forward pass
        test_outputs = model2(X_test, pos_enc)  
        predicted = scaler_y.inverse_transform(test_outputs.numpy().reshape(-1, 1))
        actual = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1))

        loss = criterion(torch.tensor(predicted, dtype=torch.float32), 
            torch.tensor(actual, dtype=torch.float32))
        test_loss += loss.item()
        
        predicted_groups = [predicted[:3], predicted[3:6], predicted[6:]]
        actual_groups = [actual[:3], actual[3:6], actual[6:]]
        
        for i in range(3):
            for j in range(len(predicted_groups[i])):
                group = i + 1
                pred_value = predicted_groups[i][j][0]
                actual_value = actual_groups[i][j][0]

                if i == 0:
                    prediction_transformer1to3.append({'Predicted Penetration Rate [m/min]': pred_value, 
                                                       'Actual Penetration Rate [m/min]': actual_value})
                elif i == 1:
                    prediction_transformer4to6.append({'Predicted Penetration Rate [m/min]': pred_value, 
                                                       'Actual Penetration Rate [m/min]': actual_value})
                elif i == 2:
                    prediction_transformer7to10.append({'Predicted Penetration Rate [m/min]': pred_value, 
                                                        'Actual Penetration Rate [m/min]': actual_value})

                print(f"Group {group} Prediction {j+1}: Predicted = {pred_value}, Actual = {actual_value}")

print(f"Total Test Loss: {test_loss:.4f}")

results_df1to3_transformer = pd.DataFrame(prediction_transformer1to3)
results_df4to6_transformer = pd.DataFrame(prediction_transformer4to6)
results_df7to10_transformer = pd.DataFrame(prediction_transformer7to10)



result_data_list_lstm_pr = []

dfs_transformer = [(results_df1to3_transformer, "Step 1-3"),
    (results_df4to6_transformer, "Step 4-6"),
    (results_df7to10_transformer, "Step 7-10")]

for df, group_name in dfs_transformer:
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
    plt.title(f'Scatter Plot Transformer for predicting Penetration Rate [m/min] - {group_name}')
    plt.xlabel('Actual Penetration Rate [m/min]')
    plt.ylabel('Predicted Penetration Rate [m/min]')
    plt.xlim(0.3, 4.2)  
    plt.ylim(0.3, 4.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# =============================================================================
# apply transformer to real data
# =============================================================================

#load raw data
path_boreholes_raw = Path(r'C:\Users\TestHoles')
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
    
def predict_and_fill_nans_transformer(model, data, hidden_dim, scaler_y, scaler_X, cols):
    output_data = []
    for df in data:
        df_copy = df.copy()
        df_copy = df_copy.sort_values(by="Total Depth [m]").reset_index(drop=True)
        # Save original values
        df_copy['Original PR'] = df_copy['Penetration Rate [m/min]']
        
        nan_segments = df_copy['Penetration Rate [m/min]'].isna()  # Search for NaNs
        
        i = 20  # Start here as model needs 20 complete rows
        while i < len(df_copy):
            if nan_segments[i]:
                nan_start = i
                # Calculate length of NaN segment
                while i < len(df_copy) and nan_segments[i]:
                    i += 1
                nan_end = i
                nan_length = nan_end - nan_start

                while nan_length > 0:
                    input_data = df_copy[cols].iloc[nan_start - 20:nan_start].values
                    # Check input shape to avoid errors with insufficient rows
                    if input_data.shape[0] < 20:
                        break  # Not enough rows to make predictions

                    input_data = scaler_X.transform(input_data)  # Scale data
                    input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 20, len(cols))

                    # Generate positional encoding using the correct approach
                    depth_values = torch.tensor(df_copy['Total Depth [m]'].iloc[nan_start - 20:nan_start].values, dtype=torch.float32)
                    depth_scaled = (depth_values - depth_values.min()) / (depth_values.max() - depth_values.min())
                    pos_enc = depth_scaled.unsqueeze(-1).repeat(1, hidden_dim).unsqueeze(0)  # Shape: (1, 20, hidden_dim)

                    # Prediction
                    with torch.no_grad():
                        predicted_output = model(input_data, pos_enc)

                    # Adjust predicted_output to correct shape
                    predicted_output = predicted_output.squeeze(0).cpu().numpy()  # Shape: (10,)
                    predicted_output = predicted_output.reshape(-1, 1)  # Shape: (10, 1)
                    predicted_output = scaler_y.inverse_transform(predicted_output)
                    
                    # Fill up NaNs
                    num_to_fill = min(nan_length, 10)  # Fill up to 10 values at a time
                    df_copy.iloc[nan_start:nan_start + num_to_fill, df_copy.columns.get_loc('Penetration Rate [m/min]')] = predicted_output[:num_to_fill, 0]

                    nan_start += num_to_fill
                    nan_length -= num_to_fill
            else:
                i += 1
        output_data.append(df_copy)
    return output_data

def plot_holes_with_depth(data, depth_min, depth_max, 
                          colors=['b', 'g', 'r'], indices_to_plot=[0, 1, 2]):
    # Check that files start from the same point
    first_filenames = [data[idx]["FileName"].iloc[0] for idx in indices_to_plot]
    if len(set(first_filenames)) > 1:
        print("Error: Files do not start at same face")
        return

    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(15, 18))
    fig.suptitle(f"Comparison of holes (with Transformer prediction) from {depth_min}-{depth_max}", fontsize=16)
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


holes_mwd_raw_transformer = predict_and_fill_nans_transformer(model2, holes_mwd_without_overlap, 
                                                              hidden_dim, scaler_y, scaler_X, cols)
plot_holes_with_depth(holes_mwd_raw_transformer, depth_min=2, depth_max=4.5)


