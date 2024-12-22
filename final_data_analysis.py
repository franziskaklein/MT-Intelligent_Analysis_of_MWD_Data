# -*- coding: utf-8 -*-
"""

@author: klein

content: new data preparation (with created long boreholes, but are not used as such)

additional: prepare longer series for autoregressive training of the data

"""

import pandas as pd
from pathlib import Path 
import numpy as np
import os


# =============================================================================
# load prepared MWD data
# =============================================================================

path_result = Path(r'C:\Users\Data\00_created_Data')

mwd_conture_holes_raw = []
for file in sorted(path_result.glob('*.csv')):
     df = pd.read_csv(file)
     mwd_conture_holes_raw.append(df)

drop_cols = ['Waterpump Pressure [Bar]', 'Stabilator Pressure [Bar]', 
             'Rotation Speed Setting [RPM]', 'Feed Speed Setting [m/min]', 
             'Rock Detect [0/1]', 'State Of Flushing Flow [0/1]', 
             'Anti-Jamming State [0/1]', 'Drilling Control Setting [%]', 
             'Air Flow [l/min]', 'Feed Pressure Setting [Bar]', 
             'Rotation Pressure Setting [Bar]', 'Air Mist Setting [%]']

for i in range(len(mwd_conture_holes_raw)):
    mwd_conture_holes_raw[i] = mwd_conture_holes_raw[i].drop(columns=drop_cols, errors='ignore')

# =============================================================================
# clean data 
# =============================================================================

#copy raw data
mwd_used_raw = [df.copy() for df in mwd_conture_holes_raw]

#delte first 0.6m of data and also a depth > 1.5 m
for i in range(len(mwd_used_raw)):
    df = mwd_used_raw[i]
    df_filtered = df[(df['Depth [m]'] > 0.6) & (df['Depth [m]'] <= 1.5)]
    mwd_used_raw[i] = df_filtered
    
#find outliers (and interpolate them)
mwd_used_total = pd.DataFrame()
for df in mwd_used_raw:
    mwd_used_total = pd.concat([mwd_used_total, df], axis=0, ignore_index=True)    

for column_index in range(3, 9):
    Q1 = mwd_used_total.iloc[:, column_index].quantile(0.25)
    Q3 = mwd_used_total.iloc[:, column_index].quantile(0.75)
    IQR = Q3 - Q1
    for df in mwd_used_raw:
        df[f'is_outlier_{column_index}'] = ((df.iloc[:, column_index] < (Q1 - 1.5 * IQR)) | (df.iloc[:, column_index] > (Q3 + 1.5 * IQR)))
    
mwd_used_clean = []

#set outliers to nans    
for df in mwd_used_raw:
    df_copy = df.copy(deep=True)
    df_copy['is_outlier_any'] = df_copy[[f'is_outlier_{col}' for col in range(3, 9)]].any(axis=1)
    #set values to nan for col 3 to 8 if there is an outlier detected
    for column_index in range(3, 9):
        df_copy.iloc[:, column_index] = df_copy.apply(lambda row: np.nan if row['is_outlier_any'] else row[column_index], axis=1)
    df_copy.drop(columns=[f'is_outlier_{col}' for col in range(3, 9)] + ['is_outlier_any'], inplace=True)
    mwd_used_clean.append(df_copy)



# =============================================================================
# split data in timelines using "depth" as time in set of 30 rows
# =============================================================================

mwd_used_thirtys = []

# each df in mwd_used_clean is one long borehole
for df in mwd_used_clean:
    df_copy = df.copy(deep=True)
    df_list = []  # collects all lists that belong to this big borehole
    # split in each actually drilled borehole by file
    for file_name in df_copy['FileName'].unique():
        borehole_df = df_copy[df_copy['FileName'] == file_name]
        file_list = []  # collects all lists that belong to this drilling = filename
        for i in range(0, len(borehole_df), 30):  # now it processes in steps of 30
            thirty_rows = borehole_df.iloc[i:i + 30]  # now taking 30 rows
            file_list.append(thirty_rows)
        df_list.append(file_list)
    mwd_used_thirtys.append(df_list)

#collect all df in one liste  
mwd_all_thirtys = []    
for df_list in mwd_used_thirtys:
    for file_list in df_list:
        for thirty_rows in file_list:
            mwd_all_thirtys.append(thirty_rows)
            
# =============================================================================
# save created data
# =============================================================================

from pathlib import Path

path = Path(r'C:\Users\Data\00_created_Data\04_thirty_row_dataframes')

for i, df in enumerate(mwd_all_thirtys):
    filename = f'borehole_{i:03d}_30rows.csv'  
    save_path = os.path.join(path, filename)    #creates path
    df.to_csv(save_path, index=False)


# =============================================================================
# split dat in sets of 40 for new training
# =============================================================================

mwd_used_fourtys = []

# each df in mwd_used_clean is one long borehole
for df in mwd_used_clean:
    df_copy = df.copy(deep=True)
    df_list = []  
    # split in each actually drilled borehole by file from big file
    for file_name in df_copy['FileName'].unique():
        borehole_df = df_copy[df_copy['FileName'] == file_name]
        file_list = []  # collects all lists that belong to this drilling = filename
        for i in range(0, len(borehole_df), 40):  # now it processes in steps of 40
            fourty_rows = borehole_df.iloc[i:i + 40]  # now taking 40 rows
            if not thirty_rows.iloc[:, 3:9].isna().any().any():
                file_list.append(fourty_rows)
        df_list.append(file_list)
    mwd_used_fourtys.append(df_list)

#collect all df in one liste  
mwd_all_fourtys = []    
for df_list in mwd_used_fourtys:
    for file_list in df_list:
        for fourty_rows in file_list:
            if len(fourty_rows) == 40:
                mwd_all_fourtys.append(fourty_rows)

#save data
from pathlib import Path

path = Path(r'C:\Users\Data\00_created_Data\05_fourty_row_dataframes')

for i, df in enumerate(mwd_all_fourtys):
    filename = f'borehole_{i:03d}_40rows.csv'  
    save_path = os.path.join(path, filename)    #creates path
    df.to_csv(save_path, index=False)


