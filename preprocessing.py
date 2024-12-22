# -*- coding: utf-8 -*-
"""

@author: klein

prepare data for further processing, adapt datatypes and check error boreholes
"""

import pandas as pd
from pathlib import Path 

# Enable Copy-on-Write to prepare for pandas 3.0 --> to get rid of error
pd.options.mode.copy_on_write = True

path_drilling = Path(r'C:\Users\Data\01_drilling')
path_mwd = Path(r'C:\Users\Data\02_mwd')

# =============================================================================
# Drilling files - adapt data
# =============================================================================


dataframes_drilling = [] #collects new dataframes
error_files = [] #collects filename and type of error in case of error
missed_dataframes = [] #collects missed dataframes that are not yet in first list
        
for file in path_drilling.glob('*.csv'):
    df = pd.read_csv(file) #create df from file
    df['FileName'] = file.name  # add name of file for later
    try:         
        #convert planned coordinates in floating numbers
        for col in df.columns[3:15]:  
            df[col] = df[col].astype(float)  
        dataframes_drilling.append(df) #append correct df to dataframes
    except ValueError as error:
        #save problematic files here:
        error_files.append((file, str(error)))
        missed_dataframes.append(df)          
    
#those errors here due to "-" --> delete all boreholes which are called Ex + check again
#save deleted boreholes in new dataframe
deleted_boreholes = pd.DataFrame()

for df in missed_dataframes:
    condition = df['Hole ID'].str.contains('^E')
    #add later on deleted boreholes to datafram
    deleted_boreholes = pd.concat([deleted_boreholes, df[condition].copy()], ignore_index=True)
    #keep just wanted boreholes 
    df.drop(df[condition].index, inplace=True)
    
    try:
        for col in df.columns[3:15]:  
            df[col] = df[col].astype(float) 
        dataframes_drilling.append(df)
    except ValueError as error:
        print('Still an error: {error}!')

#check if there are nan values left         

nans_in_drilling = pd.DataFrame()

for df in dataframes_drilling:
    mask = df.isnull().any(axis=1) | (df == '-').any(axis=1)
    nans_in_drilling = pd.concat([nans_in_drilling, df[mask]], ignore_index=True)

        

    
error_files = []
dataframes_drilling = []

for file in path_drilling.glob('*.csv'):
    try: 
        df = pd.read_csv(file) #create df from file
        #convert planned coordinates in floating numbers
        df.iloc[:, 15:27] = df.iloc[:, 15:27].astype(float) 
        dataframes_drilling.append(df) #append correct df to dataframes
    except ValueError as error:
        #save problematic files here:
            error_files.append((file, str(error)))
           
if error_files:
    print("problems within the following file:")
    for file, error in error_files:
        print(f"Datei: {file}, Fehler: {error}")
else:
    print("All files saved.")