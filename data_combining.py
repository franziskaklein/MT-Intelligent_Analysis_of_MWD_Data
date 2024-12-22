# -*- coding: utf-8 -*-
"""

@author: klein

task: match boreholes for long boreholes through tunnel + analyze data

"""

import pandas as pd
from pathlib import Path 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import math


#path in which all the files are stored
path_drilling = Path(r'C:\Users\Data\01_drilling')
path_mwd = Path(r'C:\Users\Data\02_mwd')

#change headings to english ones
excel_file_path = r'C:\Users\Data\Read me_MWD&Drilling_list_0707.xlsx'
df_drilling = pd.read_excel(excel_file_path, sheet_name='Drilling data list')
engl_columns = df_drilling.columns.tolist()


#function to extract the facenumber from the filename
def extract_number(file_name):
    match = re.search(r'(\d+)_drilling\.csv', file_name)
    if match:
        return int(match.group(1))
    return float('inf')  # If no number found, place at the end



dataframes_drilling = []
deleted_boreholes = pd.DataFrame()

# Sort files based on the extracted number
sorted_files = sorted(path_drilling.glob('*.csv'), key=lambda x: extract_number(x.name))

#create dataframes 
for file in sorted_files: 
    df = pd.read_csv(file)
    df.columns = engl_columns
    df['FileName'] = file.name  # add name of file for later
    try:         
        #convert planned coordinates and drilled in floating numbers
        for col in df.columns[list(range(3, 15)) + list(range(15, 27))]:
            df[col] = df[col].astype(float)
        dataframes_drilling.append(df) #append correct df to dataframes
    except ValueError:
        condition = df['Hole ID'].str.contains('^E')
        #add later on deleted boreholes to dataframe
        deleted_boreholes = pd.concat([deleted_boreholes, df[condition].copy()], ignore_index=True)
        #keep just wanted boreholes 
        df.drop(df[condition].index, inplace=True)
        
        try:
            for col in df.columns[list(range(3, 15)) + list(range(15, 27))]:
                df[col] = df[col].astype(float)
            dataframes_drilling.append(df)
        except ValueError: #as error
            #find rows that are nans/can not be transformed to int
            problematic_rows = df.loc[df[col].apply(lambda x: not isinstance(x, (int, float)) and not pd.isna(x))]
            deleted_boreholes = pd.concat([deleted_boreholes, problematic_rows], ignore_index=True)
            df.drop(problematic_rows.index, inplace=True)
            
            dataframes_drilling.append(df)

# plot Plan Coordinates to see what a face looks like
for df in dataframes_drilling:
    fig, ax = plt.subplots()
    for index, row in df.iterrows():
        x, y, z = df.iloc[index,3:6] #select plan coordinates start
        x = float(x)
        z = float(z)
        label = row['Hole Types']
        ax.scatter(x, z, label=label)
        ax.text(x, z, label, fontsize = 5)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_title('Plot of Plan Coordinates')
    plt.show()
    break  # Stop after plotting the first DataFrame


#keep only contour or bottom holes (as these define tunnel line)
for i in range(len(dataframes_drilling)):
    df = dataframes_drilling[i]
    condition = (df['Hole Types'] == 'ContourHole') | (df['Hole Types'] == 'BottomHole')
    dataframes_drilling[i] = df[condition].copy()
    
    
 # =============================================================================
 # match all holes in every face
 # =============================================================================

# copy df to delete already used boreholes but keep all in origin df
dataframes_drilling_copy = [df.copy() for df in dataframes_drilling]

list_holes = []
list_short_holes = []

#start point for connecting the boreholes 
#f.e. if first borehole is in df 1, start iteration at df 2 
j = 1

for df in dataframes_drilling_copy:
    #iterate over every line in the dfs
    for index, row in df.iterrows():
        #current row is the first reference row in this iteration
        reference_hole = row 
        
        df_holes = []
        
        #add first reference_hole to df
        df_holes.append(reference_hole)
                        
        #iterate over all df in the copied df for the closest hole
        for i in range(j, len(dataframes_drilling_copy)):
            df = dataframes_drilling_copy[i]
            if df.empty:
                #if no boreholes available in this face anymore
                # if df empty we have an uncontinious borehole line
                #idea: end "borehole" at this place and start same procedure later 
                #at next face again
                continue #skip to next iteration step
                
            #try to use just the same types of boreholes for matching process
            matching_rows = df[df.iloc[:, 33] == reference_hole.iloc[33]]
            if matching_rows.empty:
                continue
            distance = matching_rows.iloc[:, 21:24].apply(
                lambda row: np.linalg.norm(row.values - reference_hole.iloc[24:27].values), axis=1)
            
            if not distance.empty:
                closest_hole_ind = distance.idxmin()
                if closest_hole_ind in df.index:
                    closest_hole = df.loc[closest_hole_ind].copy()
                    #check that distance is not bigger than 1 (f.e.)
                    if distance[closest_hole_ind] > 1:
                        continue
                    closest_hole['distance'] = distance[closest_hole_ind]
                    df_holes.append(closest_hole)
                    dataframes_drilling_copy[i] = df.drop(index=closest_hole_ind)  # Update the dataframe in the list
                    reference_hole = closest_hole
                else:
                    print(f"Warning: Index {closest_hole_ind} is not in the DataFrame index")
            else:
                print("Error: Empty Distance")
                
        df_holes_df = pd.DataFrame(df_holes)
        #save it just if it is longer than 5 faces to avoid small datasets
        if len(df_holes_df) > 5:
            list_holes.append(pd.DataFrame(df_holes))
        else:
            list_short_holes.append(pd.DataFrame(df_holes))
    j = j + 1 #in next df, iterate starting with the next df

path_result = Path(r'C:\Users\Data\00_created_Data')

# for i, df in enumerate(list_holes):
#     filename = f'borehole_{i:03d}_greater5faces.csv'  
#     save_path = os.path.join(path_result, filename)
#     df.to_csv(save_path, index=False)   

list_holes = []
#read files again
for file in sorted(path_result.glob('*.csv')): 
    #add here ristriction to number and instead of sorted list
    df = pd.read_csv(file)
    list_holes.append(df)

     
 # =============================================================================
 # borehole plots (3D plot of created long boreholes)
 # =============================================================================
#plot the boreholes in list_boreholes
colors = plt.cm.rainbow(np.linspace(0, 1, len(list_holes)))

def plot_boreholes(borehole_dfs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for df, color in zip(borehole_dfs, colors):
        point_color = color
        line_color = color

        for i in range(df.shape[0]):
            x_start, y_start, z_start = df.iloc[i, 21:24]
            x_end, y_end, z_end = df.iloc[i, 24:27]
            
            #plot points and connect them
            ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], c=point_color)

            #connect end and startpoints of different faces
            if i < df.shape[0] - 1:
                next_x_start, next_y_start, next_z_start = df.iloc[i + 1, 21:24]
                ax.plot([x_end, next_x_start], [y_end, next_y_start], 
                        [z_end, next_z_start], c=line_color)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()

#plot boreholes       
plot_boreholes(list_holes)


# add explosives and measurement station to boreholes
path_expl = Path(r'C:\Users\Data\Explosives & support patern.xlsx')
explosives_df = pd.read_excel(path_expl)

for df in list_holes:
    df['explosives [kg/m3]'] = None
    for index, row in df.iterrows():
        face_no = row['FileName']
        face_no = face_no.replace("_drilling.csv", "")
        face_no = int(face_no) + 1 #correctly connect explosives to drilling
        
        #get correct line within explosives_df
        expl_face = explosives_df[explosives_df.iloc[:,0] == face_no]
        if not expl_face.empty:
            expl = expl_face.iloc[0, 16]
            df.at[index, 'explosives [kg/m3]'] = expl

      
# =============================================================================
# add MWD data to drilling data
# =============================================================================

        
#lists to collect final MWD data of boreholes       
mwd_conture_holes_raw = []
mwd_conture_holes_interpolated = [] #not used later on! delete from code!

#add english column names:
excel_file_path = r'C:\Users\Data\Read me_MWD&Drilling_list_0707.xlsx'
df_MWD = pd.read_excel(excel_file_path, sheet_name='MWD data list')
engl_columns = df_MWD.columns.tolist()
engl_columns.append('FileName')


for df in list_holes:
    
    current_hole = pd.DataFrame() #empty dataframe
    current_hole_interpolated = pd.DataFrame()
    
    for index, row in df.iterrows():
        hole_id = row['Hole ID']
        drilling_file = row['FileName'] #f.e. 1000_drilling.csv

        #find matching MWD file
        mwd_file = drilling_file.replace("drilling", "MWD")  
        path = path_mwd / mwd_file
        
        #as not each drilling file has mwd data, mark those
        if not os.path.exists(path):
            print(f"Error: MWD file for {drilling_file} not found.")
            
            #mark with row of nans
            nan_row = {col: np.nan for col in engl_columns}
            nan_row['Hole ID'] = hole_id
            nan_row['FileName'] = drilling_file
            current_hole = pd.concat([current_hole, pd.DataFrame([nan_row])], ignore_index=True)
            current_hole_interpolated = pd.concat([current_hole_interpolated, pd.DataFrame([nan_row])], ignore_index=True)
            continue
        
        #read matching file
        df_face = pd.read_csv(path)
        df_face['FileName'] = drilling_file #save file as well for later
        df_face.columns = engl_columns #change name to english column name
        df_face_filtered = df_face[df_face.iloc[:,0] == hole_id] #keep just this hole
        
        #drop column 4, 5, 11, 14-22 acc. to meeting
        drop_col = df_face_filtered.columns[[3, 4, 10] + list(range(13, 22))]
        df_face_filtered = df_face_filtered.drop(columns = drop_col)
        df_face_filtered['explosives [kg/m3]'] = row['explosives [kg/m3]']
        
        
        #delete and interpolate first 0.6 m
        df_face_interpolated = df_face_filtered.copy()
        
        #L = df_face_interpolated.iloc[:, 2].max() #max length of this hole
        condition = (df_face_interpolated.iloc[:, 2] <= 0.6)
        
        #set MWD for first and last 0.6 to nans
        df_face_interpolated.loc[condition, df_face_interpolated.columns[3:-2]] = np.nan
        
        try:
            for col in df_face_filtered.columns[3:-2]:
                df_face_filtered[col] = pd.to_numeric(df_face_filtered[col], errors='coerce')
    
            for col in df_face_interpolated.columns[3:-2]:
                df_face_interpolated[col] = pd.to_numeric(df_face_interpolated[col], errors='coerce')
        except ValueError:
            print('error')
        #calculate statistics:
        for col in df_face_filtered.columns[3:-2]:
            df_face_filtered[f'{col} min'] = df_face_filtered[col].min()
            df_face_filtered[f'{col} max'] = df_face_filtered[col].max()
            df_face_filtered[f'{col} mean'] = df_face_filtered[col].mean()
        
        for col in df_face_interpolated.columns[3:-2]:
            df_face_interpolated[f'{col} min'] = df_face_filtered[col].min()
            df_face_interpolated[f'{col} max'] = df_face_filtered[col].max()
            df_face_interpolated[f'{col} mean'] = df_face_filtered[col].mean()
        
        
        #add data to current hole 
        current_hole = pd.concat([current_hole, df_face_filtered], 
                                 axis=0, ignore_index=True)
        
        current_hole_interpolated = pd.concat([current_hole_interpolated, 
                                               df_face_interpolated], 
                                 axis=0, ignore_index=True)
    
    mwd_conture_holes_raw.append(current_hole)
    mwd_conture_holes_interpolated.append(current_hole_interpolated)


# save created boreholes
path_result = Path(r'C:\Users\Data\00_created_Data')

# for i, df in enumerate(mwd_conture_holes_raw):
#     filename = f'borehole_{i:03d}_mwd_raw.csv'  
#     save_path = os.path.join(path_result, filename)
#     df.to_csv(save_path, index=False) 
    
#load mwd
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
# check overlapping of boreholes
# =============================================================================

#find maximum lengths of individual boreholes:
max_depth_dict = {}
max_depth_dict = {'FileName': [], 'Hole ID': [], 'MaxDepth': []}

for df in mwd_conture_holes_raw:
    grouped = df.groupby('FileName')
    for (filename), group in grouped:
        max_depth_value = group['Depth [m]'].max()
        hole_id = group['Hole ID'].iloc[0]
        max_depth_dict['FileName'].append(filename)
        max_depth_dict['Hole ID'].append(hole_id)
        max_depth_dict['MaxDepth'].append(max_depth_value)
        
max_depth_df = pd.DataFrame(max_depth_dict)
# show distribution of maximum borehole lengths
plt.figure(figsize=(10, 6))
sns.histplot(max_depth_df['MaxDepth'], kde=True)
plt.xlabel('Maximum Depth [m]')
plt.ylabel('Count')
plt.title('Distribution of Maximum Depths')
plt.show()
                   
hole_length_list = []

for df in list_holes:
    for index, row in df.iterrows():
        filename = row['FileName']
        hole_id = row['Hole ID']
        x1, y1, z1 = row.iloc[21:24]
        x2, y2, z2 = row.iloc[24:27]
        holelength = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        hole_length_list.append({'FileName': filename, 'Hole ID': hole_id, 'HoleLength': holelength})

hole_length_df = pd.DataFrame(hole_length_list)

combined_df = pd.merge(max_depth_df, hole_length_df, on=['FileName', 'Hole ID'])

#compare both calculated hole length and also maximum depth 
combined_df['dH'] = combined_df.iloc[:, -1] - combined_df.iloc[:, -2]

#check if results are similar
for index, row in combined_df.iterrows():
    if row['dH'] >= 0.1:
        print(f'{row["Hole ID"]} in {row["FileName"]} delte length is {row["dH"]}')


## compare data to initial data 

overlapping_data2 = []

for df in mwd_conture_holes_raw:
    overlap_list2 = []
    next_group_data2 = []
    #one borehole is equal to one FileName per Dataframe
    grouped = df.groupby('FileName')
    
    group_filenames = list(grouped.groups.keys()) #save group names
    num_groups = len(group_filenames) #amount of groupes
    
    for i in range(num_groups):
        filename = group_filenames[i]
        group = grouped.get_group(filename)
        
        # Filter for values > 2.1 m in the current group = actual overlap
        filtered_group = group[group['Depth [m]'] > 1.5]
        overlap_list2.append(filtered_group)
        
        num_values = filtered_group.shape[0] #amount of overlap 
        
        if i + 1 < num_groups: #get matching overlap from next group
            next_filename = group_filenames[i + 1]
            next_group = grouped.get_group(next_filename)
            
            # Get the first 'num_values' rows and the first 10 columns 
            next_filtered_values = next_group.head(num_values).iloc[:, :10]
            next_group_data2.append(next_filtered_values)
    
    # Create DataFrame with overlap data
    overlap2_df = pd.concat(overlap_list2).reset_index(drop=True)
    
    # Concatenate the corresponding next group data > 0.6 m
    if next_group_data2:
        next_group_df2 = pd.concat(next_group_data2).reset_index(drop=True)
        for col in next_group_df2.columns:
            overlap2_df[f'Next_{col}'] = next_group_df2[col]
    
    overlapping_data2.append(overlap2_df)
    

# compare values:
    
for idx, df in enumerate(overlapping_data2):
    try:
        columns_to_convert = [
            'Feed Pressure [Bar]', 'Next_Feed Pressure [Bar]',
            'Flushing Flow [l/min]', 'Next_Flushing Flow [l/min]',
            'Flushing Pressure [Bar]', 'Next_Flushing Pressure [Bar]',
            'Penetration Rate [m/min]', 'Next_Penetration Rate [m/min]',
            'Percussion Pressure [Bar]', 'Next_Percussion Pressure [Bar]',
            'Rotation Pressure [Bar]', 'Next_Rotation Pressure [Bar]',
            'Rotation Speed [RPM]', 'Next_Rotation Speed [RPM]']
        for col in columns_to_convert:
           if col in df.columns:
               df[col] = pd.to_numeric(df[col], errors='coerce')
           else:
               print(f'column {col} is not in dataframe {idx}')
        
        for col in columns_to_convert:
            if df[col].isnull().any():
                print(f'NaN-Werte in Spalte {col} in DataFrame {idx}')
        
        #drop nan lines, as those are not a real overlap (no part for them to overlap,
        # the "next" column is empty)
        df.dropna(subset=columns_to_convert, inplace=True)


        df['d Feed Pressure [Bar]'] = df['Feed Pressure [Bar]'] - df['Next_Feed Pressure [Bar]']
        df['d Flushing Flow [l/min]'] =  df['Flushing Flow [l/min]'] -  df['Next_Flushing Flow [l/min]']
        df['d Flushing Pressure [Bar]'] =  df['Flushing Pressure [Bar]'] -  df['Next_Flushing Pressure [Bar]']
        df['d Penetration Rate [m/min]'] =  df['Penetration Rate [m/min]'] -  df['Next_Penetration Rate [m/min]']
        df['d Percussion Pressure [Bar]'] =  df['Percussion Pressure [Bar]'] -  df['Next_Percussion Pressure [Bar]']
        df['d Rotation Pressure [Bar]'] =  df['Rotation Pressure [Bar]'] -  df['Next_Rotation Pressure [Bar]']
        df['d Rotation Speed [RPM]'] =  df['Rotation Speed [RPM]'] -  df['Next_Rotation Speed [RPM]']
    except KeyError as e:
       print(f'Missing column in {e} in dataframe {idx}')

all_overlap2 = []
for df in overlapping_data2:
    if not df.empty and idx not in range(457, 488):
        all_overlap2.append(df)
all_overlap_df2 = pd.concat(all_overlap2, ignore_index = True)
        

cols = ['d Feed Pressure [Bar]', 'd Flushing Flow [l/min]', 'd Flushing Pressure [Bar]', 
    'd Penetration Rate [m/min]', 'd Percussion Pressure [Bar]', 
    'd Rotation Pressure [Bar]', 'd Rotation Speed [RPM]']

# fig, axes = plt.subplots(len(cols), 1, figsize=(10, 20))

# for idx, col in enumerate(cols):
#     sns.histplot(all_overlap_df2[col], kde=True, ax=axes[idx])
#     axes[idx].set_xlabel(col)
#     axes[idx].set_ylabel('Count')

# fig.suptitle('Data distribution for overlapping areas', fontsize=16)
# plt.tight_layout()
# plt.show()
        
#detect outliers:
    
cleaned_overlap_df2 = all_overlap_df2.copy()

for col in cols:
    Q1 = cleaned_overlap_df2[col].quantile(0.25)
    Q3 = cleaned_overlap_df2[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_overlap_df2 = cleaned_overlap_df2[(cleaned_overlap_df2[col] >= lower_bound) & 
                                    (cleaned_overlap_df2[col] <= upper_bound)]

fig, axes = plt.subplots(len(cols), 1, figsize=(10, 20))
for idx, col in enumerate(cols):
    sns.histplot(cleaned_overlap_df2[col], kde=True, ax=axes[idx])
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')

fig.suptitle('Data distribution for calculated parameters after outlier removal', fontsize=16)
plt.tight_layout()
plt.show()

cleaned_overlap_df2 = cleaned_overlap_df2.iloc[:, :-14]


#try using overlap for data:

# idea: delete every line that matches next_holeID and next_time
# check every depth > 1.5 if in list, else delete?

#copy df
mwd_conture_holes_wo_overlap = [df.copy() for df in mwd_conture_holes_interpolated]

#get combinations of nextHoleID and nextTime to find which to delete
combinations = set(zip(cleaned_overlap_df2['Next_Time'], cleaned_overlap_df2['Next_Hole ID']))
for df in mwd_conture_holes_wo_overlap:
    mask = df.apply(lambda row: (row['Time'], row['Hole ID']) in combinations, axis=1)
    df.drop(df[mask].index, inplace=True)

#delete overlap which is not used (so > 1.5 but not in file)
combinations_to_keep = set(zip(cleaned_overlap_df2['Depth [m]'], cleaned_overlap_df2['Time'], cleaned_overlap_df2['Hole ID']))
for df in mwd_conture_holes_wo_overlap:
    mask = df.apply(lambda row: (row['Depth [m]'], row['Time'], row['Hole ID']) in 
                    combinations_to_keep or 
                    row['Depth [m]'] <= 1.5, axis=1)
    df.drop(df[~mask].index, inplace=True)
  
    
#compare overlap data with initial data --> correlation matrix
overlap_deleted = []
greater_than_0_6 = []

for df in mwd_conture_holes_wo_overlap:
    mask_deleted = df['Depth [m]'] > 1.5  
    deleted_rows = df[mask_deleted].copy()
    
    mask_greater_than_0_6 = df['Depth [m]'] > 0.6 
    rows_greater_than_0_6 = df[mask_greater_than_0_6].sample(n=len(deleted_rows), random_state=42)  # gleiche Anzahl an Zeilen
    
    overlap_deleted.append(deleted_rows)
    greater_than_0_6.append(rows_greater_than_0_6)

overlap_deleted_df = pd.concat(overlap_deleted)
greater_than_0_6_df = pd.concat(greater_than_0_6)

out_path = r'C:\Users\Boxplots'
hole_ids = [569, 583, 597]
cols = ["Feed Pressure [Bar]", "Flushing Flow [l/min]", "Flushing Pressure [Bar]",
    "Penetration Rate [m/min]", "Percussion Pressure [Bar]", "Rotation Pressure [Bar]"]

deleted_779 = overlap_deleted_df[overlap_deleted_df['Hole ID'].isin(hole_ids) & 
                                 (overlap_deleted_df['FileName'] == '779_drilling.csv')]
greater_780 = greater_than_0_6_df[greater_than_0_6_df['Hole ID'].isin(hole_ids) & 
                                  (greater_than_0_6_df['FileName'] == '780_drilling.csv')]

#create correlation matrix
for hole_id in hole_ids:
    deleted_hole = deleted_779[deleted_779['Hole ID'] == hole_id]
    greater_hole = greater_780[greater_780['Hole ID'] == hole_id]
    
    if len(greater_hole) > len(deleted_hole):
        greater_hole = greater_hole.iloc[:len(deleted_hole)]

    correlation_deleted = deleted_hole[cols].corr()
    correlation_greater = greater_hole[cols].corr()
    
    #check if amount is same
    deleted_row_count = len(deleted_hole)
    greater_row_count = len(greater_hole)
    print(f'Hole ID {hole_id}:')
    print(f'  Deleted (Face 779): {deleted_row_count} rows')
    print(f'  Greater (Face 780): {greater_row_count} rows\n')
    
    plt.figure(figsize=(16, 6)) #plot for both heat maps
    
    # Heatmap (Face 779)
    plt.subplot(1, 2, 1)
    sns.heatmap(correlation_deleted, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Overlapping Areas (Hole ID: {hole_id}, Face 779)')

    # Heatmap(Face 780)
    plt.subplot(1, 2, 2)
    sns.heatmap(correlation_greater, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Overlapping Areas (Hole ID: {hole_id}, Face 780)')

    plot_filename = f"comparison_overlap_holeID{hole_id}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, plot_filename))
    plt.close()


# =============================================================================
# prepare Data such that overlap is not included
# =============================================================================

#copy raw data
mwd_new_raw = [df.copy() for df in mwd_conture_holes_raw]
mwd_new = []

for df in mwd_new_raw:
    df = df[df['Depth [m]'] <= 2.1] #delte to long overlap
    result_df = pd.DataFrame()
    #get all groups (baseboreholes)
    file_names = df["FileName"].unique()

    prev_sub_df = pd.DataFrame()
    for j in range(len(file_names)):
        current_file_name = file_names[j]
        current_sub_df = df[df["FileName"] == current_file_name]
        
        #consider previous filename because of overlap
        if not prev_sub_df.empty:
            #count overlap in previous file
            count = prev_sub_df[prev_sub_df["Depth [m]"] >= 1.5].shape[0]
            # keep just values after overlap (count)
            current_sub_df = current_sub_df.iloc[count:]

        #add current sub to results
        result_df = pd.concat([result_df, current_sub_df])

        # Update prev_sub_df 
        prev_sub_df = current_sub_df

    #add last prev sub
    result_df = pd.concat([result_df, prev_sub_df])

    mwd_new.append(result_df)

#create a total depth for these long boreholes
for df in mwd_new:
    start_depth = df["Depth [m]"].iloc[0]
    total_depth = [start_depth + 0.02 * i for i in range(len(df))]
    df["Total Depth [m]"] = total_depth



#find outliers and set those to Nan's    

#get total data to calculate Q1, Q3,...
mwd_new_total = pd.DataFrame()
for df in mwd_new:
    mwd_new_total = pd.concat([mwd_new_total, df], axis=0, ignore_index=True)

for column_index in range(3, 9):
    Q1 = mwd_new_total.iloc[:, column_index].quantile(0.25)
    Q3 = mwd_new_total.iloc[:, column_index].quantile(0.75)
    IQR = Q3 - Q1
    for df in mwd_new:
        df[f'is_outlier_{column_index}'] = ((df.iloc[:, column_index] < (Q1 - 1.5 * IQR)) | (df.iloc[:, column_index] > (Q3 + 1.5 * IQR)))
    
mwd_new_clean = []

#set outliers to nans    
for df in mwd_new:
    df_copy = df.copy(deep=True)
    df_copy['is_outlier_any'] = df_copy[[f'is_outlier_{col}' for col in range(3, 9)]].any(axis=1)
    #set values to nan for col 3 to 8 if there is an outlier detected
    for column_index in range(3, 9):
        df_copy.iloc[:, column_index] = df_copy.apply(lambda row: np.nan if row['is_outlier_any'] else row[column_index], axis=1)
    df_copy.drop(columns=[f'is_outlier_{col}' for col in range(3, 9)] + ['is_outlier_any'], inplace=True)
    mwd_new_clean.append(df_copy)

#delete 0,6 m values        
for df in mwd_new_clean:
    condition = (df.iloc[:, 2] <= 0.6)
    #set MWD for first 0.6 to nans
    df.loc[condition, df.columns[3:9]] = np.nan    

# #save files
# path_result = Path(r'C:\Users\Data\00_created_Data')



#load data
path = Path(r'C:\Users\Data\00_created_Data')

mwd_new_raw = []

for file in path.glob('*.csv'): 
    df = pd.read_csv(file)
    mwd_new_raw.append(df) 


#load data
path = Path(r'C:\Users\Data\00_created_Data')

mwd_new_clean = []

for file in path.glob('*.csv'): 
    df = pd.read_csv(file)
    mwd_new_clean.append(df) 

    
# =============================================================================
# analyse data created (just MWD)
# =============================================================================

#find outliers

#create df with entire mwd data that will be used
mwd_raw_total = pd.DataFrame()
for df in mwd_conture_holes_raw:
    mwd_raw_total = pd.concat([mwd_raw_total, df], axis=0, ignore_index=True)

#calculate IQR
for column_index in range(3, 9):
    Q1 = mwd_raw_total.iloc[:, column_index].quantile(0.25)
    Q3 = mwd_raw_total.iloc[:, column_index].quantile(0.75)
    IQR = Q3 - Q1
    IQR = Q3 - Q1
    
    for df in mwd_conture_holes_raw:
        df[f'is_outlier_{column_index}'] = ((df.iloc[:, column_index] < (Q1 - 1.5 * IQR)) | (df.iloc[:, column_index] > (Q3 + 1.5 * IQR)))

mwd_conture_holes_interpolated_old = []

#keep old values (delete this part later)
for df in mwd_conture_holes_interpolated:
    df_new = df.copy()
    mwd_conture_holes_interpolated_old.append(df_new)
 
#create new list for interpolation
mwd_conture_holes_interpolated = []
    
for df in mwd_conture_holes_raw:
    df['is_outlier_any'] = df[[f'is_outlier_{col}' for col in range(3, 9)]].any(axis=1)
    #set values to nan for col 3 to 8 if there is an outlier detected
    for column_index in range(3, 9):
        df.iloc[:, column_index] = df.apply(lambda row: np.nan if row['is_outlier_any'] else row[column_index], axis=1)
    df.drop(columns=[f'is_outlier_{col}' for col in range(3, 9)] + ['is_outlier_any'], inplace=True)
    df_new = df.copy()
    mwd_conture_holes_interpolated.append(df_new)
    
    
for df in mwd_conture_holes_interpolated:
    condition = (df.iloc[:, 2] <= 0.6)
    #set MWD for first 0.6 to nans
    df.loc[condition, df.columns[3:9]] = np.nan
    

#print boxplots
outliers_boxplot = pd.DataFrame()
out_path_boxplot = Path(r'C:\Users\Data\boxplot')
#via boxplot diagrams
for column in mwd_raw_total.columns[3:10]:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=mwd_raw_total[column])
    plt.title(f'Boxplot for {column}')
    plt.show()
    # plot_filename = f"boxplot_{column}_rawdata.jpg"
    # plt.savefig(os.path.join(out_path_boxplot, plot_filename))
    # plt.close()
    
for column in mwd_raw_total.columns[3:10]:
    #save outliers
    Q1 = mwd_raw_total[column].quantile(0.25)
    Q3 = mwd_raw_total[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = mwd_raw_total[((mwd_raw_total[column] < (Q1 - 1.5 * IQR)
                               ) | (mwd_raw_total[column] > (Q3 + 1.5 * IQR)))]
    
    outliers_boxplot = pd.concat([outliers_boxplot, outliers], axis=0, ignore_index=True)
    outliers_boxplot = outliers_boxplot.drop_duplicates()

#check depth of these outliers --> is it in first and/or last 0.6 m?
plt.figure(figsize=(10,6))
sns.boxplot(data=outliers_boxplot.iloc[:,2]) 
plt.title('Boxplot Depth of outliers')
plt.show()


# =============================================================================
# Code for checking correlation of data
# =============================================================================

#correlation matrix
##analyse correlation for one entire borehole
def correlation_per_face(df):
    borehole = df.groupby('FileName') 
    results = {} #dict for results
    for filename, group in borehole:
        hole_id = group['Hole ID'].iloc[0] #same for every element within group
        cols = group.iloc[:, 3:9].columns.tolist() 
        group_selected = group[cols].apply(pd.to_numeric, errors='coerce')
        correlation_matrix = group_selected.corr()    
        results[(filename, hole_id)] = correlation_matrix
    return results

mwd_raw_borehole_cor = []
for df in mwd_new_raw[560:580]:
    correlation = correlation_per_face(df)
    mwd_raw_borehole_cor.append(correlation)
    i = 0 # to stop iteration 5 iterations per df
    for (filename, hole_id), corr_matrix in correlation.items():
        #print(f"Hole {hole}:\n{corr_matrix}\n") 
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Correlation Matrix raw data for Hole {hole_id} in {filename}')
        #plt.show()
        plt.tight_layout()
        plot_filename = f"{filename}_{hole_id}_cor_matrix_rawdata.jpg"
        plt.savefig(os.path.join(out_path, plot_filename))
        plt.close()
        
        i += 1
        if i >= 3:
            break
        
    
#plot specific boreholes

idx = [603, 604, 607]

for i in idx:
    df = mwd_conture_holes_raw[i]
    df_right = df[df["FileName"].isin(["779_drilling.csv", "780_drilling.csv"])]

    cols = df.iloc[:, 3:9].columns.tolist()  
    df_selected = df_right[cols].apply(pd.to_numeric, errors='coerce')  # 
    correlation_matrix = df_selected.corr()
    hole_id = df_right['Hole ID'].iloc[0]
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix raw data for Hole {hole_id} in face 779 and 780')
    plt.tight_layout()
    plt.show()       


# =============================================================================
# Analyze parameters along length for 3 boreholes in same face
# =============================================================================

#find specific boreholes
matching_indices = []
filename = '780_drilling.csv'
hole_ids = [583, 569, 597]

for i, df in enumerate(mwd_new_raw):
    matches_all = (df["Hole ID"].isin(hole_ids)) & (df["FileName"] == filename)
    if matches_all.any():
        matching_indices.append(i)


#chosen: hole 583 569 597 in face 780
indices_to_plot = [572, 573, 576]

for idx in indices_to_plot:
    df = mwd_conture_holes_interpolated[idx]
    df.iloc[:, 3:9] = df.iloc[:, 3:9].apply(pd.to_numeric, errors='coerce')
    filtered_df = df[df["FileName"] == "780_drilling.csv"] #just that face
    filtered_df = filtered_df.dropna(subset=filtered_df.columns[3:9]) #drop nans
    hole_id = filtered_df["Hole ID"].iloc[0]
    file_name = filtered_df["FileName"].iloc[0]
    
    filtered_df2 = df[df["FileName"] == "779_drilling.csv"]
    filtered_df2 = filtered_df2.dropna(subset=filtered_df.columns[3:9])
    #calculate that values such that they are plotted before face 780
    max_depth = filtered_df2['Depth [m]'].max()
    filtered_df2['Depth [m]'] = filtered_df2['Depth [m]'] - max_depth
    
    combined_df = pd.concat([filtered_df, filtered_df2], ignore_index=True)
    
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(15, 10))
    fig.suptitle(f"Hole ID: {hole_id}, Face 779 and 780 (Interpolated Data)")

    #select parameters
    columns = combined_df.columns[3:9]

    #generate a plot for each parameter
    for i, column in enumerate(columns):
        ax = axes[i]
        x = combined_df['Depth [m]'].tolist()
        y = combined_df[column].tolist()
        ax.plot(x, y, marker='o', linestyle='', color='b')
        ax.set_xlabel('Depth [m]')
        ax.set_ylabel(column)
        ax.set_title(f'{column}')
 
    plt.tight_layout()
    plt.show()


#plot all in one figure
colors = ['b', 'g', 'r']  
indices_to_plot = [634, 635, 638]
fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(15, 18))
fig.suptitle("Comparison of holes in face 779 + 780 (Raw Data)", fontsize=16)

for idx, color in zip(indices_to_plot, colors):
    df = mwd_new_raw[idx]
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
 
#save relevant holes explicitly
path_result = Path(r'C:\Users\Plots')
for idx in indices_to_plot:
    df = mwd_new_raw[idx]
    
    filtered_df = df[df["FileName"] == "780_drilling.csv"]
    hole_id = filtered_df['Hole ID'].iloc[0]
    filename = f'borehole_{hole_id}_face780_raw.csv'
    save_path = os.path.join(path_result, filename)
    filtered_df.to_csv(save_path, index=False)
    
    filtered_df2 = df[df["FileName"] == "779_drilling.csv"]
    filename2 = f'borehole_{hole_id}_face779_raw.csv'
    save_path2 = os.path.join(path_result, filename2)
    filtered_df2.to_csv(save_path2, index=False)

    #save also combined if needed later on
    combined_df = pd.concat([filtered_df, filtered_df2], ignore_index=True)
    combined_filename = f'borehole_{hole_id}_face779_780_raw.csv'
    combined_save_path = os.path.join(path_result, combined_filename)
    combined_df.to_csv(combined_save_path, index=False) 


# =============================================================================
# data distribution raw data 
# =============================================================================
#ranges for figures such that they are compareable 
global_ranges = {col: (min(df[col].min() for df in mwd_conture_holes_raw + mwd_conture_holes_interpolated),
                      max(df[col].max() for df in mwd_conture_holes_raw + mwd_conture_holes_interpolated))
                for col in mwd_conture_holes_raw[0].columns[3:9]}

def bins_calc(data):
    q75, q25 = np.percentile(data.dropna(), [75, 25])
    bin_width = 2 * (q75 - q25) * len(data.dropna())**(-1/3)
    bins = int((data.max() - data.min()) / bin_width)
    return max(1, bins) 

cols = mwd_conture_holes_raw[0].columns[3:9] #error: chekc variable name
all_data_raw_combined = pd.concat([df[cols] for df in mwd_conture_holes_raw])
bin_counts = {col: bins_calc(all_data_raw_combined[col]) for col in cols}

num_cols = len(cols)
fig, axes = plt.subplots(nrows=2, ncols=(num_cols+1)//2, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(cols):
    all_data_raw = pd.concat([df[col] for df in mwd_conture_holes_raw])
    sns.histplot(all_data_raw, bins=bin_counts[col], binrange=global_ranges[col], 
                 kde=True, ax=axes[idx])
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')
fig.suptitle('Data distribution raw data', fontsize=16)
plt.tight_layout()
plt.show()

#for interpolated data
fig, axes = plt.subplots(nrows=2, ncols=(num_cols+1)//2, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(cols):
    all_data_int = pd.concat([df[col] for df in mwd_conture_holes_interpolated])
    sns.histplot(all_data_int, bins=bin_counts[col], binrange=global_ranges[col], 
                 kde=True, ax=axes[idx])
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')
fig.suptitle('Data distribution clean and interpolated data', fontsize=16)
plt.tight_layout()
plt.show()


#overall statistics

stat_raw = []
cols = mwd_new_raw[0].columns[3:9]
cols
for col in cols:
    all_data = pd.concat([df[col].dropna() for df in mwd_new_raw])
    stat_raw.append({'Column': col, 'Mean': round(all_data.mean(),2),
        'Median': round(all_data.median(),2),'Min': round(all_data.min(),2),
        'Max': round(all_data.max(),2),'Range': round(all_data.max() - all_data.min(),2),
        'SD': round(all_data.std(),2),#'Count': len(all_data)
    })
    
stat_raw_df = pd.DataFrame(stat_raw)
stat_raw_df.set_index('Column', inplace = True)
stat_raw_df = stat_raw_df.T

fig, ax = plt.subplots(figsize=(12, 2)) 
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=stat_raw_df.values, colLabels=stat_raw_df.columns, 
                 rowLabels=stat_raw_df.index, loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(8)

for key, cell in table.get_celld().items():
    if key[1] == 0:
        cell.set_width(0.15)
    if key[1] > 0:  # Adjust width of all columns except the first
        cell.set_width(0.2)  # set the width for columns starting from the second one


plt.title('Summary Statistics', fontsize = 12)
plt.show()


# =============================================================================
# plots
# =============================================================================

# plot for presentation (face 1035 comparison used an unused holes)
desired_hole_ids = ['680', '681', '682', '683', '684', '652', '653', '654', '655', '669', '686']
fig, ax = plt.subplots()

# face 1035
df = dataframes_drilling[536]  
for i in range(len(df)):
    x, y, z = df.iloc[i, 3:6]  
    x = float(x)
    z = float(z)
    hole_id = df.iloc[i, 0]
    
    if hole_id in desired_hole_ids:
        ax.scatter(x, z, label=hole_id, s=100)  #coloured
    else:
        ax.scatter(x, z, color='#D3D3D3', s=100)  # grey

plt.title('face no. 1035', fontsize = 18)
ax.set_xlabel('Planned Start Point X (Plan Coordinates)')
ax.set_ylabel('Planned Start Point Z (Plan Coordinates)')
legend = ax.legend(title='Hole IDs')
plt.show()



