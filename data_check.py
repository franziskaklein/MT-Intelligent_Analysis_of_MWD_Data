# -*- coding: utf-8 -*-
"""

@author: klein

initial Data Check:
check delimeters, datatypes, nulls / NaNs
"""

import pandas as pd
from pathlib import Path 
import csv


#path in which all the files are stored
path_drilling = Path(r'C:\Users\Data\01_drilling')
path_mwd = Path(r'C:\Users\Data\02_mwd')

# =============================================================================
# drilling data
# =============================================================================

#check delimeter

delimiter_drilling = []

for file in path_drilling.glob('*.csv'): #use just .csv files in folder
    with file.open('r', encoding='utf-8') as drill_file:
        drill_file.readline() #skip first line = heading
        second_line = drill_file.readline()
        
        #check file for format of csv 
        dialect = csv.Sniffer().sniff(second_line)
        
        #append delimiter to a list
        delimiter_drilling.append(dialect.delimiter)
        
#change list in a set to check if there are different delimiters used

if len(set(delimiter_drilling)) == 1:
    print(set(delimiter_drilling))
else:
    print('Different delimiters used')


#check for empty spaces (nulls) and if each column contains just one datatype
drilling_missing_files = pd.DataFrame(columns=['Filename', 'Column', 'Value'])
for file in path_drilling.glob('*.csv'):
    df_drilling = pd.read_csv(file, sep = ',')
    
    for column in df_drilling:
        #check for nulls in each column
        if df_drilling[column].isnull().any():
            missing_info = pd.DataFrame({'Filename': [file.name],
                                         'Column': [column],
                                         'Note': ['nulls']})
            drilling_missing_files = pd.concat([drilling_missing_files, 
                                                missing_info], ignore_index=True)
        #check for - instead of values
        if df_drilling[column].astype(str).str.contains(r'^-').any():
            missing_info = pd.DataFrame({'Filename': [file.name],
                                         'Column': [column],
                                         'Note': ['-']})
            drilling_missing_files = pd.concat([drilling_missing_files, 
                                                missing_info], ignore_index=True)
        #check data types of each column
        datatypes = df_drilling[column].apply(type).unique()
        if len(datatypes) > 1:
            print(f'{column} in {file.name} contains: {datatypes}') 
        #else:
         #   print('All columns are of the same datatypes')

#check datatypes used for each column
print(df_drilling.dtypes)


# =============================================================================
# MWD data
# =============================================================================



path_mwd = Path(r'C:\Users\Data\02_mwd')

#check delimeter

delimiter_mwd = []

for file in path_mwd.glob('*.csv'): #use just .csv files in folder
    with file.open('r', encoding='utf-8') as mwd_file:
        mwd_file.readline() #skip first line = heading
        second_line = mwd_file.readline()
        
        #check file for format of csv 
        dialect = csv.Sniffer().sniff(second_line)
        
        #append delimiter to a list
        delimiter_mwd.append(dialect.delimiter)
        
#change list in a set to check if there are different delimiters used

delimiter_set = (set(delimiter_mwd))
if len(delimiter_set) == 1:
    print(delimiter_set)
if len(delimiter_set) > 1:
    print(f'Different delimiters used: {delimiter_set}')



#check for empty spaces (nulls) and if each column contains just one datatype
mwd_missing_files = pd.DataFrame(columns=['Filename', 'Column', 'Value'])

for file in path_mwd.glob('*.csv'):
    df_mwd = pd.read_csv(file, sep = ',') 

    for column in df_mwd:
        #check for nulls in each column
        if df_mwd[column].isnull().any():
            missing_info = pd.DataFrame({'Filename': [file.name],
                                         'Column': [column],
                                         'Note': ['nulls']})
            mwd_missing_files = pd.concat([mwd_missing_files, missing_info], 
                                          ignore_index=True)
        #check for - instead of values
        if df_mwd[column].astype(str).str.contains(r'^-').any():
            missing_info = pd.DataFrame({'Filename': [file.name],
                                         'Column': [column],
                                         'Note': ['-']})
            mwd_missing_files = pd.concat([mwd_missing_files, missing_info], 
                                          ignore_index=True)
        #check data types of each column
        datatypes = df_mwd[column].apply(type).unique()
        if len(datatypes) > 1:
            print(f'{column} in {file.name} contains: {datatypes}') 

print(df_mwd.dtypes)



