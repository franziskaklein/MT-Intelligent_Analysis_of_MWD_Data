# MT-Intelligent_Analysis_of_MWD_Data
Code for the Masterthesis: "Intelligent Analysis of MWD Data: A Deep Learning Approach Using Transformers and LSTM"

The first four files include the data preparation:

1) data_check.py
   familiarize with data, check formats, delimeters, datatypes

2) preprocessing.py
   adapt datatypes, handle error boreholes

3) data_combining.py
   create long boreholes by matching boreholes from different faces with each other + plot results (drilling data)
   add MWD data as well as explosives to boreholes clean data: detect outliers and delete first 0.6 m
   check overlap of boreholes

4) final_data_analyis.py
   create smaller dataset for models as training and test data

The next files include all the interpolation and ML models that were finally used:

5) interpolation_methods.py
   includes linear interpolation, spline and kriging interpolation + analysis

6) transformer_multiple_timesteps.py
   includes the analysed transformer model + analysis for one feature and 10 time steps

7) LSTM_RF_multiple_timesteps.py
   includes the LSTM and Random Forest model + analysis for one feature and 10 time steps

8) LSTM_multiple_timesteps_multiple_features.py
   includes the final LSTM models forward + backward + Analysis

9) RF_multiple_timesteps_multiple_features.py
    includes final Random Forest model + Analysis
   
