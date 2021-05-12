import streamlit as st

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import xgboost as xgb

# merged dataframe
df_exoplanet_vf = pd.read_csv('planets.csv')

# Selecting all numerical column from data fram
numeical_columns_list = df_exoplanet_vf.select_dtypes(include=np.number).columns.tolist()
df_exoplanet_num= df_exoplanet_vf[numeical_columns_list]

# Selecting main categorical columns
df_exoplanet_cat = df_exoplanet_vf[['pl_letter','discoverymethod','disc_locale']]

# setting them into numerical value using factorization
df_exoplanet_cat['pl_letter'] = df_exoplanet_cat['pl_letter'].factorize()[0]
df_exoplanet_cat['discoverymethod'] = df_exoplanet_cat['discoverymethod'].factorize()[0]
df_exoplanet_cat['disc_locale'] = df_exoplanet_cat['disc_locale'].factorize()[0]

# merging dataset of selected columns 
df_exoplanet_rf = df_exoplanet_num.join(df_exoplanet_cat)

# ...and splitting dataset on 'P_HABITABLE' none or not
df_exoplanet_rf_1 = df_exoplanet_rf[df_exoplanet_rf['P_HABITABLE'].notna()]
df_exoplanet_rf_2 = df_exoplanet_rf[df_exoplanet_rf['P_HABITABLE'].isna()]

# filling missing values with the mean of each column
df_exoplanet_rf_1.fillna(df_exoplanet_rf_1.mean(), inplace=True)

# filling unknown 'P_HABITABLE' with 0 for ML sake
df_exoplanet_rf_2[df_exoplanet_rf_2['P_HABITABLE']!=0] = 0

# starting ML with XGboost
y = df_exoplanet_rf_1["P_HABITABLE"]
X = df_exoplanet_rf_1.drop("P_HABITABLE", axis=1)

#training data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)

# fitting model on training data
model = XGBClassifier()
model.fit(X_train, y_train)

# making prediction on unknown dataset
df_exoplanet_rf_2["predictions"] = model.predict(df_exoplanet_rf_2.drop(columns='P_HABITABLE'))

df_test = df_exoplanet_vf[['pl_name','S_CONSTELLATION']]  

df_final = pd.merge(df_test,df_exoplanet_rf_2,left_index=True,right_index=True)
st.write(df_final)
