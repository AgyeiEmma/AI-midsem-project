import streamlit as st #importing streamlit
import numpy as np #importing numpy
import pickle #importing pickle
import pandas as pd #importing pandas


st.title('MIDSEM FIFA Rating Project')
#
# col1, col2 = st.columns(2)
#
# col1.image(str('fifa-soccer.png'))
image_column, content_column = st.columns([1, 4])

# Add the image to the first column

# Add content to the second column




column_1 = st.number_input("potential", min_value=0, max_value=100)
column_2 = st.number_input("value_eur", min_value=0, max_value=100)
column_3 = st.number_input("wage_eur", min_value=0, max_value=100)
column_4 = st.number_input("age", min_value=0, max_value=100)
column_5 = st.number_input("release_clause_eur", min_value=0, max_value=100)
column_6 = st.number_input("shooting", min_value=0, max_value=100)
column_7 = st.number_input("passing", min_value=0, max_value=100)
column_8 = st.number_input("dribbling", min_value=0, max_value=100)
column_9 = st.number_input("physic", min_value=0, max_value=100)
column_10 = st.number_input("attacking_short_passing", min_value=0, max_value=100)
column_11 = st.number_input("skill_long_passing", min_value=0, max_value=100)
column_12 = st.number_input("movement_reactions", min_value=0, max_value=100)
column_13 = st.number_input("power_shot_power", min_value=0, max_value=100)
column_14 = st.number_input("mentality_vision", min_value=0, max_value=100)
column_15 = st.number_input("menality_composure", min_value=0, max_value=100)
column_16 = st.number_input("goalkeeping_speed", min_value=0, max_value=100)




slider_value = [column_1, column_2, column_3, column_4, column_5, column_6, column_7, column_8, column_9, column_10, column_11, column_12, column_13, column_14, column_15, column_16]
slider_value = np.array([slider_value])


scaler = pickle.load(open('scaler.pkl', 'rb'))
scaled_user_inputs = pd.DataFrame(scaler.transform(slider_value))

model = pickle.load(open('midsem.pkl', 'rb'))

overall = model.predict(scaled_user_inputs)

if st.button('SUBMIT'):
    st.write(round(overall[0]))



