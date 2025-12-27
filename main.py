import pandas as pd
import streamlit as st 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
data = pd.read_csv(r"C:/Users/test/OneDrive/Desktop/study_data.csv")
st.title("Linear Regression App")
x = data[["hoursStudied"]]
y = data["examScore"] 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(x,y)
st.title("EXAM SCORE PREDICTOR")
st.write("ENTER HOURS STUDIED TO PREDICT THE EXAM SCORE")
hours=st.number_input("hoursStudied:",min_value=0.0,step=0.1)
if st.button("predict score"):
    predicted_score=model.predict([[hours]])[0]
    st.success(f"predicted score:{predicted_score:.2f}")
st.write("###sample training data")
st.dataframe(data)




 
 



