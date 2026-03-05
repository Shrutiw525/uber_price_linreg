import streamlit as st
import pandas as pd
st.title("💰Uber Price Prediction🚗")
#upload dataset
df=pd.read_csv("uber_price.csv")
#input from user
kilometers=st.number_input("Enter the distance in kilometers",min_value=1.0,step=0.5)
passengers=st.number_input("Enter the number of passengers",min_value=1,max_value=10)
#finding the price
if st.button("Predict Price"):
    #using model to predict price
    x=df.drop("Fare",axis=1)
    y=df["Fare"]
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    from sklearn.linear_model import LinearRegression
    model=LinearRegression()
    model.fit(x_train,y_train)
    price=model.predict([[kilometers,passengers]])
    st.success(f"The predicted price for your Uber ride is: ₹{price[0]:.2f}")
    #why price[0]?the whole price is in an array and we want the first element which is the predicted price
    #equal splitting if more than one passenger
    if passengers>1:
        split_price=price[0]/passengers
        st.info(f"The price per passenger is: ₹{split_price:.2f}")





