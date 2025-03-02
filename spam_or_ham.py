import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
model_path = r"C:\python_project\machine learning task\Task_streamlit\NB_model.pkl"
vectorizer_path = r"C:\python_project\machine learning task\Task_streamlit\vectorize.pkl"
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Model or vectorizer file not found. Please check file paths.")
else:
    model= pickle.load(open(model_path,"rb")) 
    vect= pickle.load(open(vectorizer_path,"rb")) 
if not hasattr(vect, "transform"):
    st.error("Loaded vectorizer is not fitted. Please retrain and save it properly.")
st.title("Spam message classifier")
st.write("Enter a message below to check if its spam or not spam.")
user_input=st.text_area("Message","")
st.markdown("""
                <style>
                div.stButton>button:first-child{
                    background-color: blue;
                    color: white;
                    font-size: 18px;
                    padding: 10px 20px;
                    border_radius: 5px;
                    border: none;
                    transition: 0.3s;
                }
                div.stButton>button:first-child:hover{
                    background-color: black;
                }
                </style>
               """,unsafe_allow_html=True)
if st.button("predict"):
    if user_input.strip():
        input_data=vect.transform([user_input])
        prediction=model.predict(input_data)[0]
        if prediction==1:
            st.error("This message is spam!")
        else:
            st.success("This message is ham(not spam)")
    else:
        st.warning("Please enter a message to predict.")
       