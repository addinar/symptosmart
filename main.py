import streamlit as st
from symptosmart import Symptosmart

file_1, file_2 = 'dm1', 'dm2'
symptosmart = Symptosmart(file_1, file_2)
st.image('images/logo.jpg', width=200)
st.caption("Instant symptom analysis, triage, and healthcare recommendations at home.")
start = st.container()
start.header("How are you feeling?")
user_input = start.text_input(label="e.g. 'high fever'", placeholder='Start by entering a symptom')

if user_input:
    symptosmart.search_filtering(user_input)

if symptosmart.diagnose_me:
    symptosmart.triage()
    if symptosmart.urgent:
        st.subheader("Your condition may be urgent.")
        symptosmart.diagnose()
    else:
        st.subheader("Your condition is most likely not urgent.")
        symptosmart.non_urgent_suggestions()
            
with st.popover('ℹ️'):
  st.caption("This project is possible with the following dataset: https://huggingface.co/datasets/shanover/disease_symptoms_prec_full")
