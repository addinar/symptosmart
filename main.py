import streamlit as st
from symptosmart import Symptosmart

file_1, file_2 = 'dm1', 'dm2'
key = st.sidebar.text_input(label='Enter OpenAI API Key.')
symptosmart = Symptosmart(file_1, file_2, key)
st.image('images/logo.jpg', width=200)
st.caption("Instant symptom analysis, triage, and healthcare recommendations at home.")
start = st.container()
start.header("How are you feeling? Select symptoms you've experienced.")
# user_input = start.text_input(label="e.g. 'high fever'", placeholder='Start by entering a symptom')
with start:
    symptoms = symptosmart.multiselect()
    diagnose_me = st.button('Diagnose me!')

if diagnose_me:
    symptosmart.triage()
    if symptosmart.urgent:
        st.subheader("Your condition may be urgent.")
        symptosmart.diagnose(symptoms)
    else:
        st.subheader("Your condition is most likely not urgent.")
        symptosmart.non_urgent_suggestions()
            
with st.popover('ℹ️'):
  st.caption("This project is possible with the following dataset: https://huggingface.co/datasets/shanover/disease_symptoms_prec_full")
