import pickle
import pandas as pd
import time
import streamlit as st
import numpy as np
from openai import OpenAI

class Symptosmart:
    def __init__(self, file_1, file_2, key=None):
        self.file_1 = file_1
        self.file_2 = file_2
        self.openai_key = key
        self.load_pickles()
        self.diagnose_me = False
        self.urgent = False

    def load_pickles(self):
        file_1, file_2 = self.file_1, self.file_2
        with open(f'models/{file_1}/{file_1}.pkl', 'rb') as f:
            df, preprocessed, model, vectorizer, svd = pickle.load(f)
        self.df = df
        self.preprocessed = preprocessed
        self.model_1 = model
        self.vectorizer = vectorizer
        self.svd = svd
        with open(f'models/{file_2}/{file_2}.pkl', 'rb') as f:
            model = pickle.load(f)
        self.model_2 = model

    def search_filtering(self, user_input):
        self.user_input = user_input
        with st.spinner('Symptoms loading...'):
            time.sleep(2)
        df = self.df
        try:
            filtered_df = df[self.preprocessed.str.contains(user_input, case=False)]
            filtered_df = filtered_df['symptoms_preprocessed']
            all_symptoms = []
            for _, symptoms in filtered_df.items():
                symptoms_list = symptoms.split(',')
                all_symptoms.extend(symptoms_list)
            all_symptoms = list(set(all_symptoms))
            self.remove_elements(all_symptoms)
            self.multiselect(all_symptoms)
            self.diagnose_me = st.button('Diagnose me!')
        except:
            st.error("Error: Hmmm... looks like we couldn't find any symptoms matching your input. Try inputting something similar or more specific. e.g. instead of 'cold', try 'chills'.")

    def multiselect(self, symptoms):
        self.selected = []
        self.selected = st.multiselect('Did you experience any of these other symptoms?', symptoms)
        self.selected.append(self.user_input) 
        

    def diagnose(self):
        st.write("Here are possible diagnoses along with their treatments:")
        with st.spinner('Retrieving your diagnosis...'):
          time.sleep(2)
        models = [self.model_1, self.model_2]
        self.diagnoses = []
        selected_symptoms_tfidf = self.vectorizer.transform(self.selected)
        selected_symptoms_reduced = self.svd.transform(selected_symptoms_tfidf)
        for model in models:
            prediction = model.predict(selected_symptoms_reduced)
            array = np.array2string(prediction)
            array = array.replace(' ', ',')
            array = eval(array)
            self.diagnoses += array
        self.diagnoses = list(set(self.diagnoses))
        for i in range(len(self.diagnoses)):
            self.diagnoses[i] = self.diagnoses[i].replace(',', ' ')
        for diagnosis in self.diagnoses:
            with st.container(border=True):
              st.subheader(diagnosis)
              st.caption("View AI generated explanations and advice.")
              pop1 = st.expander("Do I have this disease?")
              pop1.caption("If you experienced these other symptoms, it is likely you have this issue.")
              pop2 = st.expander("What are my next steps if I have this disease?")
              prompt = f"List the symptoms and causes of {diagnosis}."
              def stream_response():
                  explanation = str(self.generate_ai(prompt))
                  for word in explanation.split(" "):
                      yield word + " "
                      time.sleep(0.01)
              pop1.write_stream(stream_response)
              prompt = f"If I think I have {diagnosis}, what are my next steps?"
              def stream_response():
                  explanation = str(self.generate_ai(prompt))
                  for word in explanation.split(" "):
                      yield word + " "
                      time.sleep(0.01)
              pop2.write_stream(stream_response)
              
        
    def remove_elements(self, list):
        list.remove(self.user_input)
        if 'family history' in list:
          list.remove('family history')
        if 'coma' in list:
          list.remove('coma')
        if 'extra marital contacts' in list:
          list.remove('extra marital contacts')

    def get_key(self):
        with open('hidden/key.txt', 'r') as file:
            key = file.read()
        return key


    def triage(self):
        with st.spinner('Triaging your symptoms...'):
          time.sleep(2)
        prompt = f"In one word, determine if the given list of symptoms is urgent or non-urgent:{self.selected}."
        triage = self.generate_ai(prompt)
        if triage.lower() == "urgent":
            self.urgent = True

    def non_urgent_suggestions(self):
        st.write("However, here are possible next steps you can take:")
        prompt = f"Assuming these following symptoms are non-urgent, list the next steps to take:{self.selected}"
        def stream_response():
          explanation = str(self.generate_ai(prompt))
          for word in explanation.split(" "):
              yield word + " "
              time.sleep(0.05)
        with st.expander("AI generated suggestions:", expanded=True):
            st.write_stream(stream_response)
            
    def generate_ai(self, prompt):
        if not self.openai_key:
            response = "You need to input your OpenAI API Key first to get accurate triaging and full diagnoses."
            return response
        client = OpenAI(
            api_key=self.openai_key,
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {
                  "role": "user",
                  "content": prompt
                },
            ],
            model="gpt-3.5-turbo-0125",
        )
        response = chat_completion.choices[0].message.content
        return response