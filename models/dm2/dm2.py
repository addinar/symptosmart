from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from datasets import load_dataset
import pickle

data = load_dataset("shanover/disease_symptoms_prec_full")
df = pd.DataFrame(data)

diseases = []
symptoms = []
for item in df['train']:
    diseases.append(item['disease'])
    symptoms.append(item['symptoms'])
df = pd.DataFrame({'disease': diseases, 'symptoms': symptoms})


def preprocesser(symptoms):
    symptoms_list = symptoms.split(",")
    for i, symptom in enumerate(symptoms_list):
        symptoms_list[i] = symptom.replace("_", " ")
    return ','.join(symptoms_list)


df['symptoms_preprocessed'] = df['symptoms'].apply(preprocesser)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['symptoms_preprocessed'])

svd = TruncatedSVD(n_components=100)
X = svd.fit_transform(X)

model = RandomForestClassifier(n_estimators=200, max_features='log2')
y = df['disease']
model.fit(X, y)

with open('dm2.pkl', 'wb') as f:
    pickle.dump(model, f)