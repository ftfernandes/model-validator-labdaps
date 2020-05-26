import streamlit as st
import time #time.sleep
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from MLFlow_Preprocess import *

#Ref: https://blog.jcharistech.com/2019/12/25/building-a-drag-drop-machine-learning-app-with-streamlit/


#Variáveis globais
model = None

st.title("Teste um modelo ML em Produção")

model_pkl = st.sidebar.file_uploader("1) Suba um arquivo com o modelo salvo (*.pkl)", type=("pkl"))

dataSetTestar = st.sidebar.file_uploader("2) Suba um arquivo com o CSV a predizer o desfecho", type=["csv"])

status_text = st.sidebar.empty()
progress_bar = st.sidebar.progress(0)

if st.sidebar.button("Confirmar Modelo e dados"):
    st.spinner("Processando...")

    for i in range(1, 101):
        
        progress_bar.progress(i)
        time.sleep(0.05)
        status_text.text("%i%% Complete" % i)

    import joblib

    prep_pipe, model = joblib.load(model_pkl)
    #scaler_pkl = joblib.load(scaler_pkl)

    st.subheader("Parâmetros do modelo treinado")
    st.write(model)
    #st.success("Finalizado!")
    if dataSetTestar is not None:
        st.subheader("Visualizando os dados a testar sem pré-processar")
        df = pd.read_csv(dataSetTestar, sep=';', encoding='ISO-8859-1')
        st.dataframe(df.head())

        st.subheader("Visualizando os dados a testar após pré-processar")

        #st.write(vars(prep_pipe))
        dfPreProcessado = prep_pipe.transform(df)
        st.dataframe(dfPreProcessado)

        ynew = model.predict(dfPreProcessado)

        ynewProb = model.predict_proba(dfPreProcessado)

        st.write("classe predita")
        st.write(ynew)

        st.write("Probabilidades preditas")
        st.write(ynewProb)



progress_bar.empty()
