import streamlit as st
import time #time.sleep
import pickle
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from MLFlow_Preprocess import *

#Ref: https://blog.jcharistech.com/2019/12/25/building-a-drag-drop-machine-learning-app-with-streamlit/


#@st.cache
#def read_data():
    #BASEURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"    
    #url_confirmed = f"{BASEURL}/time_series_covid19_confirmed_global.csv"
    #url_deaths = f"{BASEURL}/time_series_covid19_deaths_global.csv"
    #url_recovered = f"{BASEURL}/time_series_covid19_recovered_global.csv"

    #confirmed = pd.read_csv(url_confirmed, index_col=0)
    #deaths = pd.read_csv(url_deaths, index_col=0)
    #recovered = pd.read_csv(url_recovered, index_col=0)

    # sum over potentially duplicate rows (France and their territories)
    #confirmed = confirmed.groupby("Country/Region").sum().reset_index()
    #deaths = deaths.groupby("Country/Region").sum().reset_index()
    #recovered = recovered.groupby("Country/Region").sum().reset_index()


    #return (confirmed, deaths, recovered)

def ler_modelo_pipe_e_dados():
    prep_pipe, model = joblib.load(model_pkl)
    df = pd.read_csv(dataSetTestar, sep=';', encoding='ISO-8859-1')

    return(prep_pipe,model,df)


def main():

    mensagens_sistema = st.empty()

    st.title("Testes de modelo ML em Produção")
    st.markdown("""\
            App para validar modelos desenvolvidos com o MLFlow em produção
        """)

    tipo_analise = st.sidebar.selectbox("Escolha a opção:", ["Testar Modelo", "Explorar Dados"])

    if tipo_analise == "Testar Modelo":

        global model_pkl,dataSetTestar
        model_pkl = st.sidebar.file_uploader("1) Suba um arquivo com o modelo salvo (*.pkl)", type=("pkl"))
        dataSetTestar = st.sidebar.file_uploader("2) Suba um arquivo com o CSV a predizer o desfecho", type=["csv"])

        status_text = st.sidebar.empty()
        progress_bar = st.sidebar.progress(0)

        if st.sidebar.button("Confirmar Modelo e dados"):
            st.spinner("Processando...")

            if model_pkl is not None:
                prep_pipe, model, df = ler_modelo_pipe_e_dados()

                for i in range(1, 101):
                
                    progress_bar.progress(i)
                    time.sleep(0.03)
                    status_text.text("%i%% Complete" % i)

                st.subheader("Parâmetros do modelo treinado")
                st.write(model)

                if dataSetTestar is not None:
                    st.subheader("Visualizando os dados a testar sem pré-processar")
                
                    st.dataframe(df.head())

                    st.subheader("Visualizando os dados a testar após pré-processar")

                    dfPreProcessado = prep_pipe.transform(df)
                    st.dataframe(dfPreProcessado)

                    ynew = model.predict(dfPreProcessado)

                    ynewProb = model.predict_proba(dfPreProcessado)

                    st.write("classe predita")
                    st.write(ynew)

                    st.write("Probabilidades preditas")
                    st.write(ynewProb)

                    st.success('Modelo validado com novos dados!')
                
            else:
                st.warning('Escolha um arquivo pkl primeiro!')

        progress_bar.empty()

    else:
        st.warning('Opção em construção!')

if __name__ == "__main__":
    main()