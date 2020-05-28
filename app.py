import pickle
import time  # time.sleep

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
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

plt.style.use('bmh')

def ler_CSV_explorar():
    df = pd.read_csv(arquivoCSVExplorar, sep=';', encoding='ISO-8859-1')

    return(df)

def ler_modelo_pipe_e_dados():
    prep_pipe, model = joblib.load(model_pkl)
    df = pd.read_csv(dataSetTestar, sep=';', encoding='ISO-8859-1')

    return(prep_pipe,model,df)


def main():

    mensagens_sistema = st.empty()

    st.title("Bem-vindo ao explorador de dados")
    
    tipo_analise = st.sidebar.selectbox("Escolha a opção:", ["Calcular Risco" ,"Explorar Dados","Testar diferentes modelos"])

    if tipo_analise == "Testar diferentes modelos":

        global model_pkl,dataSetTestar
        model_pkl = st.file_uploader("1) Suba um arquivo com o modelo salvo (*.pkl)", type=("pkl"))
        dataSetTestar = st.file_uploader("2) Suba um arquivo com o CSV a predizer o desfecho", type=["csv"])

        status_text = st.empty()
        progress_bar = st.progress(0)

        if st.button("Confirmar Modelo e dados"):
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

    elif tipo_analise == "Explorar Dados":

        global arquivoCSVExplorar
        arquivoCSVExplorar = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

        if st.button("Confirmar"):

            if arquivoCSVExplorar is not None:

                with st.spinner('Processando. Aguarde...'):
                    df = ler_CSV_explorar()
                    st.write(df.head())


                    st.subheader("Histogramas")

                    df_num = df.select_dtypes(include = ['float64', 'int64'])
                    df_num.hist(figsize=(20, 20), bins=50, xlabelsize=8, ylabelsize=8)

                    st.pyplot()

                    st.subheader("Correlação de variáveis")

                    corr = df_num.dropna().corr()
                    mask = np.zeros_like(corr)
                    mask[np.triu_indices_from(mask)] = True

                    sns.heatmap(corr, cmap=sns.diverging_palette(256,0,sep=80,n=7,as_cmap=True), annot=True, mask=mask)

                    st.pyplot()
                
                st.success('Finalizado')

            else:
                st.warning('Escolha um arquivo csv')

    else:
        st.subheader("Preencha os campos ao lado para realizar uma estimativa")
        
        preg = st.sidebar.number_input('Número de Gravidez:', value=0, min_value=0, max_value=10,step=1)

        plasma = st.sidebar.slider('Plasma', 0, 199, 25)

        pres  = st.sidebar.slider('Pressão Sanguínea', 0, 122, 70)

        skin  = st.sidebar.slider('Skin Thickness', 0, 99, 20)

        test  = st.sidebar.slider('Insulina', 0, 122, 70)

        mass  = st.sidebar.slider('BMI', 0, 122, 70)

        pedi  = st.sidebar.slider('Diabetes Pedigree', 0, 122, 70)

        age  = st.sidebar.slider('Idade', 0, 100, 30)

        if st.sidebar.button("Confirmar"):
            st.warning("Em construção")


        #st.inp
        
        

if __name__ == "__main__":
    main()
