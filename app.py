# Author: Fernando T. Fernandes <ftfernandes@usp.br>
# License: MIT

import pickle
import time  # time.sleep
import urllib.request

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler

from MLFlow_Preprocess import *

def read_saved_final_model():
    BASEURL = "https://github.com/ftfernandes/model-validator-labdaps/raw/master"    
    xgb_model_pkl = f"{BASEURL}/saved_models/final_xgb.pkl"

    prep_pipe, model = joblib.load(urllib.request.urlopen(xgb_model_pkl))
    return(prep_pipe,model)

@st.cache
def read_logo():
    BASEURL = "https://github.com/ftfernandes/model-validator-labdaps/raw/master" 
    logoLab = f"{BASEURL}/images/LabdapsLogo.jpg"

    from PIL import Image
    image = Image.open(urllib.request.urlopen(logoLab))

    #image = image.resize(387,122)

    return(image)

plt.style.use('bmh')

def read_external_csv():
    df = pd.read_csv(arquivoCSVExplorar, sep=';', encoding='ISO-8859-1')

    return(df)

def read_model_and_pipeline():
    prep_pipe, model = joblib.load(model_pkl)
    df = pd.read_csv(dataSetTestar, sep=';', encoding='ISO-8859-1')

    return(prep_pipe,model,df)

def main():

    mensagens_sistema = st.empty()

    st.image(read_logo())

    st.title("Diagnóstico de Diabetes com Machine Learning")
    
    tipo_analise = st.selectbox("Escolha a opção:", ["Calcular Risco" ,"Explorar Dados","Testar diferentes modelos"])

    if tipo_analise == "Testar diferentes modelos":

        global model_pkl,dataSetTestar
        model_pkl = st.sidebar.file_uploader("1) Suba um arquivo com o modelo salvo no MLFlow (*.pkl)", type=("pkl"))
        dataSetTestar = st.sidebar.file_uploader("2) Suba um arquivo com o CSV a predizer o desfecho", type=["csv"])

        status_text = st.empty()
        progress_bar = st.progress(0)

        if st.button("Confirmar Modelo e dados"):
            st.spinner("Processando...")

            if model_pkl is not None:
                prep_pipe, model, df = read_model_and_pipeline()

                for i in range(1, 101):
                
                    progress_bar.progress(i)
                    time.sleep(0.03)
                    status_text.text("%i%% Complete" % i)
                

                if dataSetTestar is not None:
                    st.subheader("Visualizando os dados a testar sem pré-processar")
                
                    st.table(df.head())

                    st.subheader("Visualizando os dados a testar após pré-processar")

                    dfPreProcessado = prep_pipe.transform(df)
                    st.table(dfPreProcessado)

                    ynew = model.predict(dfPreProcessado)

                    ynewProb = model.predict_proba(dfPreProcessado)

                    st.write("classe predita")
                    st.table(ynew)

                    st.write("Probabilidades preditas")
                    st.table(ynewProb)

                    st.subheader("Parâmetros do modelo treinado")
                    st.write(model)
                
            else:
                st.warning('Escolha um arquivo pkl primeiro!')

        progress_bar.empty()

    elif tipo_analise == "Explorar Dados":

        global arquivoCSVExplorar
        arquivoCSVExplorar = st.sidebar.file_uploader("Escolha um arquivo CSV", type=["csv"])

        if st.button("Confirmar"):

            if arquivoCSVExplorar is not None:

                with st.spinner('Processando. Aguarde...'):
                    df = read_external_csv()
                    st.table(df.head())

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

        tipo_predicao = st.radio("Tipo de predição",("Individual","Arquivo CSV"))

        exibirDetalhes = st.checkbox('Exibir detalhes da amostra')

        if tipo_predicao == 'Individual':

            preg = st.sidebar.number_input('Número de Gravidez:', value=0, min_value=0, max_value=10,step=1)

            plas = st.sidebar.slider('Plasma', 0, 199, 99)

            pres  = st.sidebar.slider('Pressão Sanguínea', 0, 122, 70)

            skin  = st.sidebar.slider('Prega Tríceps', 0, 99, 10)

            test  = st.sidebar.slider('Insulina', 0, 846, 85)

            mass  = st.sidebar.slider('IMC', 0, 67, 26)

            pedi  = st.sidebar.slider('Diabetes Pedigree Function (dpf)', 0.08, 2.42, 0.30)

            age  = st.sidebar.slider('Idade', 0, 45, 27)

            if st.button("Confirmar"):

                instrucao = st.empty()

                with st.spinner('Processando. Aguarde...'):

                    prep_pipe, model = read_saved_final_model()
                    #cria nova obs
                    df = pd.DataFrame(columns=['preg','plas','pres','skin','test','mass','pedi','age'])
                    df.loc[0] = [preg,plas,pres,skin,test,mass,pedi,age]
                   
                    dfPreProcessado = prep_pipe.transform(df)

                    st.header("Diagnóstico estimado")

                    ynew = model.predict(dfPreProcessado)
                    ynewProb = model.predict_proba(dfPreProcessado)
                    pctRisco = np.around(ynewProb[:,1].item() * 100,2)

                    textoNegativo = "Risco baixo. Probabilidade de desenvolver diabetes: **" + str(pctRisco) + "%**"
                    textoPositivo = "Risco alto. Probabilidade de desenvolver diabetes: **" + str(pctRisco) + "%**"
                    
                    if ynew==1:
                        st.error(textoPositivo)
                    else:
                        st.success(textoNegativo)

                    if exibirDetalhes:

                        st.header("Detalhes da amostra")
                    
                        st.subheader("Observação original")
                        st.table(df)

                        st.subheader("Observação normalizada")
                        st.table(dfPreProcessado)

                        st.subheader("Probabilidades")
                        st.table(ynewProb)

                st.info("""\
                        
                    by: [Labdaps](https://labdaps.github.io/) | source: [GitHub](https://github.com/ftfernandes/model-validator-labdaps)
                    | data source: [PIMA Dataset (Kaggle)](https://www.kaggle.com/uciml/pima-indians-diabetes-database). 
                """)
        else:
            
            dataSetTestar = st.sidebar.file_uploader("Escolha um arquivo CSV para realizar a predição (sem o desfecho)", type=["csv"])

            if st.button("Confirmar"):
            
                if dataSetTestar is not None:

                    with st.spinner('Processando. Aguarde...'):

                        prep_pipe, model = read_saved_final_model()
                        df = pd.read_csv(dataSetTestar, sep=';', encoding='ISO-8859-1')
                        dfPreProcessado = prep_pipe.transform(df)

                        st.header("Diagnóstico estimado")
                        ynew = model.predict(dfPreProcessado)
                        ynewProb = model.predict_proba(dfPreProcessado)
                        st.table(ynewProb)

                        #pctRisco = np.around(ynewProb[:,1].item() * 100,2)

                        #textoPositivo = "Risco alto. Probabilidade de desenvolver diabetes: **" + str(pctRisco) + "%**"
                        if exibirDetalhes:

                            st.header("Detalhes da Amostra")

                            st.subheader("Dados originais")
                            st.table(df.head())

                            st.subheader("Dados normalizados")
                            st.table(dfPreProcessado)
                            
                            st.subheader("Probabilidade de desenvolver diabetes")
                            st.table(ynewProb)

                    st.info("""\
                        
                    by: [Labdaps](https://labdaps.github.io/) | source: [GitHub](https://github.com/ftfernandes/model-validator-labdaps)
                    | data source: [PIMA Dataset (Kaggle)](https://www.kaggle.com/uciml/pima-indians-diabetes-database). 
                """)

                else:
                    st.warning("Escolhar um arquivo CSV!")


if __name__ == "__main__":
    main()
