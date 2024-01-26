import streamlit as st
import pandas as pd
from joblib import load

# Carregar o modelo, o dicionário de LabelEncoders e o StandardScaler
model = load('modelo_rf.joblib')
label_encoders = load('label_encoders.joblib')
scaler = load('standard_scaler.joblib')

# Função para preparar os dados de entrada
def prepare_input(data, original_columns):
    # Remover colunas que não estão no conjunto de dados original
    data = data[original_columns]

    # Tratamento de valores ausentes e aplicação do LabelEncoder e StandardScaler
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].fillna('Desconhecido')
            data[column] = label_encoders[column].transform(data[column])
        else:
            data[column] = data[column].fillna(data[column].median())

    data_scaled = scaler.transform(data)
    return data_scaled

# Função para realizar a transformação inversa das previsões
def inverse_transform_predictions(predictions, label_encoder):
    return label_encoder.inverse_transform(predictions)

# Interface Streamlit
st.title("Previsão de Abandono do Ensino Superior")

# Lista de colunas do conjunto de dados original (excluindo 'Abandono?')
original_columns = ['Desc. nacionalidade', 'A/S curricular aluno', 'Sexo (M/F)', 'Desc. tipos aluno', 
                    'Cód. curso', 'Nome curso', 'Área Científica', 'Empregabilidade', 
                    'Ano lectivo ingresso', 'Candidatos - Formade Ingresso 1.º ano 1.ª vez', 
                    'Nota de Ingresso', 'Ano Nascimento', 'Idade neste ano letivo', 
                    'Ano conclusão habilitação anterior', 'Classificação habilitação anterior', 
                    'Desc. habilitação anterior', 'Desc. regime frequência', 'Bolsa DGES', 
                    'Concelho', 'Km Distância ISEC Lisboa - Concelho Morada Indicada pelo estudante', 
                    'Desc. habilitação literária mãe', 'Desc. habilitação literária pai', 
                    'Desc. profissão mãe', 'Desc. profissão pai', 'Desc. situação profissional mãe', 
                    'Desc. situação profissional pai', 'Desc. situação profissional (aluno)', 
                    'Desc. grupo profissional (aluno)', 'ECTS (aprovados)', 'ECTS (inscrito)', 
                    '% de ECTS feitos face inscritos', 'Média UC Realizadas', 
                    'Diplomou-se até 2022 (Curso)', 'Diplomou-se até 2022 (Data)']

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type=["csv"])
if uploaded_file is not None:
    # Ler o arquivo CSV
    input_data = pd.read_csv(uploaded_file)

    # Armazenar os números dos alunos, se a coluna 'Nº' estiver presente
    student_numbers = input_data.get('Nº', pd.Series())

    # Preparar os dados para o modelo
    input_data_prepared = prepare_input(input_data, original_columns)

    # Botão para realizar a previsão
    if st.button('Prever Abandono'):
        # Realizar a previsão
        prediction = model.predict(input_data_prepared)
        
        # Transformação inversa das previsões para obter rótulos originais
        le = label_encoders['Abandono?']  # Substituir pelo nome da coluna alvo usada no treinamento
        original_labels = inverse_transform_predictions(prediction, le)
        
        # Mostrar os resultados
        results_df = pd.DataFrame({
            'Nº do Aluno': student_numbers,
            'Previsão Numérica': prediction,
            'Previsão': original_labels
        })
        st.write("Tabela de Previsões de Abandono:")
        st.table(results_df)

# Executar o Streamlit usando `streamlit run app.py` no terminal
