# Redes Neurais de Grafos Aplicadas à Detecção de Lavagem de Dinheiro

[![Python Version](https://img.shields.io/badge/python-3.8.10-green)](https://www.python.org/downloads/release/python-3810/)
[![Virtualenv Version](https://img.shields.io/badge/virtualenv-20.0.17-green)](https://virtualenv.pypa.io/en/20.0.17/user_guide.html)


Códigos relativos à dissertação de mestrado apresentada ao Programa de Pós Graduação em Ciência da Computação, em 2023, de mesmo título.

link: http://repositorio.ufla.br/handle/1/56843
## Instalação

Para instalar o projeto, é necessário ter o [Python][1] e o [Virtualenv][2] instalados.

[1]: https://www.python.org/downloads/
[2]: https://virtualenv.pypa.io/en/latest/

Tendo as dependências instaladas, entre na pasta raiz do projeto e execute os comandos:

`virtualenv -p python39 venv`

`source venv/bin/activate`

`pip install -r requirements.txt`


## Estrutura de Diretórios

A estrutura de diretórios deste repositório está organizada como se segue:

 - `data/`: Contém todos os dados usados neste projeto, no formato do Pytorch Geometric (`.pt`)
  - `data_original/`: Contém os dados originais gerados pelo AMLSim, antes de serem convertidos para grafos.
    - `amlsim_31/`: Arquivos do conjunto AMLSim 1/3.
    - `amlsim_51/`: Arquivos do conjunto AMLSim 1/5.
    - `amlsim_101/`: Arquivos do conjunto AMLSim 1/10.
    - `amlsim_201/`: Arquivos do conjunto AMLSim 1/20.
 - `models/`: Contém todos os códigos dos modelos usados neste projeto.
    - `model_gcn.py`: Código para a Rede Convolucional de Grafos.
    - `model_skipgcn.py`: Código para a Skip-GCN.
    - `model_nenn.py`: Código para a NENN.
 - `results/`: Diretório de saída para os resultados.
 - `requirements.txt`: Arquivo para os requirementos, a ser lido pelo pip.
 - `test_gcn_amlsim.py`: Código do teste para a GCN.
 - `test_gcn_xgboost_amlsim.py`: Código do teste para GCN + XGBoost.
 - `test_skipgcn_amlsim.py`: Código do teste para a Skip-GCN. 
 - `test_skipgcn_xgboost_amlsim.py`: Código do teste para Skip-GCN + XGBoost. 
 - `test_nenn_amlsim.py`: Código do teste para a NENN. 
 - `test_nenn_xgboost_amlsim.py`: Código do teste para NENN + XGBoost.
 - `test_gcn_amlsim_tune.py`: Código do teste para a hiperparametrização da GCN.
 - `test_gcn_xgboost_amlsim_tune.py`: Código do teste para a hiperparametrização da GCN + XGBoost.
 - `test_skipgcn_amlsim_tune.py`: Código do teste para a hiperparametrização da Skip-GCN. 
 - `test_skipgcn_xgboost_amlsim_tune.py`: Código do teste para a hiperparametrização da Skip-GCN + XGBoost. 
 - `test_nenn_amlsim_tune.py`: Código do teste para a hiperparametrização da NENN. 
 - `test_nenn_xgboost_amlsim_tune.py`: Código do teste para a hiperparametrização da NENN + XGBoost.
 - `test_gcn_amlsim_softmax_xgb.py`: Código do teste para a combinação GCN + Softmax + XGBoost.
 - `test_skipgcn_amlsim_softmax_xgb.py`: Código do teste para a combinação Skip-GCN + Softmax + XGBoost. 
 - `test_nenn_amlsim_softmax_xgb.py`: Código do teste para a combinação + NENN + Softmax + XGBoost. 

## Execução

É necessário estar no ambiente virtual para executar qualquer uma das implementações:

`source venv/bin/activate`

Após isso, entre em qualquer uma das pastas e siga as instruções em seu arquivo `README.md`.
