# Redes Neurais de Grafos Aplicadas à Detecção de Lavagem de Dinheiro

Projeto em execução. Por hora, foi implementada a metodologia do artigo ["Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"][1].

A base de dados utilizada é a [Elliptico][2].

Os testes feitos com a GCN simples descrita no artigo, implementada com o Pytorch Geometric estão no arquivo `gcn_crua_testes.ipynb`.
A mesma implementação, porém implementada somente com o Pytorch, está no arquivo `gcn_kaggle_testes.ipynb`, encontrada no Kaggle [[3]].
A implementação da _Skip_ GCN está em `skip_gcn_tests.ipynb`.

[2]: https://www.kaggle.com/ellipticco/elliptic-data-set
[1]: https://arxiv.org/pdf/1908.02591.pdf
[3]: https://www.kaggle.com/karthikapv/gcn-elliptic-dataset 

