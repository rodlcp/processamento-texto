import configparser
from engine import ProcessadorConsultas, GeradorListaInvertida, Indexador, Buscador
import numpy as np

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

config = configparser.ConfigParser()
config.read('conf.ini');

pc = ProcessadorConsultas(config)
pc.run()

gli = GeradorListaInvertida(config)
gli.escreve()

def tfidf(matriz_td):
    X = matriz_td.copy()
    T, N = X.shape

    for i in range(N):
        X[:, i] /= X[:, i].sum()

    df = np.log(N / (matriz_td > 0).sum(1))
    for i in range(T):
        X[i, :] *= df[i, 0]
    return X

index = Indexador(config)
# n eh necessario sempre passar a metrica, apenas mostro como exemplo para definir metrica propria
index.gera_modelo(metrica = tfidf); 

buscador = Buscador(config)
buscador.roda_consultas();