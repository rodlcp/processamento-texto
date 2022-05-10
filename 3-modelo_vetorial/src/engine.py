from xml.dom import minidom

import numpy as np
import pandas as pd
import scipy.sparse as sp

import re
import unidecode

import configparser

import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def get_xml_field(xml_dom, field_name):
    objects = xml_dom.getElementsByTagName(field_name)
    objects_list = []

    for o in objects:
        objects_list.append(o.firstChild.data)
    return objects_list

def text_handler(text):
    # only keep alphanumeric and spaces
    res = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    
    # remove extra spaces
    res = re.sub(' +', ' ', res)
    
    # convert accented characters
    res = unidecode.unidecode(res)
    
    if res[-1] == ' ':
        res = res[:-1]
    if res[0] == ' ':
        res = res[1:]
    
    return res.upper()

def tfidf(matriz_td):
    X = matriz_td.copy()
    T, N = X.shape

    for i in range(N):
        X[:, i] /= X[:, i].sum()

    df = np.log(N / (matriz_td > 0).sum(1))
    for i in range(T):
        X[i, :] *= df[i, 0]
    return X

class ProcessadorConsultas():
    def __init__(self, config):
        logging.info('Processador de Consultas - Iniciando leitura do arquivo de configuracao')
        self.leia = config.get('pc', 'leia')
        self.consultas = config.get('pc', 'consultas')
        self.esperados = config.get('pc', 'esperados')
        logging.info('Processador de Consultas - Arquivo de configuracao lido')
        self.dom = None
    
    def le_entrada(self):
        logging.info(f'Processador de Consultas - Iniciando leitura do arquivo LEIA ({self.leia})')
        self.dom = minidom.parse(self.leia)
        
        self.query_number = get_xml_field(self.dom, 'QueryNumber')
        logging.info(f'Processador de Consultas - Lidos {len(self.query_number)} QueryNumber\'s')
        
        self.query_text = get_xml_field(self.dom, 'QueryText')
        logging.info(f'Processador de Consultas - Lidos {len(self.query_text)} QueryText\'s')
        
        self.results = get_xml_field(self.dom, 'Results')
        logging.info(f'Processador de Consultas - Lidos {len(self.results)} Results\'')
        
        self.records = self.dom.getElementsByTagName('Records')

        self.doc_number = []
        self.score = []
        k = 0

        for r in self.records:
            record_doc = []
            record_score = []

            for i in r.getElementsByTagName('Item'):
                record_doc.append(i.firstChild.data)
                record_score.append(i.getAttribute('score'))

            self.doc_number.append(record_doc)
            self.score.append(record_score)
            k += len(record_score)
        logging.info(f'Processador de Consultas - Lidos {k} Scores')
        logging.info(f'Processador de Consultas - Arquivo LEIA ({self.leia}) lido')
    
    def gera_consultas(self):
        if self.dom == None:
            self.le_entrada()
        
        logging.info(f'Processador de Consultas - Iniciando geracao do arquivo CONSULTAS ({self.consultas})')
        consultas = pd.DataFrame({
            'QueryNumber': self.query_number,
            'QueryText': self.query_text
        })

        consultas['QueryNumber'] = consultas['QueryNumber'].astype(int)
        consultas['QueryText'] = consultas['QueryText'].map(text_handler)

        consultas.sort_values('QueryNumber', inplace = True)

        consultas.to_csv(self.consultas, sep = ";", index = False)
        logging.info(f'Processador de Consultas - Arquivo CONSULTAS ({self.consultas}) gerado')
        return consultas
    
    def gera_esperados(self):
        if self.dom == None:
            self.le_entrada()
        
        logging.info(f'Processador de Consultas - Iniciando geracao do arquivo ESPERADOS ({self.esperados})')
        esperados = pd.DataFrame({
            'QueryNumber': self.query_number,
            'DocNumber': self.doc_number,
            'DocVotes': self.score
        }).explode(['DocNumber', 'DocVotes'])

        esperados['QueryNumber'] = esperados['QueryNumber'].astype(int)
        esperados['DocNumber'] = esperados['DocNumber'].astype(int)
        esperados['DocVotes'] = esperados['DocVotes'].map(lambda x: sum([int(i) for i in x]))

        esperados.sort_values(['QueryNumber', 'DocNumber'], inplace = True)
        esperados.to_csv(self.esperados, sep = ";", index = False)
        logging.info(f'Processador de Consultas - Arquivo ESPERADOS ({self.esperados}) gerado')
        return esperados
    
    def run(self):
        self.le_entrada()
        self.gera_consultas()
        self.gera_esperados()

class GeradorListaInvertida():
    def __init__(self, config):
        logging.info('Gerador de Lista Invertida - Iniciando leitura do arquivo de configuracao')
        self.leia = config.get('gli', 'leia').replace(' ', '').split(',')
        self.escreva = config.get('gli', 'escreva')
        logging.info('Gerador de Lista Invertida - Arquivo de configuracao lido')
        self.record_num = None
    
    def le(self):
        logging.info('Gerador de Lista Invertida - Iniciando leitura dos arquivos LEIA')
        self.record_num = []
        self.abstract = []

        for file in self.leia:
            logging.info(f'Gerador de Lista Invertida - Iniciando leitura do arquivo {file}')

            k1 = 0
            k2 = 0
            k3 = 0
            k4 = 0

            file_dom = minidom.parse(file)
            records = file_dom.getElementsByTagName('RECORD')
            for r in records:
                self.record_num.append(r.getElementsByTagName('RECORDNUM')[0].firstChild.data)

                # try getting abstract
                record_abstract = r.getElementsByTagName('ABSTRACT')

                if len(record_abstract) > 0:
                    self.abstract.append(record_abstract[0].firstChild.data)
                    k1 += 1
                else:
                    # if no abstract try getting extract
                    record_abstract = r.getElementsByTagName('EXTRACT')

                    if len(record_abstract) > 0:
                        self.abstract.append(record_abstract[0].firstChild.data)
                        k2 += 1
                    else:
                        # if no extract try getting title
                        record_abstract = r.getElementsByTagName('TITLE')

                        if len(record_abstract) > 0:
                            self.abstract.append(record_abstract[0].firstChild.data)
                            k3 += 1
                        else:
                            self.abstract.append('')
                            k4 += 1
            logging.info(f'Gerador de Lista Invertida - Arquivo {file} lido. Total de linhas: {k1+k2+k3+k4}')
            if k1 > 0:
                logging.info(f'Gerador de Lista Invertida - Total de Abstract\'s: {k1}')
            if k2 > 0:
                logging.info(f'Gerador de Lista Invertida - Total de Extract\'s: {k2}')
            if k3 > 0:
                logging.info(f'Gerador de Lista Invertida - Total de Title\'s: {k3}')
            if k4 > 0:
                logging.info(f'Gerador de Lista Invertida - Total de Artigos Vazios: {k4}')
        
        logging.info(f'Gerador de Lista Invertida - Inicio do processamento do texto')
        self.handled_abstract = [text_handler(i).split(' ') for i in self.abstract]
        logging.info(f'Gerador de Lista Invertida - Processamento dos textos concluido')
        logging.info(f'Gerador de Lista Invertida - Fim da leitura dos arquivos LEIA. Total de artigos: {len(self.handled_abstract)}')
    
    def escreve(self):
        if self.record_num == None:
            self.le()
        
        logging.info(f'Gerador de Lista Invertida - Iniciando geracao do arquivo ESCREVA ({self.escreva})')
        escreva = pd.DataFrame({
            'RecordNum': self.record_num,
            'Abstract': self.handled_abstract
        }).explode('Abstract')

        escreva['RecordNum'] = escreva['RecordNum'].astype(int)
        escreva = escreva.groupby('Abstract')['RecordNum'].apply(lambda x: sorted(list(x)))
        escreva = pd.DataFrame(escreva).reset_index()
        escreva.to_csv(self.escreva, sep = ';', index = False)
        logging.info(f'Gerador de Lista Invertida - Arquivo ESCREVA ({self.escreva}) gerado.')
        k = sum([len(i) for i in self.handled_abstract])
        logging.info(f'Gerador de Lista Invertida - Total de palavras: {k}')
        logging.info(f'Gerador de Lista Invertida - Palavras unicas: {escreva.shape[0]}')
        return escreva

class Indexador():
    def __init__(self, config):
        logging.info('Indexador - Iniciando leitura do arquivo de configuracao')
        self.leia = config.get('index', 'leia')
        self.escreva = config.get('index', 'escreva')
        logging.info('Indexador - Arquivo de configuracao lido')
        self.matriz_td = None
        
    def gera_matriz_td(self):
        logging.info(f'Indexador - Iniciando leitura do arquivo LEIA ({self.leia})')
        self.lista_invertida = pd.read_csv(self.leia, sep = ';', keep_default_na=False)

        isalpha = self.lista_invertida['Abstract'].str.isalpha()
        logging.info(f'Indexador - Total de palavras com digitos: {sum(1 - isalpha)}')
        self.lista_invertida = self.lista_invertida.loc[isalpha]

        small = self.lista_invertida['Abstract'].str.len() >= 2
        logging.info(f'Indexador - Total de palavras com menos de 2 caracteres: {sum(1 - small)}')
        self.lista_invertida = self.lista_invertida.loc[small]
        
        logging.info(f'Indexador - Total de palavras a serem indexadas: {self.lista_invertida.shape[0]}')
        self.lista_invertida['RecordNum'] = self.lista_invertida['RecordNum'].map(lambda x: np.array(eval(x)))
        self.lista_invertida.reset_index(drop = True, inplace = True)
        
        logging.info(f'Indexador - Gerando Matriz Termo-Documento')
        coo = self.lista_invertida\
            .explode('RecordNum').reset_index()\
            .groupby(['index', 'RecordNum']).size()\
            .reset_index().values
        
        self.matriz_td = sp.csr_matrix((coo[:, 2], (coo[:, 0], coo[:, 1] - 1)), dtype = float)
        
        self.mapping = {j: i for i, j, _ in self.lista_invertida.reset_index().values}
    
    def gera_modelo(self, metrica = tfidf):
        if self.matriz_td == None:
            self.gera_matriz_td()
        
        logging.info(f'Indexador - Gerando matriz do modelo de acordo com a metrica informada')
        matriz_modelo = metrica(self.matriz_td)
        
        logging.info(f'Indexador - Iniciando geracao do arquivo ESCREVA ({self.escreva})')
        logging.info(f'Indexador - Salvando matriz do modelo e mapeamento termo-linha da matriz')
        np.save(self.escreva, [matriz_modelo, self.mapping])
        logging.info(f'Indexador - Arquivo ESCREVA ({self.escreva}) gerado')
        return matriz_modelo

class Buscador():
    def __init__(self, config):
        logging.info('Buscador - Iniciando leitura do arquivo de configuracao')
        self.modelo = config.get('buscador', 'modelo')
        self.consultas = config.get('buscador', 'consultas')
        self.resultados = config.get('buscador', 'resultados')
        logging.info('Buscador - Arquivo de configuracao lido')
        self.matriz = None
    
    def carrega_dados(self):
        logging.info(f'Buscador - Iniciando leitura do arquivo MODELO ({self.modelo})')
        self.matriz, self.mapping = np.load(self.modelo, allow_pickle = True)
        logging.info(f'Buscador - Arquivo MODELO ({self.modelo}) lido')
        
        self.T, self.N = self.matriz.shape
        
        logging.info(f'Buscador - Iniciando leitura do arquivo CONSULTAS ({self.consultas})')
        self.consultas_df = pd.read_csv(self.consultas, sep = ";")
        self.consultas_df['QueryIDs'] = self.consultas_df['QueryText'].map(lambda x: [self.mapping[i] for i in x.split(' ') if i in self.mapping.keys()])
        logging.info(f'Buscador - Arquivo CONSULTAS ({self.consultas}) lido')
    
    def consulta(self, query_number, query_ids, metrica):
        logging.info(f'Buscador - Iniciando processamento da consulta {query_number}')
        query_ids = np.unique(query_ids)
        query = np.zeros(self.matriz.shape[0])
        query[query_ids] = 1
        
        out = np.zeros(self.N)
        for i in range(self.N):
            out[i] = metrica(self.matriz[:, i].toarray()[:, 0], query)
        logging.info(f'Buscador - Fim do processamento da consulta {query_number}')
        return out
    
    def roda_consultas(self, metrica = lambda a, b: np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)):
        if self.matriz == None:
            self.carrega_dados()
        
        logging.info(f'Buscador - Iniciando processamento de {self.consultas_df.shape[0]} consultas')
        self.scores = np.vstack([
            self.consulta(i, x, metrica)
            for i, x in self.consultas_df[['QueryNumber', 'QueryIDs']].values
        ])
        self.ranking = self.scores.argsort(axis=1)[:, ::-1]
        logging.info(f'Buscador - Fim do processamento das consultas')
        
        logging.info(f'Buscador - Iniciando geracao do arquivo RESULTADOS ({self.resultados})')
        out = self.consultas_df[['QueryNumber']].copy()
        out['Result'] = [[[j + 1, e, self.scores[i, j]] for j, e in enumerate(r)] for i, r in enumerate(self.ranking)]
        out.to_csv(self.resultados, sep = ";", index = False)
        logging.info(f'Buscador - Arquivo RESULTADOS ({self.resultados}) gerado')
        return out