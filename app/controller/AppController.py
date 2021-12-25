from flask import request, jsonify
from app import app
from app.module.Engine import preprocessing, splitParagraphIntoSentences, LSA, CLSA, summary_sentence, sum_frame_by_column
import pandas as pd
import os
from flask import render_template

from sklearn.feature_extraction.text import CountVectorizer # tf-idf
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer # tf-idf
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
import pandas as pd
import scipy as sp
import numpy as np
from scipy.linalg import svd 
from numpy import dot
import numpy

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/proses', methods=['POST','GET'])
def proses():
    kalimat = str(request.form['kalimat'])
    params_angka = request.form['input_kalimat']

    pemisal_kalimat = splitParagraphIntoSentences(kalimat)
    simpan_sementara_isi_berita = list()
    berita_asli = list()
    for per_kalimat in pemisal_kalimat:
        simpan_sementara_isi_berita.append(preprocessing(per_kalimat.strip()))
        berita_asli.append(per_kalimat.strip())

    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords,smooth_idf=False, norm=None)
    X = tfidf_vectorizer.fit_transform(simpan_sementara_isi_berita)
    return_TFIDF  = pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names()).T

    return_LSA = pd.DataFrame(LSA(X)).T
    return_CLSA = pd.DataFrame(CLSA(X)).T

    rank_LSA = sum_frame_by_column(return_LSA, 'total_score_document', [i for i in range(len(return_CLSA[0]))])
    rank_CLSA = sum_frame_by_column(return_CLSA, 'total_score_document', [i for i in range(len(return_CLSA[0]))])

    docs = [str(x) for x in simpan_sementara_isi_berita]
    documentNames = list()
    for i,simpan_sementara_isi_berita in enumerate(docs):
        documentNames.append("Document_{}".format(i+1))
    
    return_LSA['documentNames'] = documentNames
    return_LSA['rank'] = return_LSA['total_score_document'].rank(method='first', ascending=False).astype(int)
    return_CLSA['documentNames'] = documentNames
    return_CLSA['rank'] = return_CLSA['total_score_document'].rank(method='first', ascending=False).astype(int)

    aftersort_LSA = rank_LSA.sort_values(['total_score_document'], ascending=[False])
    aftersort_LSA['rank'] = range(1, len(aftersort_LSA) + 1)
    aftersort_CLSA = rank_CLSA.sort_values(['total_score_document'], ascending=[False])
    aftersort_CLSA['rank'] = range(1, len(aftersort_CLSA) + 1)

    sentences_lsa = summary_sentence(berita_asli, X, params_angka, types='lsa')
    sentences_clsa = summary_sentence(berita_asli, X, params_angka, types='clsa')
    
    return render_template('proses.html', 
        sebelum_preprocessing=request.form['kalimat'] ,
        sesudah_preprocessing=simpan_sementara_isi_berita ,
        tables_TFIDF=[return_TFIDF.to_html()],
        tables_LSA = [return_LSA.to_html()] ,
        tables_CLSA = [return_CLSA.to_html()],
        sum_tables_LSA = [aftersort_LSA.to_html()],
        sum_tables_CLSA = [aftersort_CLSA.to_html()],
        sentences_lsa = sentences_lsa,
        sentences_clsa = sentences_clsa
    )
