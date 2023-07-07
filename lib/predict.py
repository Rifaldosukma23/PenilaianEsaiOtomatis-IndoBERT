import torch
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import stanza
import spacy
import spacy_stanza

stanza.download('id')
nlp = spacy_stanza.load_pipeline("id")

nltk.download('punkt')
nltk.download('stopwords')

tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")

def case_folding(text):
    text = text.lower()
    return text

def replace_abbreviations(text):
    abbreviation_dict = {"p": "panjang", "l": "lebar", "x": "kali", "*": "kali", "p*l": "panjang kali lebar", "=": "sama dengan"}
    words = text.split()
    for i in range(len(words)):
        word = words[i]
        if word in abbreviation_dict:
            words[i] = abbreviation_dict[word]
    text = ' '.join(words)
    return text

def cleaning(text):
    if isinstance(text, str) and text.strip():
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', " ", text)
    else:
        text = ''

    return text

def tokenization(text):
    text = nltk.word_tokenize(text)
    return text

def pos_dep(text):
    doc = nlp(case_folding(text))
    pred = ''
    for token in doc:
        pred += '{} <{}> [{}] '.format(token.text, token.pos_, token.dep_)
    return pred

def stopword_removal(text):
    listStopwords = set(stopwords.words('indonesian'))
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text

def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(token) for token in text]
    return text

def to_sentence(list_words):
    sentence = ' '.join(word for word in list_words)
    return sentence

def encode_text(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_state = outputs.last_hidden_state
    return last_hidden_state.squeeze().mean(dim=0).detach().numpy()

def calculate_score(question, kunci_jawaban, jawaban_siswa):
    constanta = 1

    question = case_folding(question)
    question = replace_abbreviations(question)
    question = pos_dep(question)
    question = tokenization(question)
    question = stopword_removal(question)
    question = stemming(question)
    question = to_sentence(question)

    jawaban_siswa = cleaning(jawaban_siswa)
    jawaban_siswa = case_folding(jawaban_siswa)
    jawaban_siswa = replace_abbreviations(jawaban_siswa)
    jawaban_siswa = tokenization(jawaban_siswa)
    jawaban_siswa = stopword_removal(jawaban_siswa)
    jawaban_siswa = stemming(jawaban_siswa)
    jawaban_siswa = to_sentence(jawaban_siswa)

    kunci_jawaban = cleaning(kunci_jawaban)
    kunci_jawaban = case_folding(kunci_jawaban)
    kunci_jawaban = replace_abbreviations(kunci_jawaban)
    kunci_jawaban = tokenization(kunci_jawaban)
    kunci_jawaban = stopword_removal(kunci_jawaban)
    kunci_jawaban = stemming(kunci_jawaban)
    kunci_jawaban = to_sentence(kunci_jawaban)

    vektor_siswa = encode_text(jawaban_siswa)
    vektor_kunci = encode_text(kunci_jawaban)

    cosine_value = cosine_similarity(vektor_siswa.reshape(1, -1), vektor_kunci.reshape(1, -1))[0][0]

    qpos = ['VERB', 'NOUN']
    pos_analysis = pos_dep(question)

    if re.search("urut", question) is not None:
        jawaban_siswa_arr = jawaban_siswa.split(" ")
        kunci_jawaban_arr = kunci_jawaban.split(" ")

        if len(jawaban_siswa_arr) != len(kunci_jawaban_arr):
            constanta = 0.5
        else:
            if all(x == y for x, y in zip(jawaban_siswa_arr, kunci_jawaban_arr)):
                constanta = 1
            else:
                constanta = 0.5

    final_score = cosine_value * constanta
    result = final_score * 4

    if result < 1:
        result = 0
    elif 1 <= result < 1.5:
        result = 25
    elif 1.5 <= result < 2.5:
        result = 50
    elif 2.5 <= result < 3.5:
        result = 75
    else:
        result = 100

    return result
