from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import spacy
import numpy as np
import wikipedia

# NLP pipelines
nlp_sp = spacy.load('en')
# QA pipeline
nlp_qa = pipeline('question-answering')

SIM_THRESHOLD = 0.01

vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=1,
    max_df=.75,
    ngram_range=(1,3))

def lemmatize(text):
    return " ".join([word.lemma_ for word in nlp_sp(text)])

# Retrieve text
subject = 'covid19'
results = wikipedia.search(subject)
page = wikipedia.page(results[0])
corpus = page.content.split('\n')

lemmas = [lemmatize(paragraph) for paragraph in corpus]
corpus_tfidf = vectorizer.fit_transform(lemmas)

for question in [
    "what is COVID-19?",
    "when did it start?",
    "where did it start?",
    "what are the symptoms?",
    "is there a treatment?"
]:
    question_lemma = vectorizer.transform([lemmatize(question)])
    arr_similarity = (corpus_tfidf * question_lemma.T).toarray()

    arr_scores = []
    arr_answers = []

    for ix,paragraph in enumerate(corpus):

        if arr_similarity[ix] < SIM_THRESHOLD:
            continue

        context = paragraph[:1000]
        try:
            output = nlp_qa({
                'question': question,
                'context': context
            })
            arr_scores.append(output['score'])
            arr_answers.append(output['answer'])
        except:
            continue

    arr_scores = np.array(arr_scores)
    arr_answers = np.array(arr_answers)
    if len(arr_scores):
        print(f">> {question}\n<< {arr_answers[np.argmax(arr_scores)]}")
        for score,answer in zip(arr_scores,arr_answers):
            print(f"<< ({score:.3f},{answer})")
    else:
        print(f">> {question}\n<< (no answer)")
