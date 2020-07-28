from transformers import pipeline
import wikipedia

# QA pipeline
nlp = pipeline('question-answering');

# Retrieve context
subject = 'covid19'
page = wikipedia.page(subject)
context = f"{page.content}".replace('\n',' ')[:1000]

for question in [
    "what is COVID-19?",
    "when did it start?",
    "where did it start?",
    "what are the symptoms?",
    "is there a treatment?"
]:
    try:
        output = nlp({
            'question': question,
            'context': context
        })
        print(f">> {question}\n<< {output['answer']}")
    except:
        print(f">> {question}\n<< (no answer)")
