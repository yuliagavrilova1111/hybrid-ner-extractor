import spacy

class ClassicalNER:
    def __init__(self):
        self.nlp = spacy.load('ru_core_news_lg')

    def extract(self, text: str) -> list:
        doc = self.nlp(text)
        return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]