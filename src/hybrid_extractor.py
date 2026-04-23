from .classical_ner import ClassicalNER
from .llm_extractor import LLMExtractor

class HybridExtractor:
    def __init__(self):
        self.classical_ner_extr = ClassicalNER()
        self.llm_extr = LLMExtractor()
    def extract(self, text: str) -> dict:
        list_spacy = self.classical_ner_extr.extract(text)
        llm_res = self.llm_extr.extract(text)
        spacy_res = {}
        for x in list_spacy:
            ent_text = x['text']
            ent_label = x['label']
            spacy_res[ent_label] = spacy_res.get(ent_label, []) + [ent_text]
        
        for key, values in llm_res.items():
            if key in spacy_res:
                spacy_res[key].extend([v for v in values if v not in spacy_res[key]])
            else:
                spacy_res[key] = values
        
        return spacy_res
    
if __name__ == '__main__':
    extractor = HybridExtractor()
    test_texts = [
        'Пациент Иванов жалуется на кашель. Врач прописал Амоксициллин.',
        'Доктор Смирнова назначила Петру Сидорову лечение от гриппа препаратом Нурофен.',
        'Больная Анна Петрова страдает от аллергии. Врач рекомендовал Зиртек.'
    ]
    for text in test_texts:
        print(f'Текст: {text}')
        result = extractor.extract(text)
        print(f'Сущности: {result}\n')