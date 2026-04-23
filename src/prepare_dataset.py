import os
import json
import re
from yandex_ai_studio_sdk import AIStudio
from dotenv import load_dotenv

load_dotenv()

YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')

def call_yandex_gpt(prompt: str, temperature: float = 0.7) -> str:
    sdk = AIStudio(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_API_KEY)
    model = sdk.models.completions('yandexgpt-lite')
    model = model.configure(temperature=temperature, max_tokens=3000)

    result = model.run([
        {'role': 'system', 'text': 'Ты — эксперт по созданию синтетических данных для NLP.'},
        {'role': 'user', 'text': prompt}
    ])

    return result[0].text

def generate_samples(num_samples: int = 10) -> list:
    prompt = f"""Сгенерируй {num_samples} примеров для обучения модели извлечению сущностей. Формат каждого примера — JSON-объект с двумя полями: "text" и "entities".

Типы сущностей (строго соблюдай):
- PATIENT: имя и фамилия пациента (полное, например, "Иван Петров").
- DOCTOR: имя и фамилия врача (полное, например, "Анна Смирнова").
- DRUG: конкретное лекарство (торговое название, например, "Амоксициллин", "Нурофен").
- DISEASE: конкретное заболевание или симптом (например, "грипп", "головная боль").

Важные требования к каждому примеру:
1. Текст должен быть на русском языке, длиной 15–40 слов.
2. Каждый пример должен содержать ВСЕ ЧЕТЫРЕ типа сущностей (PATIENT, DOCTOR, DRUG, DISEASE).
3. Сущности должны быть именно в том виде, как они встречаются в тексте.
4. Не используй общие слова: вместо "врач" пиши "доктор Анна Смирнова", вместо "пациент" — "Иван Петров", вместо "лекарство" — конкретное название.
5. Разнообразь ситуации: разные болезни, лекарства, имена, стили речи.

Пример правильного ответа (в виде списка из одного элемента):
[
  {{
    "text": "Пациент Иван Петров обратился к доктору Анне Смирновой с жалобами на сильную головную боль. Врач выписал ему Нурофен.",
    "entities": {{"PATIENT": ["Иван Петров"], "DOCTOR": ["Анна Смирнова"], "DISEASE": ["головная боль"], "DRUG": ["Нурофен"]}}
  }}
]

Важно:
- Ответ должен быть JSON-списком из {num_samples} объектов.
- Сущности должны быть именно в том виде, как они встречаются в тексте.
- Не добавляй лишний текст до или после JSON. Только чистый JSON.
"""
    response = call_yandex_gpt(prompt)
    try:
        samples = json.loads(response)
        if isinstance(samples, list):
            return samples
        raise ValueError('Ответ не является списком')
    except json.JSONDecodeError:
        match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            print('Не удалось распарсить JSON. Ответ модели:')
            print(response)
            return []

def generate_many_samples(total: int = 600, batch_size: int = 10):
    all_samples = []
    batch_num = 1
    while len(all_samples) < total:
        remaining = total - len(all_samples)
        curr_batch = min(batch_size, remaining)
        print(f'Батч {batch_num}: генерация {curr_batch} примеров...')
        batch = generate_samples(num_samples=curr_batch)

        if not batch:
            break
        
        all_samples.extend(batch)
        batch_num += 1
    
    return all_samples


def save_to_jsonl(samples: list, output_path: str):
    """Сохраняет список примеров в JSONL файл с полями prompt и completion."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            text = sample['text']
            entities = sample['entities']
            prompt = f'Извлеки сущности из текста и верни результат в формате JSON.\nТекст: {text}'
            completion = json.dumps(entities, ensure_ascii=False)
            full_text = f'{prompt}<|endoftext|>{completion}'
            line = json.dumps({'text': full_text}, ensure_ascii=False)
            f.write(line+'\n')

def main():
    print('Генерация синтетических данных через YandexGPT...')
    samples = generate_many_samples(total=600, batch_size=10)
    if not samples:
        print('Генерация не удалась. Проверь API ключ и интернет.')
        return
    output_path = 'data/processed/train.jsonl'
    save_to_jsonl(samples, output_path)
    print(f'Сохранено {len(samples)} примеров в {output_path}')

    val_samples = generate_many_samples(total=50, batch_size=10)
    if val_samples:
        output_path_val = 'data/processed/val.jsonl'
        save_to_jsonl(val_samples, output_path_val)
        print(f'Сохранено {len(val_samples)} примеров в {output_path_val}')

if __name__ == '__main__':
    main()