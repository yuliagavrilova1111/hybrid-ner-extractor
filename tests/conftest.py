import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.hybrid_extractor import HybridExtractor

@pytest.fixture(scope='session')
def extractor():
    return HybridExtractor()

@pytest.fixture
def client(extractor):
    with TestClient(app) as client:
        app.state.extractor = extractor
        yield client

@pytest.fixture
def sample_texts():
    """
    набор тестовых текстов для разных сценариев:
        simple — короткий обычный текст.
        full — текст, содержащий все типы сущностей.
        empty — пустая строка (для проверки валидации).
        long — длинный текст (для проверки ограничения по длине).
    """
    return {
        'simple': 'Пациент Иванов жалуется на кашель. Врач прописал Амоксициллин.',
        'full': 'Доктор Смирнова назначила Петру Сидорову лечение от гриппа препаратом Нурофен.',
        'empty': ''
    }