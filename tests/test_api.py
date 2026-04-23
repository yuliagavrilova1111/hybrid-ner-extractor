def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}

def test_extract_entities(client, sample_texts):
    response = client.post('/extract', json={'text': sample_texts['simple']})
    assert response.status_code == 200
    data = response.json()
    assert 'response' in data
    assert isinstance(data['response'], dict)
    assert len(data['response']) > 0

def test_extract_empty(client, sample_texts):
    response = client.post('/extract', json={'text': sample_texts['empty']})
    assert response.status_code == 422 # ошибка валидации, потому что min_length=1