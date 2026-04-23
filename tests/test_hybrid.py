def test_hybrid_extractor(extractor, sample_texts):
    text = sample_texts['full']
    response = extractor.extract(text)
    assert isinstance(response, dict)
    expected_keys = {'PATIENT', 'DISEASE', 'DOCTOR', 'DRUG'}
    assert any(key in expected_keys for key in response), 'No expected entity types found'
