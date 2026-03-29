"""
tests/test_app.py
Pytest suite covering:
  - /health endpoint
  - /predict happy path (survive / not survive)
  - /predict validation errors
  - /predict edge cases & missing fields
  - Feature engineering helper
"""

import json
import pytest
import sys
import os

# Ensure app is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app, prepare_input, validate_input


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


# Canonical passenger payloads
SURVIVED_PAYLOAD = {
    "Age": 28, "Fare": 75.0, "SibSp": 0, "Parch": 0,
    "Sex": "female", "Embarked": "C", "Title": "Miss", "Pclass": 1
}

PERISHED_PAYLOAD = {
    "Age": 35, "Fare": 7.5, "SibSp": 0, "Parch": 0,
    "Sex": "male", "Embarked": "S", "Title": "Mr", "Pclass": 3
}


# ── Health check ──────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get('/health')
        assert r.status_code == 200

    def test_health_json_schema(self, client):
        r    = client.get('/health')
        body = json.loads(r.data)
        assert 'status' in body
        assert 'model_loaded' in body
        assert body['status'] == 'ok'


# ── Index page ────────────────────────────────────────────────────────────────
class TestIndex:
    def test_index_returns_html(self, client):
        r = client.get('/')
        assert r.status_code == 200
        assert b'Titanic' in r.data

    def test_index_content_type(self, client):
        r = client.get('/')
        assert 'text/html' in r.content_type


# ── /predict — happy path ─────────────────────────────────────────────────────
class TestPredict:
    def _post(self, client, payload):
        return client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )

    def test_predict_survived_status_200(self, client):
        r = self._post(client, SURVIVED_PAYLOAD)
        assert r.status_code in (200, 503)  # 503 if no model in CI

    def test_predict_response_schema(self, client):
        r    = self._post(client, SURVIVED_PAYLOAD)
        if r.status_code == 503:
            pytest.skip("Model not loaded in this environment")
        body = json.loads(r.data)
        assert 'survived'    in body
        assert 'probability' in body
        assert 'confidence'  in body
        assert 'message'     in body

    def test_predict_survived_is_binary(self, client):
        r = self._post(client, SURVIVED_PAYLOAD)
        if r.status_code == 503:
            pytest.skip("Model not loaded")
        body = json.loads(r.data)
        assert body['survived'] in (0, 1)

    def test_predict_probability_in_range(self, client):
        r = self._post(client, SURVIVED_PAYLOAD)
        if r.status_code == 503:
            pytest.skip("Model not loaded")
        body = json.loads(r.data)
        assert 0.0 <= body['probability'] <= 1.0

    def test_predict_perished_payload(self, client):
        r = self._post(client, PERISHED_PAYLOAD)
        assert r.status_code in (200, 503)

    def test_predict_family_size_derived(self, client):
        """FamilySize should be auto-derived if not in payload."""
        payload = {**PERISHED_PAYLOAD}
        payload.pop('SibSp', None)
        payload.pop('Parch', None)
        r = self._post(client, payload)
        assert r.status_code in (200, 503)

    def test_predict_invalid_json_returns_400(self, client):
        r = client.post('/predict',
                        data='not-json',
                        content_type='application/json')
        assert r.status_code == 400

    def test_predict_empty_body_returns_400(self, client):
        r = client.post('/predict',
                        data='',
                        content_type='application/json')
        assert r.status_code == 400


# ── /predict — validation ─────────────────────────────────────────────────────
class TestValidation:
    def _post(self, client, payload):
        return client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )

    def test_invalid_age_out_of_range(self, client):
        payload = {**SURVIVED_PAYLOAD, 'Age': 200}
        r = self._post(client, payload)
        assert r.status_code == 422

    def test_invalid_age_negative(self, client):
        payload = {**SURVIVED_PAYLOAD, 'Age': -5}
        r = self._post(client, payload)
        assert r.status_code == 422

    def test_invalid_age_non_numeric(self, client):
        payload = {**SURVIVED_PAYLOAD, 'Age': 'old'}
        r = self._post(client, payload)
        assert r.status_code == 422

    def test_invalid_pclass(self, client):
        payload = {**SURVIVED_PAYLOAD, 'Pclass': 5}
        r = self._post(client, payload)
        assert r.status_code == 422

    def test_invalid_sex(self, client):
        payload = {**SURVIVED_PAYLOAD, 'Sex': 'unknown'}
        r = self._post(client, payload)
        assert r.status_code == 422

    def test_invalid_embarked(self, client):
        payload = {**SURVIVED_PAYLOAD, 'Embarked': 'X'}
        r = self._post(client, payload)
        assert r.status_code == 422

    def test_invalid_fare_negative(self, client):
        payload = {**SURVIVED_PAYLOAD, 'Fare': -10}
        r = self._post(client, payload)
        assert r.status_code == 422

    def test_validation_returns_errors_list(self, client):
        payload = {**SURVIVED_PAYLOAD, 'Age': 999, 'Pclass': 9}
        r    = self._post(client, payload)
        body = json.loads(r.data)
        assert r.status_code == 422
        assert 'errors' in body
        assert isinstance(body['errors'], list)
        assert len(body['errors']) >= 2


# ── Unit: feature engineering ─────────────────────────────────────────────────
class TestPrepareInput:
    def test_family_size_derived(self):
        df = prepare_input({'SibSp': 2, 'Parch': 1})
        assert df['FamilySize'].iloc[0] == 4

    def test_defaults_applied(self):
        df = prepare_input({})
        assert df['Sex'].iloc[0] == 'male'
        assert df['Pclass'].iloc[0] == 3

    def test_all_features_present(self):
        from app import ALL_FEATURES
        df = prepare_input(SURVIVED_PAYLOAD)
        for feat in ALL_FEATURES:
            assert feat in df.columns, f"Missing feature: {feat}"

    def test_explicit_family_size_respected(self):
        df = prepare_input({'SibSp': 0, 'Parch': 0, 'FamilySize': 5})
        assert df['FamilySize'].iloc[0] == 5


# ── Unit: validate_input ─────────────────────────────────────────────────────
class TestValidateInput:
    def test_valid_payload_no_errors(self):
        errors = validate_input(SURVIVED_PAYLOAD)
        assert errors == []

    def test_empty_payload_no_errors(self):
        """Empty dict — all fields optional, defaults used at inference."""
        errors = validate_input({})
        assert errors == []

    def test_bad_age_string(self):
        errors = validate_input({'Age': 'twenty'})
        assert len(errors) == 1

    def test_multiple_errors(self):
        errors = validate_input({'Age': -1, 'Pclass': 9, 'Sex': 'it'})
        assert len(errors) == 3
