import unittest
import json
from app import app

class FlaskApiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_root_endpoint(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')

    def test_health_endpoint(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'])
        self.assertIn('Child Age', data['required_features'])

    def test_predict_single_exact_keys(self):
        payload = {
            "Child Age": 12,
            "Birth Weight": 3.1,
            "Birth Length": 49.0,
            "ANC Visits": 4,
            "Exclusive Breastfeeding": 1,
            "Complementary Feeding Start Age": 6,
            "Immunization": 1,
            "Sanitation": 1,
            "Drinking Water Source": 1
        }
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('prediction', data['data'])
        self.assertIn('risk_label', data['data'])
        self.assertIn('probability_stunting', data['data'])
        print("\nSingle Prediction Response (Exact keys):", json.dumps(data, indent=2))

    def test_predict_single_snake_case_keys(self):
        payload = {
            "child_age": 24,
            "birth_weight": 2.2,
            "birth_length": 45.0,
            "anc_visits": 2,
            "exclusive_breastfeeding": 0,
            "mpasi_start_age": 4,
            "immunization": 0,
            "sanitation": 0,
            "drinking_water_source": 0
        }
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('prediction', data['data'])
        print("\nSingle Prediction Response (Snake case keys):", json.dumps(data, indent=2))

    def test_predict_batch(self):
        payload = [
            {
                "child_age": 12,
                "birth_weight": 3.2,
                "birth_length": 49.5,
                "anc_visits": 5,
                "exclusive_breastfeeding": 1,
                "complementary_feeding_start_age": 6,
                "immunization": 1,
                "sanitation": 1,
                "drinking_water_source": 1
            },
            {
                "child_age": 18,
                "birth_weight": 2.0,
                "birth_length": 44.0,
                "anc_visits": 1,
                "exclusive_breastfeeding": 0,
                "complementary_feeding_start_age": 3,
                "immunization": 0,
                "sanitation": 0,
                "drinking_water_source": 0
            }
        ]
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['data']), 2)
        print("\nBatch Prediction Response:", json.dumps(data, indent=2))

    def test_predict_invalid_type(self):
        payload = {
            "child_age": "bukan_angka"
        }
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')

if __name__ == '__main__':
    unittest.main()
