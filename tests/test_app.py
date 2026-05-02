from app import app


def test_home():
    tester = app.test_client()
    response = tester.get("/")
    assert response.status_code == 200


def test_prediction():
    tester = app.test_client()
    response = tester.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    assert response.status_code == 200
    assert "prediction" in response.get_json()