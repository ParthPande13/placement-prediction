def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "running" in r.json()["message"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["accuracy"] > 0.8
    assert "features" in body


def test_predict_strong_student_is_placed(client, strong_student):
    r = client.post("/predict", json=strong_student)
    assert r.status_code == 200
    body = r.json()
    assert body["placed"] is True
    assert body["prediction"] == "Placed"
    assert 0.5 <= body["placement_probability"] <= 1


def test_predict_weak_student_not_placed(client, weak_student):
    r = client.post("/predict", json=weak_student)
    assert r.status_code == 200
    body = r.json()
    assert body["placed"] is False
    assert body["prediction"] == "Not Placed"
    assert body["placement_probability"] < 0.5


def test_predict_batch(client, strong_student, weak_student):
    r = client.post("/predict/batch", json={"students": [strong_student, weak_student]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert [p["placed"] for p in body["predictions"]] == [True, False]


def test_missing_field_returns_422(client, strong_student):
    strong_student.pop("cgpa")
    assert client.post("/predict", json=strong_student).status_code == 422


def test_out_of_range_cgpa_returns_422(client, strong_student):
    strong_student["cgpa"] = 14
    assert client.post("/predict", json=strong_student).status_code == 422


def test_wrong_type_returns_422(client, strong_student):
    strong_student["iq"] = "high"
    assert client.post("/predict", json=strong_student).status_code == 422


def test_empty_batch_returns_422(client):
    assert client.post("/predict/batch", json={"students": []}).status_code == 422
