from src.preprocessing import load_data

def test_labels():
    _, _, y_train, _ = load_data()

    assert set(y_train).issubset({0, 1, 2})
