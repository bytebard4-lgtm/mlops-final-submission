from src.preprocessing import load_data

def test_data_shape():
    X_train, X_test, y_train, y_test = load_data()

    assert len(X_train) > 0
    assert len(X_test) > 0
