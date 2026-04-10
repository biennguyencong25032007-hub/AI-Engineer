from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import TEST_SIZE, RANDOM_STATE

def preprocess(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

def split(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)