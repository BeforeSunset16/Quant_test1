class Predictor:
    def __init__(self, model=None):
        self.model = model
    def predict(self, X):
        return self.model.predict_proba(X)[:,1]