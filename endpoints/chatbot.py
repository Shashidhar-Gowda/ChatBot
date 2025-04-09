import torch
import torch.nn as nn
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, intent_labels, vectorizer):
        super(IntentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.labels = intent_labels
        self.vectorizer = vectorizer

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

    def predict(self, text):
        x = self.vectorizer.transform([text]).toarray()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        output = self.forward(x_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        return self.labels[pred_idx]

def predict_intent(text: str) -> str:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("intents.json") as f:
        intents = json.load(f)

    labels = list(set(intent["tag"] for intent in intents["intents"]))

    input_dim = len(vectorizer.get_feature_names_out())
    model = IntentClassifier(input_dim, 64, len(labels), labels, vectorizer)
    model.load_state_dict(torch.load("intent_model.pt"))
    model.eval()

    return model.predict(text)
