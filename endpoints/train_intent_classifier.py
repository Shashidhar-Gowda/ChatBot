# endpoints/train_intent_classifier.py

import json
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from chatbot import IntentClassifier

# Load intents
with open("intents.json") as f:
    intents = json.load(f)

all_patterns = []
all_tags = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        all_patterns.append(pattern)
        all_tags.append(intent["tag"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_patterns).toarray()
y_labels = list(set(all_tags))
y = [y_labels.index(tag) for tag in all_tags]

input_dim = X.shape[1]
hidden_dim = 64
output_dim = len(set(all_tags))

model = IntentClassifier(input_dim, hidden_dim, output_dim, y_labels, vectorizer)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Training loop
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = loss_fn(output, y_tensor)
    loss.backward()
    optimizer.step()

print("Training complete.")

# Save model and vectorizer
torch.save(model.state_dict(), "intent_model.pt")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
