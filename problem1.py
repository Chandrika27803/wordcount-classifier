"""
Problem 1 – Multithreaded Word Count Classifier

Build a program using two threads and two models (scikit + PyTorch).
- Thread 1: Reads lines from messages.txt, cleans text, pushes to queue.
- Thread 2: Uses TfidfVectorizer and trains Logistic Regression.
- Main thread: Builds a PyTorch classifier (1 hidden layer) using TF-IDF tensors.
- Predict for 5 test sentences using both models and compare results.
-------------------------------------------------------------------------------------
logic :

1.create a queue for sharing data between treads safely
2.create a list to store all the cleaned data
3.thread 1--> 1) open the text file in read model and read lines from it
              2) remove spaces using strip() and convert every line to lowercase
              3) append the cleaned text to the list that we created
              4) then put the cleaned string into the queue by using put() function
              5) after all the lines are read use a marker to tell other thread there are no more lines
4.thread 2--> 1) call get() function to receive the strings sent by thread 1 until end of the file and store the cleaned strings
              2) do TfidfVectorization
              3) train the logistic regression classfier on the sparse matrix retuned after TfidfVectorization
              4) store the trained model for main thread to access it
5.Main thread waits for both threads to finish
6.check if the thread saved the vectorizer,it is to be used for training in pytorch
7.define the neural network architecture with a hidden layer
8.train the model and test it with sentences and predict class

"""


import threading
import queue
import time
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim


q = queue.Queue()
data_store = {}          # store cleaned text + labels
model_store = {}         # store sklearn model + tfidf vectorizer

# Thread 1 → Read file, clean text, push to queue
def thread_reader():
    with open("messages.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text, label = line.rsplit(",", 1)
            text = text.lower().strip()
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
            q.put((text, int(label)))
    q.put(None)   # signal thread is finished

# Thread 2 → Train TF-IDF + Logistic Regression
def thread_sklearn_model():
    texts = []
    labels = []
    while True:
        item = q.get()
        if item is None:
            break  # reader is done
        t, l = item
        texts.append(t)
        labels.append(l)
    # save cleaned data
    data_store["texts"] = texts
    data_store["labels"] = labels
    # TF-IDF
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    # Logistic Regression Classifier
    clf = LogisticRegression()
    clf.fit(X, labels)
    model_store["vec"] = vec
    model_store["clf"] = clf
    print("\n[Thread 2] Logistic Regression Training Completed.\n")

# PyTorch model
class PTClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

# Training PyTorch model
def train_pytorch_model():
    vec = model_store["vec"]
    clf = model_store["clf"]
    texts = data_store["texts"]
    labels = data_store["labels"]
    # TF-IDF → tensor
    X = vec.transform(texts).toarray()
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    model = PTClassifier(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # training
    for epoch in range(12):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    print("[Main Thread] PyTorch training completed.\n")
    return model

# PREDICTION TEST
def test_models(pt_model):

    test_sentences = [
        "You won a lottery claim now",
        "Meeting at office tomorrow",
        "Free money click this link",
        "Let's go out for lunch",
        "Your bank account is locked verify now"
    ]

    vec = model_store["vec"]
    clf = model_store["clf"]
    print("===== PREDICTIONS =====\n")

    for s in test_sentences:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", s.lower())
        X = vec.transform([cleaned])
        # Scikit prediction
        sc_pred = clf.predict(X)[0]
        sc_prob = clf.predict_proba(X)[0][1]
        # PyTorch prediction
        tensor_input = torch.tensor(X.toarray(), dtype=torch.float32)
        pt_prob = pt_model(tensor_input).item()
        pt_pred = 1 if pt_prob > 0.5 else 0

        print(f"Sentence: {s}")
        print(f"  Sklearn → Pred: {sc_pred}, Prob: {round(sc_prob, 4)}")
        print(f"  PyTorch → Pred: {pt_pred}, Prob: {round(pt_prob, 4)}")
        print("")

# MAIN
def main():
    # start threads
    t1 = threading.Thread(target=thread_reader)
    t2 = threading.Thread(target=thread_sklearn_model)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print("[Main Thread] Both threads completed.\n")

    # train pytorch
    pt_model = train_pytorch_model()

    # test
    test_models(pt_model)

if __name__=="__main__":
    main()

