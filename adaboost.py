import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert the data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize weights
weights = torch.full((len(X_train),), 1/len(X_train))

# Number of boosting rounds
M = 100

# Initialize empty list for storing models and their alpha values
models = []
alpha_values = []

for m in range(M):
    # Train a decision tree classifier
    model = DecisionTreeClassifier(max_depth=1)
    model.fit(X_train, y_train, sample_weight=weights.detach().numpy())

    # Make predictions
    pred = model.predict(X_train)
    pred = torch.tensor(pred, dtype=torch.float32)

    # Calculate error
    error = weights[(pred != y_train)].sum() / weights.sum()

    # Calculate alpha
    alpha = 0.5 * torch.log((1 - error) / error)

    # Update weights
    weights = weights * torch.exp(-alpha * y_train * pred)
    weights = weights / weights.sum()

    # Store the model and alpha value
    models.append(model)
    alpha_values.append(alpha)

# Make predictions on the test set
preds = [alpha * torch.tensor(model.predict(X_test), dtype=torch.float32) for model, alpha in zip(models, alpha_values)]
y_pred = torch.sign(sum(preds))

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)