# Training on complete dataset without splitting
import dataset
import mymodels as mm
import neuralnetwork as nn
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

# Read dataset
d = dataset.get_data()
classes = d[0]
m_train, n, X_train, y_train = d[1]
m_test, X_test, y_test = d[2]
m_trial, X_trial, y_trial = d[3]

# Prepare parameters
learning_rates = [4 ** i for i in range(-3, 2)]
m_examples = [10, 40, 100, 200, 400, 800, 1600]
# depth = [1, 2, 3, 4]
depth = [2]
hidden_units = [4 ** i for i in range(3, 5)]
iterations = 100
name = "model_3"

if not os.path.exists(name):
    os.mkdir(name)
"""
# load model
print("m, learning_rate, hidden_units, train_loss, test_loss, train_misc_score, test_misc_score")
for s in tqdm(range(len(m_examples))):
    m = m_examples[s]
    nn.m = m
    for lr in learning_rates:
        for hu in hidden_units:
            model, optimizer = mm.model_3(n, classes, hu, lr, iterations)
            best_network = model.fit(X_train[:m], y_train[:m], X_test, y_test,
                                     optimizer=optimizer, verbose=0)
            train_error, test_error = np.array(optimizer.train_err), np.array(optimizer.test_err)
            # first element is loss and second is accuracy
            train_loss, train_misc = train_error[-1, :]
            test_loss, test_misc = test_error[-1, :]
            print(f"{m}, {lr}, {hu}, {round(train_loss, 4)}, {round(test_loss, 4)}, \
                                    {round(train_misc, 4)}, {round(test_misc, 4)}")
"""
lr = 64
m = 1600
depth = 1
hu = 256
nn.m = m
# model, optimizer = mm.model_1(n, classes, lr, iterations)
# model, optimizer = mm.model_2(n, classes, hu, depth, lr, iterations)
model, optimizer = mm.model_3(n, classes, hu, lr, iterations)
best_network = model.fit(X_train[:m], y_train[:m], X_test, y_test,
                         optimizer=optimizer, verbose=0)
train_error, test_error = np.array(optimizer.train_err), np.array(optimizer.test_err)
"""
plt.figure(1)
plt.plot(train_error[:, 0], label="Training loss")
plt.plot(test_error[:, 0], label="Testing loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (MSE) loss")
plt.title("Best selected model (Model 3)")
plt.savefig(f"{name}//loss_{m}_{lr}.png", dpi=128)

plt.figure(2)
plt.plot(train_error[:, 1], label="Train Misclassification error")
plt.plot(test_error[:, 1], label="Test Misclassification error")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Misclassification error")
plt.title("Best selected model (Model 3)")
plt.savefig(f"{name}//misc_{m}_{lr}.png", dpi=128)

# 5-fold cross-validation
folds = 5
cv = KFold(n_splits=folds)

cv_errors = []

for train_index, test_index in cv.split(X_train):
    xtrain, ytrain = X_train[train_index], y_train[train_index]
    xtest, ytest = X_train[test_index], y_train[test_index]
    model, optimizer = mm.model_1(n, classes, lr, iterations)
    best_network = model.fit(xtrain, ytrain, xtest, ytest, optimizer=optimizer, verbose=0)
    train_error, test_error = np.array(optimizer.train_err), np.array(optimizer.test_err)
    avg_train_misclassification_rate = np.mean(train_error[:, 1])
    avg_test_misclassification_rate = np.mean(test_error[:, 1])
    cv_errors.append([avg_train_misclassification_rate, avg_test_misclassification_rate])

plt.figure(3)
cv_errors = np.asarray(cv_errors)
plt.plot(cv_errors[:, 1], marker="o")
plt.xlabel("Fold")
plt.ylabel("Misclassification error")
plt.title("Averaged CV test misclassification error (Model 3)")
plt.savefig(f"{name}//cv_test_misc_{m}_{lr}.png", dpi=128)
plt.show()

print("Cross-validation results\n---------")
print("cv_avg_train_error, cv_avg_test_error")
for e in cv_errors:
    print(f"{e[0]}, {e[1]}")
"""
# print(X_train.shape)
print(X_trial.shape)
op = model.forwardprop(X_trial.T)
y_pred = np.argmax(op, axis=1)
y_true = np.argmax(y_trial, axis=1)

# for t, p in zip(y_true, y_pred):
#    print(t, p)
# print(best_network.layer[1].W.shape)
shape = best_network.layer[1].W.shape
random_indices = np.random.choice(shape[0], size=10, replace=False)

for i in range(len(random_indices)):
    idx = random_indices[i]
    print(idx)
    plt.imshow(best_network.layer[1].W[idx, 1:].reshape(32, 32))
    plt.axis("off")
    plt.title(f"Output unit = {i+1}")
    plt.savefig(f"model_3//output_unit_{i+1}.png", dpi=128, bbox_inches='tight')
print("Finished.")
