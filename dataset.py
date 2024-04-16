# Instructor provided code to load the dataset
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import numpy as np

out_enc = LabelBinarizer()
np.random.seed(42)


def get_data():
    d = np.loadtxt("data//optdigits_train.dat")
    d_test = np.loadtxt("data//optdigits_test.dat")
    d_trial = np.loadtxt("data//optdigits_trial.dat")

    m, n = d.shape[0], d.shape[1] - 1
    x = d[:, :-1].reshape(m, n)
    y = d[:, -1].reshape(m, 1)

    y_ohe = out_enc.fit_transform(y)
    k = y_ohe.shape[1]
    m_test = d_test.shape[0]
    x_test = d_test[:, :-1].reshape(m_test, n)
    y_test = d_test[:, -1].reshape(m_test, 1)
    y_test_ohe = out_enc.transform(y_test)

    m_trial = d_trial.shape[0]
    x_trial = d_trial[:, :-1].reshape(m_trial, n)
    y_trial = d_trial[:, -1].reshape(m_trial, 1)
    y_trial_ohe = out_enc.transform(y_trial)

    return [k, (m, n, x, y_ohe), (m_test, x_test, y_test_ohe), (m_trial, x_trial, y_trial_ohe)]
