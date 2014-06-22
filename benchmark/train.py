import time

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

t0 = time.clock()
x_train, y_train = load_svmlight_file("data/australian.train")
x_test, y_test = load_svmlight_file("data/australian.test")
t1 = time.clock()

m = LogisticRegression(C = 1)
m.fit(x_train, y_train)
t_train = time.clock()
y1_test = m.predict(x_test)
t_test = time.clock()

precision = precision_score(y_test, y1_test)
recall = recall_score(y_test, y1_test)
accuracy = accuracy_score(y_test, y1_test)
f1 = f1_score(y_test, y1_test)

print "precision=%.5f recall=%.5f accuracy=%.5f f1=%.5f" % (precision, recall, accuracy, f1)
print "time: read=%.3f train=%.3f test=%.3f" % (t1-t0, t_train-t1, t_test-t_train)
