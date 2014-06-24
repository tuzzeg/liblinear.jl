import sys
import time

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

prefix_file = sys.argv[1]
train_file = prefix_file
test_file = '%s.t' % prefix_file

t0 = time.clock()
x_train, y_train = load_svmlight_file(train_file)
r_train, c_train = x_train.shape

x_test, y_test = load_svmlight_file(test_file)
r_test, c_test = x_test.shape

if c_train != c_test:
  x_test = x_test[:, 0:c_train]
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

print "python: train=%s test=%s [%s]" % (x_train.shape, x_train.shape, prefix_file)
print "  precision=%.5f recall=%.5f f1=%.5f" % (precision, recall, f1)
print "  read=%.5f train=%.5f test=%.5f" % (t1-t0, t_train-t1, t_test-t_train)
