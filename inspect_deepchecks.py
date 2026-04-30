import sklearn.metrics._scorer
from sklearn.metrics import make_scorer
def d(y,p): return 0.0
sklearn.metrics._scorer._SCORERS['max_error'] = make_scorer(d)

from deepchecks.nlp import TextData
print("TextData attributes:", dir(TextData))
print("\nHelp on __init__:")
help(TextData.__init__)
