from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, # 1000 observations 
    n_features=10, # 5 total features
    n_informative=3, # 3 'useful' features
    n_classes=5, # binary target/label 
    random_state=999 # if you want the same results as mine
)
print(X.shape)
print(y.shape)