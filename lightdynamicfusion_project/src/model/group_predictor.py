from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR


def build_estimator(name: str = 'ridge', random_state: int = 42):
    name = name.lower()
    if name == 'ridge':
        return Ridge(alpha=1.0, random_state=random_state)
    if name == 'lasso':
        return Lasso(alpha=0.05, random_state=random_state, max_iter=5000)
    if name == 'elastic':
        return ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=random_state, max_iter=5000)
    if name == 'svr':
        return SVR(C=1.0, kernel='rbf')
    raise ValueError(f'Unsupported estimator: {name}')
