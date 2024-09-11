from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


def log_regWithGridView(x_train, y_train):
    print(" - Execute LR with Grid View")
    print(style.YELLOW + "\tCalcolo degli iperparametri ottimali" + style.RESET)

    #creazione di una pipeline che include standardScaler per normalizzare i dati e logisticRegression come classificatore
    pipe = Pipeline([('sc', StandardScaler()), ('logr', LogisticRegression(C=0.1, penalty='l2'))])
    # Creiamo la griglia degli iperparametri da esplorare
    param_grid = {'logr__penalty': ['l2'],
                  'logr__C': [0.001, 0.01, 0.1, 1, 2, 3, 5, 10, 100, 1000]}
   
    #utilizzo GridSearchCV per trovare i migliori iperparametri
    optimal_params = GridSearchCV(estimator=pipe,
                                  param_grid=param_grid,
                                  cv=5,  # we are taking 5-fold as in k-fold cross validation
                                  scoring='accuracy')

    optimal_params.fit(x_train, y_train)
    return optimal_params
