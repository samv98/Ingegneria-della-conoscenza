import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.preprocessing import StandardScaler
from loading_data.AdjustDataset import loadDataset, optimizationData
from methods.DecisionTree import dtcWithGridView
from methods.RandomForest import rfcWithGridView
from methods.KNN import kncWithGridView
from methods.LogisticRegression import log_regWithGridView
from methods.MultiLayerPerc import mlpWithGridView
from methods.NaiveBayes import nbWithGridView
from methods.KMeans import kmns
import pickle

#file path
dct_filename = 'Progetto_diab\models/dct_model.pkl'
knc_filename = 'Progetto_diab\models/knc_model.pkl'
rfc_filename = 'Progetto_diab\models/rfc_model.pkl'
mlp_filename = 'Progetto_diab\models/mlp_model.pkl'
lr_filename = 'Progetto_diab\models/lr_model.pkl'
nb_filename = 'Progetto_diab\models/nb_model.pkl'
kms_filename = 'Progetto_diab\models/kms_model.pkl'

#carichiamo i dataset e prepariamo i dati
X_train, X_test, y_train, y_test, x, y, df = loadDataset(optimizationData())
df_ytrain = pd.DataFrame(y_train)
df_ytest = pd.DataFrame(y_test)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


def gridSearch():
    #qui stiamo ottimizzando i parametri per il decision tree classifier
    dtc_gd = dtcWithGridView(X_train_scaled, y_train)       # Decision Tree Classifier
    print(f"- DTC Best Params: {dtc_gd.best_params_}")

    #Qui stiamo ottimizzando i parametri per il random forest
    rfc_gd = rfcWithGridView(X_train_scaled, y_train)       # Random Forest
    print(f"- RFC Best Params: {rfc_gd.best_params_}")

    #Qui stiamo ottimizzando i parametri per KNN
    knn_gd = kncWithGridView(X_train_scaled, y_train)       # KNN
    print(f"- KNN Best Params: {knn_gd.best_params_}")

    #qui stiamo ottimizzando i parametri per la Logistic Regression
    lr_gd = log_regWithGridView(X_train_scaled, y_train)    # Logistic Regression
    print(f"- LR Best Params: {lr_gd.best_params_}")

    #Qui stiamo ottimizzando i parametri per il multi Layer Perceptron
    mlp_gd = mlpWithGridView(X_train_scaled, y_train)       # Multi Layer Perceptron
    print(f"- MLP Best Params: {mlp_gd.best_params_}")

    #Qui stiamo ottimizzando i parametri per il Naive Bayes
    nb_gd = nbWithGridView(X_train_scaled, y_train)         # Naive Bayes
    print(f"- NB Best Params: {nb_gd.best_params_}")
    print("End of optimization!")


def modelsTraining(df):
    # Creazione delle feature X e del target Y
    x = df.to_numpy()
    y = df['Outcome'].to_numpy()

    # K-Fold Cross Validation
    kf = RepeatedKFold(n_splits=5, n_repeats=5)
    counter = 0

    #Pipeline per il Decision Tree Classifier 
    pipe_dct = Pipeline([('scaler', StandardScaler()), ('dtc', DecisionTreeClassifier())])
    param_grid = {'dtc__criterion': ['gini', 'entropy'],
                  'dtc__max_depth': range(1, 100)}
    opt_dct = GridSearchCV(estimator=pipe_dct, param_grid=param_grid, scoring='accuracy')

    #Pipeline per il Random forest
    opt_rfc = Pipeline([('sc', StandardScaler()), ('rfc', RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=9, max_features='log2', n_estimators=30))])
    n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]  # Numero di alberi nel random forest
    max_features = ['sqrt', 'log2']    # Numero di feature da considerare ad ogni split
    max_depth = range(1, 10)   # numero massimo di livelli nell'albero
    criterion = ['gini', 'entropy']  # misura della qualità di uno split
    bootstrap = [True, False]   # Metodo di selezione dei campioni per addestrare ogni albero
    param_grid = {'rfc__n_estimators': n_estimators,
                  'rfc__max_features': max_features,
                  'rfc__max_depth': max_depth,
                  'rfc__criterion': criterion,
                  'rfc__bootstrap': bootstrap}
    #opt_rfc = GridSearchCV(estimator=pipe_rfc, param_grid=param_grid, scoring='accuracy')

    #Pipeline per il KNN
    pipe_knn = Pipeline([('sc', StandardScaler()), ('knn', KNeighborsClassifier())])
    param_grid = {'knn__n_neighbors': [5, 10, 15, 30, 60, 90, 120],
                  'knn__weights': ['uniform', 'distance'],
                  'knn__algorithm': ['kd_tree', 'ball_tree', 'brute'],
                  'knn__p': [1, 2]}
    opt_knn = GridSearchCV(estimator=pipe_knn, param_grid=param_grid, scoring='accuracy', return_train_score=False)

    #Pipeline per la Logistic Regression 
    pipe_logr = Pipeline([('sc', StandardScaler()), ('logr', LogisticRegression())])
    param_grid = {'logr__penalty': ['l2'],
                  'logr__C': [0.001, 0.01, 0.1, 1, 2, 3, 5, 10, 100, 1000]}
    opt_logr = GridSearchCV(estimator=pipe_logr, param_grid=param_grid, scoring='accuracy')

    #Pipeline per il Multi Layer Perceptron
    opt_mlp = Pipeline([('sc', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(17,), max_iter=500))])
    param_grid = [
        {
            'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'mlp__solver': ['lbfgs', 'sgd', 'adam'],
            'mlp__hidden_layer_sizes': [
                (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,),
                (17,), (18,), (19,), (20,), (21,)
            ]
        }
    ]
    #opt_mlp = GridSearchCV(estimator=pipe_mlp, param_grid=param_grid, scoring='accuracy')

    #Pipeline per il Naive Bayes
    pipe_nb = Pipeline([('sc', StandardScaler()), ('nb', GaussianNB())])
    param_grid = {'nb__var_smoothing': np.logspace(0, -9, num=100)}
    opt_nb = GridSearchCV(estimator=pipe_nb, param_grid=param_grid, scoring='accuracy')

    # Dizionario per memorizzare le metriche dei modelli
    models = {
        'DecisionTree': {'accuracy_list': 0.0,
                         'precision_list': 0.0,
                         'recall_list': 0.0,
                         'f1_list': 0.0
                         },

        'RandomForest': {'accuracy_list': 0.0,
                         'precision_list': 0.0,
                         'recall_list': 0.0,
                         'f1_list': 0.0
                         },

        'KNN': {'accuracy_list': 0.0,
                'precision_list': 0.0,
                'recall_list': 0.0,
                'f1_list': 0.0
                },

        'LogisticRegression': {'accuracy_list': 0.0,
                               'precision_list': 0.0,
                               'recall_list': 0.0,
                               'f1_list': 0.0
                               },

        'MultiLayerPerc': {'accuracy_list': 0.0,
                           'precision_list': 0.0,
                           'recall_list': 0.0,
                           'f1_list': 0.0
                           },

        'GaussianNB': {'accuracy_list': 0.0,
                       'precision_list': 0.0,
                       'recall_list': 0.0,
                       'f1_list': 0.0
                       }

    }

    print(style.YELLOW + "Fitting dei modelli..." + style.RESET)

    #ciclo di cross-validation
    for train_index, test_index in kf.split(x, y):
        training_set, testing_set = x[train_index], x[test_index]

        counter = counter+1
        print(style.BLUE + "Contatore: " + style.RESET, counter, "/25")

        # dati di addestramento
        data_train = pd.DataFrame(training_set, columns=df.columns)
        X_train = data_train.drop("Outcome", axis=1)
        y_train = data_train.Outcome

        # data di test
        data_test = pd.DataFrame(testing_set, columns=df.columns)
        X_test = data_test.drop("Outcome", axis=1)
        y_test = data_test.Outcome

        # addestramento dei classificatori
        opt_dct.fit(X_train, y_train)
        print("DCT fitted (1/6)")
        opt_rfc.fit(X_train, y_train)
        print("RFC fitted (2/6)")
        opt_knn.fit(X_train, y_train)
        print("KNN fitted (3/6)")
        opt_logr.fit(X_train, y_train)
        print("LR fitted (4/6)")
        opt_mlp.fit(X_train, y_train)
        print("MLP fitted (5/6)")
        opt_nb.fit(X_train, y_train)
        print("NB fitted (6/6)")

        #predizione
        y_pred_dct = opt_dct.predict(X_test)
        print("DCT predict (1/6)")
        y_pred_rfc = opt_rfc.predict(X_test)
        print("RFC predict (2/6)")
        y_pred_knn = opt_knn.predict(X_test)
        print("KNN predict (3/6)")
        y_pred_logr = opt_logr.predict(X_test)
        print("LR predict (4/6)")
        y_pred_mlp = opt_mlp.predict(X_test)
        print("MLP predict (5/6)")
        y_pred_nb = opt_nb.predict(X_test)
        print("NB predict (6/6)")

        # salvataggio delle metriche nel dizionario
        print("Salvo le metriche...")
        models['DecisionTree']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_dct))
        models['DecisionTree']['precision_list'] = (metrics.precision_score(y_test, y_pred_dct))
        models['DecisionTree']['recall_list'] = (metrics.recall_score(y_test, y_pred_dct))
        models['DecisionTree']['f1_list'] = (metrics.f1_score(y_test, y_pred_dct))

        models['RandomForest']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_rfc))
        models['RandomForest']['precision_list'] = (metrics.precision_score(y_test, y_pred_rfc))
        models['RandomForest']['recall_list'] = (metrics.recall_score(y_test, y_pred_rfc))
        models['RandomForest']['f1_list'] = (metrics.f1_score(y_test, y_pred_rfc))

        models['KNN']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_knn))
        models['KNN']['precision_list'] = (metrics.precision_score(y_test, y_pred_knn))
        models['KNN']['recall_list'] = (metrics.recall_score(y_test, y_pred_knn))
        models['KNN']['f1_list'] = (metrics.f1_score(y_test, y_pred_knn))

        models['LogisticRegression']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_logr))
        models['LogisticRegression']['precision_list'] = (metrics.precision_score(y_test, y_pred_logr))
        models['LogisticRegression']['recall_list'] = (metrics.recall_score(y_test, y_pred_logr))
        models['LogisticRegression']['f1_list'] = (metrics.f1_score(y_test, y_pred_logr))

        models['MultiLayerPerc']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_mlp))
        models['MultiLayerPerc']['precision_list'] = (metrics.precision_score(y_test, y_pred_mlp))
        models['MultiLayerPerc']['recall_list'] = (metrics.recall_score(y_test, y_pred_mlp))
        models['MultiLayerPerc']['f1_list'] = (metrics.f1_score(y_test, y_pred_mlp))

        models['GaussianNB']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_nb))
        models['GaussianNB']['precision_list'] = (metrics.precision_score(y_test, y_pred_nb))
        models['GaussianNB']['recall_list'] = (metrics.recall_score(y_test, y_pred_nb))
        models['GaussianNB']['f1_list'] = (metrics.f1_score(y_test, y_pred_nb))

        # report dei modelli
        def model_report(model):

            df_models = []

            for clf in model:
                df_model = pd.DataFrame({'model': [clf],
                                         'accuracy': [np.mean(model[clf]['accuracy_list'])],
                                         'precision': [np.mean(model[clf]['precision_list'])],
                                         'recall': [np.mean(model[clf]['recall_list'])],
                                         'f1score': [np.mean(model[clf]['f1_list'])]
                                         })

                df_models.append(df_model)
            return df_models

    df_models_concat = pd.concat(model_report(models), axis=0).reset_index()  # concatenazione dei modelli
    df_models_concat = df_models_concat.drop('index', axis=1)  # rimozione del'indice
    print("\n", df_models_concat)  # visualizzazione della tabella

    """
    # Accuracy Graph
    x = df_models_concat.model
    y = df_models_concat.accuracy

    plt.bar(x, y)
    plt.title("Accuracy")
    plt.show()

    # Precision Graph
    x = df_models_concat.model
    y = df_models_concat.precision

    plt.bar(x, y)
    plt.title("Precision")
    plt.show()

    # Recall Graph
    x = df_models_concat.model
    y = df_models_concat.recall

    plt.bar(x, y)
    plt.title("Recall")
    plt.show()

    # F1score Graph
    x = df_models_concat.model
    y = df_models_concat.f1score

    plt.bar(x, y)
    plt.title("F1score")
    plt.show()
    """

    # deviazione standard
    print(style.YELLOW + "\nCalcolo la deviazione standard..." + style.RESET)
    std_dtc = np.std(cross_val_score(opt_dct, X_test, y_test, cv=5))
    std_rfc = np.std(cross_val_score(opt_rfc, X_test, y_test, cv=5))
    std_knn = np.std(cross_val_score(opt_knn, X_test, y_test, cv=5))
    std_logr = np.std(cross_val_score(opt_logr, X_test, y_test, cv=5))
    std_mlp = np.std(cross_val_score(opt_mlp, X_test, y_test, cv=5))
    std_nb = np.std(cross_val_score(opt_nb, X_test, y_test, cv=5))
    """plt.plot(["DecisionTree", "RandomForest", "KNN",  "LogisticRegression", "MultiLayerPerceptron", "GaussianNB"],
             [std_dtc, std_rfc, std_knn, std_logr, std_mlp, std_nb])
    plt.title("Standard deviation")
    plt.ylabel("Standard deviation value")
    plt.xlabel("Classifiers")
    plt.show()"""
    print("\ndeviazione standard per il  DecisionTree:", std_dtc)
    print("\ndeviazione standard per il RandomForest:", std_rfc)
    print("\ndeviazione standard per il Knn:", std_knn)
    print("\ndeviazione standard per il LogisticRegression:", std_logr)
    print("\ndeviazione standard per il MultilayerPerceptron:", std_mlp)
    print("\ndeviazione standard per il GaussianNB:", std_nb)

    #pickle.dump(opt_dct.fit(X_train, y_train), open(dct_filename, 'wb'))
    #pickle.dump(opt_rfc.fit(X_train, y_train), open(rfc_filename, 'wb'))
    #pickle.dump(opt_knn.fit(X_train, y_train), open(knc_filename, 'wb'))
    #pickle.dump(opt_logr.fit(X_train, y_train), open(lr_filename, 'wb'))
    #pickle.dump(opt_mlp.fit(X_train, y_train), open(mlp_filename, 'wb'))
    #pickle.dump(opt_nb.fit(X_train, y_train), open(nb_filename, 'wb'))
    print(style.GREEN + "Models saved!" + style.RESET)

    return opt_dct, opt_rfc, opt_knn, opt_logr, opt_mlp, opt_nb

#funzione per il salvataggio del modello di clustering
def notSup():

    kms = kmns(df, True)
    pickle.dump(kms.fit(X_train, y_train), open(kms_filename, 'wb'))
    print(style.GREEN + "Model saved!" + style.RESET)

#funzione per testare i modelli caricati
def test():

    #carica i modelli salvati
    dct_load = pickle.load(open(dct_filename, 'rb'))
    knn_load = pickle.load(open(knc_filename, 'rb'))
    rfc_load = pickle.load(open(rfc_filename, 'rb'))
    mlp_load = pickle.load(open(mlp_filename, 'rb'))
    lr_load = pickle.load(open(lr_filename, 'rb'))
    nb_load = pickle.load(open(nb_filename, 'rb'))
    kms_load = pickle.load(open(kms_filename, 'rb'))
    print(f"\n-- Loaded trained models.")

    #qui stiamo definendo alcuni campioni di test
    # 'Preg', 'Gluc', 'BloodP', 'SkinTh', 'Insul', 'BMI', 'DPF', 'Age'
    xx = [[1, 89, 66, 23, 94, 28.1, 0.167, 21]]        # reale: non-diabetico (riga 5 dataset)
    xy = [[2, 197, 70, 45, 543, 30.5, 0.158, 53]]      # reale: diabetico (riga 10 del dataset)
    xz = [[0, 118, 84, 47, 230, 45.8, 0.551, 31]]      # reale: diabetico (riga 18 del dataset)
    zz = [[1, 85, 64, 25, 95, 27, 0.165, 31]]        # inventato: non-diabetico (ipotetico)
    zx = [[3, 150, 75, 40, 250, 41, 0.256, 42]]      # inventato: diabetico (ipotetico)
    # 'Preg', 'Gluc', 'BloodP', 'SkinTh', 'Insul', 'BMI', 'DPF', 'Age'

    #test con un paziente non diabetico
    test = xx
    print(style.YELLOW + "\nTEST su un paziente non diabetico: " + style.RESET, test[0], "\n")
    pred7 = kms_load.predict(test)  #predizione del cluster con Kmeans
    pred1 = dct_load.predict_proba(test)[:, 1] #prendo la probabilità che sia diabetico con decision tree
    pred2 = rfc_load.predict_proba(test)[:, 1] #prendo la probabilità che sia diabetico con randomForest
    pred3 = knn_load.predict_proba(test)[:, 1] #prendo la probabilità che sia diabetico con KNN
    pred4 = lr_load.predict_proba(test)[:, 1] #prendo la probabilità che sia diabetico con LogisticRegression
    pred5 = mlp_load.predict_proba(test)[:, 1] #prendo la probabilità che sia diabetico con MultiLayerPerceptron
    pred6 = nb_load.predict_proba(test)[:, 1] #prendo la probabilità che sia diabetico con GaussianNB

    # PREDIZIONE
    """print("DCT prediction: ", pred1)
    print("RFC prediction: ", pred2)
    print("KNN prediction: ", pred3)
    print("LR prediction: ", pred4)
    print("MLP prediction: ", pred5)
    print("NB prediction: ", pred6)"""
    print(style.BLUE + "predizione KMeans" + style.RESET, "Cluster: ", pred7)

    prob = (pred1 + pred2 + pred3 + pred4 + pred5 + pred6) / 6
    if prob > 0.5:
        print("questo paziente", style.RED + "ha il diabete" + style.RESET, ".\n-Probabilità: ", style.RED + str(round(float(prob)*100, 2)) + style.RESET, "%")
    else:
        print("questo paziente", style.GREEN + "non ha il diabete" + style.RESET, ".\n-Probabilità: ", style.GREEN + str(round(float(1-prob)*100, 2)) + style.RESET, "%")

    #test con un paziente diabetico
    test = xy
    print(style.YELLOW + "\nTEST su un paziente diabetico: " + style.RESET, test[0], "\n")
    pred7 = kms_load.predict(test) #predizione del cluster con KMeans
    pred1 = dct_load.predict_proba(test)[:, 1]  # probabilità che sia diabetico con DecisionTree
    pred2 = rfc_load.predict_proba(test)[:, 1] #probabilità che sia diabetico con Random Forest
    pred3 = knn_load.predict_proba(test)[:, 1] #probabilità che sia diabetico con KNN
    pred4 = lr_load.predict_proba(test)[:, 1] #probabilità che sia diabetico con LogisticRegression
    pred5 = mlp_load.predict_proba(test)[:, 1] #probabilità che sia diabetico con MultiLayerPerceptron
    pred6 = nb_load.predict_proba(test)[:, 1] #probabilità che sia diabetico con GaussianNB

    # PREDIZIONE
    """print("DCT prediction: ", pred1)
    print("RFC prediction: ", pred2)
    print("KNN prediction: ", pred3)
    print("LR prediction: ", pred4)
    print("MLP prediction: ", pred5)
    print("NB prediction: ", pred6)"""
    print(style.BLUE + "predizione KMeans" + style.RESET, "Cluster: ", pred7)

    prob = (pred1 + pred2 + pred3 + pred4 + pred5 + pred6) / 6
    if prob > 0.5:
        print("questo paziente", style.RED + "ha il diabete" + style.RESET, ".\n-Probabilità: ",
              style.RED + str(round(float(prob) * 100, 2)) + style.RESET, "%")
    else:
        print("questo paziente", style.GREEN + "non ha il diabete" + style.RESET, ".\n-Probabilità: ",
              style.GREEN + str(round(float(1 - prob) * 100, 2)) + style.RESET, "%")


def prediction():
    print("Per favore inserisci i tuoi valori.")
    print(style.RED + "se un parametro è rosso è obbligatorio!" + style.RESET)

    ok = 0
    while ok == 0:
        preg = int(input("Inserisci il numero delle tue gravidanze:\n"))
        if preg < -2 and preg != -1:
            print("Inserisci un numero corretto.")
        elif preg == -1:
            preg = 2
            ok = 1
        else:
            ok = 1
    ok = 0
    while ok == 0:
        gluc = float(input(style.RED + "Inserisci il tuo livello di glucosio nel sangue:\n(>70)" + style.RESET))
        if gluc < 70 and gluc != -1:
            print("Inserisci un numero corretto.")
        elif gluc == -1:
            print(style.YELLOW + "devi inserire il valore" + style.RESET)
        else:
            ok = 1
    ok = 0
    while ok == 0:
        print(style.YELLOW + "se non conosci la risposta inserisci '-1'." + style.RESET)
        blood = int(input("Inserisci la tua pressione sanguigna diastolica:\n(>40)"))
        if blood < 40 and blood != -1:
            print("Inserisci un numero corretto.")
        elif blood == -1:
            blood = 75
            ok = 1
        else:
            ok = 1
    ok = 0
    while ok == 0:
        print(style.YELLOW + "se non conosci la risposta premi '-1'." + style.RESET)
        skin = float(input("Inserisci lo spessore della tua pelle:\n(>15)"))
        if skin < 15 and skin != -1:
            print("Inserisci un numero corretto.")
        elif skin == -1:
            skin = 25
            ok = 1
        else:
            ok = 1
    ok = 0
    while ok == 0:
        insul = float(input(style.RED + "Inserisci il tuo livello di insulina:\n(>16)" + style.RESET))
        if insul < 16 and insul != -1:
            print("Inserisci un numero corretto.")
        elif insul == -1:
            print(style.YELLOW + "devi inserire un valore." + style.RESET)
        else:
            ok = 1
    ok = 0
    while ok == 0:
        print(style.YELLOW + "se non conosci la risposta premi '-1'." + style.RESET)
        bmi = float(input("Inserisci la tua BMI:\n(>19)"))
        if bmi < 19 and bmi != -1:
            print("Inserisci il numero corretto.")
        elif bmi == -1:
            bmi = 20
            ok = 1
        else:
            ok = 1
    ok = 0
    while ok == 0:
        print(style.YELLOW + "se non conosci la risposta premi '-1'." + style.RESET)
        dpf = int(input("Inserisci la tua funzione di pedigree del diabete\n(0<x<100)"))
        if dpf < -2:
            print("Inserisci un numero corretto.")
        elif dpf == -1:
            dpf = 50
            ok = 1
        else:
            ok = 1
    ok = 0
    while ok == 0:
        age = int(input(style.RED + "Inserisci la tua età: \n" + style.RESET))
        if age < 20 and age != -1:
            print("inserisci un numero corretto.")
        elif age == -1:
            print(style.YELLOW + "devi inserire un valore" + style.RESET)
        else:
            ok = 1

    dct_load = pickle.load(open(dct_filename, 'rb'))
    knn_load = pickle.load(open(knc_filename, 'rb'))
    rfc_load = pickle.load(open(rfc_filename, 'rb'))
    mlp_load = pickle.load(open(mlp_filename, 'rb'))
    lr_load = pickle.load(open(lr_filename, 'rb'))
    nb_load = pickle.load(open(nb_filename, 'rb'))
    kms_load = pickle.load(open(kms_filename, 'rb'))
    print(f"\n-- modelli caricati.")

    patient = [[preg, gluc, blood, skin, insul, bmi, dpf, age]]

    print("\nvalori del paziente:\t", patient)
    pred7 = kms_load.predict(patient)
    pred1 = dct_load.predict_proba(patient)[:, 1]  # prendo la probabilità che sia diab(=1)
    pred2 = rfc_load.predict_proba(patient)[:, 1]
    pred3 = knn_load.predict_proba(patient)[:, 1]
    pred4 = lr_load.predict_proba(patient)[:, 1]
    pred5 = mlp_load.predict_proba(patient)[:, 1]
    pred6 = nb_load.predict_proba(patient)[:, 1]

    print(style.YELLOW + "predizione KMeans" + style.RESET, "Cluster: ", pred7)
    prob = (pred1 + pred2 + pred3 + pred4 + pred5 + pred6) / 6
    if prob > 0.5:
        print("questo paziente", style.RED + "ha il diabete" + style.RESET, ".\n-Probability: ", style.RED + str(round(float(prob)*100, 2)) + style.RESET, "%")
    else:
        print("questo paziente", style.GREEN + "non ha il diabete" + style.RESET, ".\n-Probability: ", style.GREEN + str(round(float(1-prob)*100, 2)) + style.RESET, "%")


# Classe per la gestione dei colori e stili del testo
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


print("\n", style.BLUE + "predizione del diabete" + style.RESET)
# System call
os.system("")

menu_options = {
    1: 'Apprendimento Supervisionato',
    2: 'Apprendimento non supervisionato',
    3: 'Ottimizza i parametri',
    4: 'Esegui Test',
    5: 'Predizione',
    6: 'Esci dal programma',
}


def print_menu():
    for key in menu_options.keys():
        print(key, '--', menu_options[key])


if __name__ == '__main__':
    while (True):
        print('')
        print_menu()
        option = ''
        try:
            option = int(input('scegli un numero per iniziare ad esplorare il programma:\n '))
        except:
            print('per favore inserisci un numero')
        if option == 1:
            print(style.RED + "questa operazione richiederà un pò di tempo\n attendi per favore" + style.RESET)
            modelsTraining(df)
        elif option == 2:
            notSup()
        elif option == 3:
            gridSearch()
        elif option == 4:
            test()
        elif option == 5:
            prediction()
        elif option == 6:
            print('il programma è in chiusura')
            exit()
        else:
            print('per favore inserisci un numero da 1 a 5.')
