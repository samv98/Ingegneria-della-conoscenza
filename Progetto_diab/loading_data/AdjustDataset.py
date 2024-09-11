import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


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


# Funzione per mostrare percentuali maggiori di 1% sui grafici a torta
def autopct(pct):
    return ('%.2f' % pct + "%") if pct > 1 else ''  # mostra solo i valori delle etichette maggiori di 1%

#funzione per ottimizzare i dati del dataset
def optimizationData():
    df = pd.read_csv(r"Progetto_diab\diabetes.csv")
    pd.set_option('display.max_columns', None)
    print("Dataset caricato.")

    # OTTIMIZZAZIONE DATI

    # Eliminazione valori nulli
    print(df.describe())

    # Sostituisco i valori pari a 0 con NaN per alcune colonne 
    df.loc[df["Glucose"] == 0.0, "Glucose"] = np.nan
    df.loc[df["BloodPressure"] == 0.0, "BloodPressure"] = np.nan
    df.loc[df["SkinThickness"] == 0.0, "SkinThickness"] = np.nan
    df.loc[df["Insulin"] == 0.0, "Insulin"] = np.nan
    df.loc[df["BMI"] == 0.0, "BMI"] = np.nan

    #visualizzazione dei valori nulli
    print(style.RED + "\nValori con zero." + style.RESET)
    print(df.isnull().sum()[1:6])

     # Riempimento dei valori nulli con la media della colonna corrispondente
    print(style.YELLOW + "Filling null values..." + style.RESET)
    # Imput sui nan facendo la media
    df["Glucose"].fillna(df["Glucose"].mean(), inplace=True)
    df["BloodPressure"].fillna(df["BloodPressure"].mean(), inplace=True)
    df["SkinThickness"].fillna(df["SkinThickness"].mean(), inplace=True)
    df["Insulin"].fillna(df["Insulin"].mean(), inplace=True)
    df["BMI"].fillna(df["BMI"].mean(), inplace=True)
    print(df.isnull().sum())
    print("\n")


    print(style.YELLOW + "Controllo del bilanciamento delle classi" + style.RESET)
    """
    # vediamo quanti diab e non
    labels = ["Not diabetes", "Diabetes"]
    ax = df['Outcome'].value_counts().plot(kind='pie', figsize=(5, 5), autopct=autopct, labels=None)
    ax.axes.get_yaxis().set_visible(False)
    plt.title("Graph of occurrence of diabetes and non-diabetes")
    plt.legend(labels=labels, loc="best")
    plt.show()
    """
    #stampa il numero di casi di diabete e non-diabete nel dataset
    print(style.GREEN + "Non diabetici: " + style.RESET, df.Outcome.value_counts()[0],
                      '(% {:.2f})'.format(df.Outcome.value_counts()[0] / df.Outcome.count() * 100))
    print(style.RED + "Diabetici: " + style.RESET, df.Outcome.value_counts()[1],
                    '(% {:.2f})'.format(df.Outcome.value_counts()[1] / df.Outcome.count() * 100))

    #creazione del dataset per il resampling
    df_majority = df[df["Outcome"] == 0] #classe maggioritaria (non diabetici)
    df_minority = df[df["Outcome"] == 1] #classe minoritaria (diabetici)

    #aumentiamo il numero di campioni della classe minoritaria fino a 500(oversampling)
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=500, random_state=42)
    #combiniamo i dati originali della classe maggioritaria
    df = pd.concat([df_minority_upsampled, df_majority])

    #mostriamo i valori dopo il bilanciamento delle classi
    print(style.YELLOW + "\nValori dopo oversampling:" + style.RESET)
    print(style.GREEN + "Non diabetici: " + style.RESET, df.Outcome.value_counts()[0],
                      '(% {:.2f})'.format(df.Outcome.value_counts()[0] / df.Outcome.count() * 100))
    print(style.RED + "Diabetici: " + style.RESET, df.Outcome.value_counts()[1],
                    '(% {:.2f})'.format(df.Outcome.value_counts()[1] / df.Outcome.count() * 100))

    """
    # Visualizzazione grafica del rapporto tra diabetici e non diabetici
    labels = ["Not diabetes", "Diabetes"]
    ax = df['Outcome'].value_counts().plot(kind='pie', figsize=(5, 5), autopct=autopct, labels=None)
    ax.axes.get_yaxis().set_visible(False)
    plt.title("Graph of occurrence of diabetes and non-diabetes\n\nafter oversampling")
    plt.legend(labels=labels, loc="best")
    plt.show()
    
    
    # vediamo la correlazione tra le features
    # figure size
    plt.figure(figsize=(10, 10))
    # correlation matrix
    dataplot = sns.heatmap(df.corr(), annot=True, fmt='.2f')
    plt.show()
    """
    return df

#funzione per caricare il dataset e separare i dati di training e test
def loadDataset(df):

    y = df['Outcome'].values
    #selezioniamo le variabili indipendenti
    X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
            'Age']].values

    #calcoliamo la matrice di correlazione e ordiniamo i valori in base alla correlazione con l'outcome
    correlation_matrix = df.corr()
    correlation_matrix['Outcome'].sort_values(ascending=False)
    #print("Correlation Matrix:\n", correlation_matrix)
    #suddivisione dei dati in set di addestramento e test (75 training 25 set)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=25)
    return X_train, X_test, y_train, y_test, X, y, df