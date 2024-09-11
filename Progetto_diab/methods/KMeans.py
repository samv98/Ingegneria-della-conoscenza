from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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

#funzione per eseguire il clustering con il k-means
def kmns(df, infoPrint):

    # classificatore k-means
    if infoPrint:
        print("\n-- ", style.BLUE + "K-MEANS" + style.RESET)

    #rimuove la colonna outcome e altre colonne non rilevanti per il clustering
    new_df = df.drop("Outcome", axis=1)
    new_df = new_df.drop(["Pregnancies", "BloodPressure", "SkinThickness"], axis=1)
    #standardizzazione delle colonne selezionate
    scaler = StandardScaler()
    new_df[['Glucose_T', 'Insulin_T', 'BMI_T', 'DiabetesPedigreeFunction_T', 'Age_T']] = scaler.fit_transform(new_df)
    #dico che ho aggiunto nuove colonne nel dataset
    print(style.YELLOW + "Nuove colonne nel dataset" + style.RESET)
    print(new_df.head())
    print("\n")

    #crea una funzione per determinare il numero ottimale di cluster
    def optimise_k_means(data, max_k):
        means = []
        inertias = []

        for k in range(1, max_k):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)

            means.append(k)
            inertias.append(kmeans.inertia_)

        #genera il grafico elbow plot
        fig = plt.subplots(figsize=(10, 5))
        plt.plot(means, inertias, 'o-')
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.grid(True)
        plt.show()

    print("Numero di cluster:", style.GREEN + "3" + style.RESET)
    optimise_k_means(new_df[['Glucose_T', 'BMI_T']], 10)
    kmeans = KMeans(n_clusters=3)  # eseguo K-means con 3 cluster 
    kmeans.fit_predict(new_df[['BMI_T', 'Glucose_T']]) #addestra e predice i cluster sui dati standardizzati di glucose e BMI
    new_df['labels'] = kmeans.labels_ #aggiunge una nuova colonna labels per i cluster assegnati
    centroids = kmeans.cluster_centers_ #salva i centroidi del cluster 
    print(style.YELLOW + "Cluster completato " + style.RESET)
    print(new_df.head(20))

    #stampo il risultato
    plt.scatter(x=new_df['BMI'], y=new_df['Glucose'], c=new_df['labels'])
    plt.show()

    return kmeans
