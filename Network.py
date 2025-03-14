import math
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import itertools



#otteniamo il numero di colonne presenti nei dati:
input_dim = 115

#creiamo la classe che modella l'autoencoder:
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded







#definisco una seconda rete neurale che apprenda come classificare il
#dataset come anomalo o meno. Questa rete riceverà il valore di
#"threshold" dall'autoencoder che tiene conto della difficoltà di
#ricostruzione e, tramite quella, classifica il dataset corrente,
#a partire dall'errore di ricostruzione calcolato per quel dataset
#dall'autoencoder, come "anomalo" o "non anomalo":
class Classificatore(nn.Module):
    def __init__(self):
        super(Classificatore, self).__init__()
        self.liv1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.liv2 = nn.Linear(4, 8)
        self.liv3 = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()

    def forward(self, threshold, loss):
        threshold_tensor = torch.tensor([threshold], dtype=torch.float32)
        loss_tensor = torch.tensor([loss], dtype=torch.float32)
        combined_input = torch.cat((threshold_tensor, loss_tensor), dim=0)

        out = self.liv1(combined_input)
        out = self.relu(out)
        out = self.liv2(out)
        out = self.relu(out)
        out = self.liv3(out)
        out = self.sig(out)
        return out


#creo un'istanza di rete neurale:
model = Classificatore()



def addestraSecondaRete(threshold, losses, anomalo):
    #scelgo i parametri di apprendimento della rete:
    learning_rate = 0.001
    num_epochs = 150#100
    global model

    #scelgo la funzione di errore e l'ottimizzatore da utilizzare:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # effettuo l'addestramento:
    for epoch in range(num_epochs):

        for indice in range(len(losses)):
            # definisco il risultato corrette di classificazione in approccio
            # di apprendimento supervisionato:
            if anomalo[indice]==False:
                labels = np.array([0.0])
            else:
                labels = np.array([1.0])
            # valuto risultato e errore:
            outputs = model(threshold, losses[indice])
            tensor_from_numpy = torch.tensor(labels, dtype=torch.float32)
            loss = criterion(outputs, tensor_from_numpy)

            # Backpropagation e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def prepara_dati_addestramento(percorso, autoencoder, criterion, threshold, scaler):
    # A questo punto non si vuole usare direttamente il valore di sogli appena
    # calcolato per rilevare anomalie, ma si decide di addestrare la seconda
    # rete neurale che apprenda come riconoscere, dato un valore di soglia e
    # l'errore attuale se si sta valutando il modello in condizioni di
    # anomalie della rete o in condizioni di traffico normale:
    elementi = 40 #30
    anomalie = 20   #7
    booleani=[]
    perditeAddestramento = []
    lista_indici = []
    meta = math.ceil(len(percorso) / 2)
    while len(lista_indici) != meta:
        i = random.randint(0, len(percorso) - 1)
        if i not in lista_indici:
            lista_indici.append(i)
    for j in lista_indici:

        for epoch in range(elementi):
                data_path = f"{percorso[0]}/benign_traffic.csv"
                data = pd.read_csv(data_path)
                #genero un seme random
                casuale = random.randint(10, 100)
                data = data.sample(frac=0.1, random_state=casuale)
                scaler = StandardScaler()
                data_normalized = scaler.fit_transform(data)
                data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
                outputs = autoencoder(data_tensor)
                # determiniamo l'errore di Ricostruzione dell'autoencoder:
                loss = criterion(outputs, data_tensor)
                # valuto risultato e errore:
                perditeAddestramento.append(loss)
                booleani.append(False)
        for _ in range(anomalie):
                # prendiamo il dataset relativo allo specifico attacco:
                data_path2 = f"{percorso[0]}/gafgyt_attacks/udp.csv"
                # si leggono e normalizzano i dati tramite la libreria pandas:
                data2 = pd.read_csv(data_path2)
                data2 = data2.sample(frac=0.1, random_state=10)
                data_normalized2 = scaler.transform(data2)
                data_tensor2 = torch.tensor(data_normalized2, dtype=torch.float32)
                # si determina il risultato e l'errore di RICOSTRUZIONE del modello:
                outputs2 = autoencoder(data_tensor2)
                loss2 = criterion(outputs2, data_tensor2)
                perditeAddestramento.append(loss2)
                booleani.append(True)








    addestraSecondaRete(threshold, perditeAddestramento, booleani)






def testSecondaRetethreshold(threshold, loss):

    global model
    # valuto risultato e errore:
    outputs = model(threshold, loss)
    return outputs


best_epochs = None
best_lr = None


def stimaMiglioriParametri(data_tensor):


    all_epochos = [250]#150
    all_lr = [0.001]

    global best_epochs
    global best_lr

    best_loss = float('inf')

    # per ogni possobile combinazione si verifica se è quella ottimale:
    for num_epochs, lr in itertools.product(all_epochos, all_lr):
        # Crea una nuova istanza della classe:
        autoencoder = Autoencoder()
        # usiamo l'MSE per calcolare l'errore prodotto:
        criterion = nn.MSELoss()
        # Crea un nuovo ottimizzatore con il tasso di apprendimento corrente
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

        losses = []

        # Addestramento dell'autoencoder
        for epoch in range(num_epochs):
            outputs = autoencoder(data_tensor)
            loss = criterion(outputs, data_tensor)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # prima facevo solo:
        # average_loss = sum(losses) / len(losses)

        perdite = sum(losses[-40:]) / len(losses[-40:])
        deviazioneStandard = np.std(losses[-60:])

        if perdite < best_loss and deviazioneStandard >= 0.00000001: #MODIFICA
            best_loss = perdite
            best_epochs = num_epochs
            best_lr = lr




def buildAutoencoder(percorso, assembla):
    if assembla:
        unifica_dati_anomali(percorso)
        return
    # creiamo un'istanza della classe Autoencoder:
    autoencoder = Autoencoder()
    somma=0
    lunghezza=0
    i = random.randint(0, len(percorso)-1)
    # importo i dati che indicano un flusso benigno:
    data_path = f"{percorso[i]}/benign_traffic.csv"
    # si legge tramite la libreria Pandas i dati (che si presentano
    # in formato csv):
    data = pd.read_csv(data_path)
    # Di questo dataset prendiamo solo il 10% per la fase di addestramento:
    data = data.sample(frac=0.1, random_state=42)
    # standardizziamo i dati. Questo processo è molto utile al fine di portare
    # tutti i dati che si trovano su scale differenti alla medesima scala
    # al fine di essere trattati allo stesso modo durante l'analisi
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    # La variabile "data_normalized" contiene i dati standardizzati

    # trasformiamo i dati in tensori torch:
    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)


    #OTTENIAMO I MIGLIORI PARAMETRI DI LR E EPOCHE PER L'AUTOENCODER
    #IN MODO CHE SI OTTIMIZZI LA CAPACITA' DI RICOSTRUZIONE DELLA RETE
    #NEURALE:
    stimaMiglioriParametri(data_tensor)



    # usiamo come cost funcion la mean square error:
    criterion = nn.MSELoss()
    # usiamo ottimizatore "Adam" con i parametri sotto mostrati:
    optimizer = optim.Adam(autoencoder.parameters(), lr=best_lr)
    #optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    num_epochs = best_epochs
        #num_epochs = 150
        # memorizziamo in un array "losses" tutti gli errori di ricostruzione
        # nelle varie epoche (questo vettore, per come abbiamo scritto il codice,
        # conterrà num_epochs elementi). Questo vettore lo useremo per rappresentare
        # tramite la libreria matpltlib il suo grafico: ci si augura che tale grafico
        # mostri un andamento decrescente:
    losses = []

        # addestramento della rete neurale:
    for epoch in range(num_epochs):
            # calcoliamo il risultato di ricostruzione dalla rete autoencoder:
            outputs = autoencoder(data_tensor)
            # determiniamo l'errore di Ricostruzione:
            loss = criterion(outputs, data_tensor)
            # lo aggiungiamo alla lista discussa sopra:
            losses.append(loss.item())
            # tramite l'ottimizzatore ottimizziamo i parametri della rete:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # eseguiamo una stampa per mostrare l'errore correntemente calcolato:
            #print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    somma += sum(losses[-int(num_epochs*0.6):])#1
    lunghezza += len(losses[-int(num_epochs*0.6):]) #1

    # alla fine della fase di training della rete calcoliamo questo valore
    # di soglia che ci servirà per classificare le anomalie. Usiamo la deviazione
    # standard per effettuare il calcolo. In particolare andiamo a settare questo
    # valore pari ad un moltiplo della deviazione standard sopra la media calcolata:

    #WARN:
    #quantity = int(num_epochs * 0.6)
    media = somma / lunghezza
    # il multiplo di deviazione usata:
    #print("media", media)
    multiplo = 2
    deviazione = np.std(losses)
    threshold = media + multiplo * deviazione
    #print("valore soglia", threshold)
    #print("threshold", threshold)

    # A questo punto non si vuole usare direttamente il valore di sogli appena
    # calcolato per rilevare anomalie, ma si decide di addestrare la seconda
    # rete neurale che apprenda come riconoscere, dato un valore di soglia e
    # l'errore attuale se si sta valutando il modello in condizioni di
    # anomalie della rete o in condizioni di traffico normale:
    prepara_dati_addestramento(percorso, autoencoder, criterion, threshold, scaler)
    results = []


    #SI MANTENGONO LE STRUTTURE DATI PER MEMORIZZARE I DATI PER MEMORIZZARE LE
    #MISURE DI VALUTAZIONE:
    AUC_ROC_results = np.array([])
    AUC_PR_results = np.array([])
    F1_results = np.array([])
    TPR_results = np.array([])
    FPR_results = np.array([])


    # test
    for j in range(len(percorso)):

        num_trials = 2 #20  #3 sarebbe meglio


        average_loss = 0.0

        for _ in range(num_trials):
            # attacchi presenti nel dataset:
            attacks = ["combo", "udp", "tcp", "junk", "scan"]
            miraiAttacks = ["ack", "scan", "syn", "udp", "udpplain"]

            num_trials = 1#10

            anomaly_count = 0

            corrected = np.array([])
            predicted = np.array([])
            correctedPositive = np.array([])
            predictedPositive = np.array([])

            for i in range(num_trials):

                for attack in attacks:
                    # prendiamo il dataset relativo allo specifico attacco:
                    data_path = f"{percorso[j]}/gafgyt_attacks/{attack}.csv"
                    # si leggono e normalizzano i dati tramite la libreria pandas:
                    data = pd.read_csv(data_path)
                    data = data.sample(frac=0.3, random_state=i)
                    data_normalized = scaler.transform(data)
                    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
                    # si determina il risultato e l'errore di RICOSTRUZIONE del modello:
                    outputs = autoencoder(data_tensor)
                    loss = criterion(outputs, data_tensor)

                    # essendo questo dataset un dataset che illustra valori di traffico
                    # affetto da una qualche anomalia il corretto risultato della
                    # classificazione deve portare a riconoscere l'anomalia, quindi
                    # il valore i-esimo della lista "corrected" deve contenere uno:
                    corrected = np.append(corrected, 1)

                    ris = testSecondaRetethreshold(threshold, loss)
                    if (ris >= 0.5):
                        #print(f"Anomaly detected in {attack} attack!")
                        anomaly_count += 1
                        # è stata riconosciuta l'anomalia:
                        predicted = np.append(predicted, 1)
                    else:
                        print(f"Anomaly NOT detected in {attack} attack!")
                        predicted = np.append(predicted, 0)

                for attack in miraiAttacks:
                    # prendiamo il dataset relativo allo specifico attacco:
                    data_path = f"{percorso[j]}/mirai_attacks/{attack}.csv"
                    # si leggono e normalizzano i dati tramite la libreria pandas:
                    data = pd.read_csv(data_path)
                    data = data.sample(frac=0.3, random_state=i+1)
                    data_normalized = scaler.transform(data)
                    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
                    # si determina il risultato e l'errore di RICOSTRUZIONE del modello:
                    outputs = autoencoder(data_tensor)
                    loss = criterion(outputs, data_tensor)

                    # essendo questo dataset un dataset che illustra valori di traffico
                    # affetto da una qualche anomalia il corretto risultato della
                    # classificazione deve portare a riconoscere l'anomalia, quindi
                    # il valore i-esimo della lista "corrected" deve contenere uno:
                    corrected = np.append(corrected, 1)

                    ris = testSecondaRetethreshold(threshold, loss)
                    if (ris >= 0.5):
                        #print(f"Mirai Anomaly detected in {attack} attack!")
                        anomaly_count += 1
                        # è stata riconosciuta l'anomalia:
                        predicted = np.append(predicted, 1)
                    else:
                        print(f"Mirai Anomaly NOT detected in {attack} attack!")
                        predicted = np.append(predicted, 0)

                # a questo punto, si mostra il "comportamento" della rete nel riconoscere un
                # traffico di rete benigno:
                data_path = f"{percorso[j]}/benign_traffic.csv"
                data = pd.read_csv(data_path)
                data = data.sample(frac=0.3, random_state=i+2)
                data_normalized = scaler.transform(data)
                data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
                outputs = autoencoder(data_tensor)
                loss = criterion(outputs, data_tensor)
                correctedPositive = np.append(correctedPositive, 0)
                corrected = np.append(corrected, 0)
                ris = testSecondaRetethreshold(threshold, loss)
                if ris >= 0.5:
                    # errore nella predizione di rete:
                    anomaly_count += 1
                    predictedPositive = np.append(predictedPositive, 1)
                    predicted = np.append(predicted, 1)
                    print("FALSE POSITIVE!!!")
                else:
                    predictedPositive = np.append(predictedPositive, 0)
                    predicted = np.append(predicted, 0)
                    #print("TRUE NEGATIVE")

            FP = 0
            FN = 0
            TP = 0
            TN = 0
            for i in range(len(corrected)):
                if corrected[i] == 1 and predicted[i] == 1:
                    TP += 1
                if corrected[i] == 1 and predicted[i] == 0:
                    FN += 1
                if corrected[i] == 0 and predicted[i] == 0:
                    TN += 1
                if corrected[i] == 0 and predicted[i] == 1:
                    FP += 1
            TPR = round(TP / (TP + FN), 2)
            FPR = round(FP / (FP + TN), 2)
            # si calcolano i valodi di auc e f-measure:
            auc_roc = roc_auc_score(corrected, predicted)
            print("AUC-ROC:", auc_roc)
            f1 = f1_score(corrected, predicted)
            print("F-measure:", f1)
            #print(f"Anomaly count for {num_trials} trials: {anomaly_count}")
            # auc-pr:
            precision, recall, _ = precision_recall_curve(corrected, predicted)
            auc_pr = auc(recall, precision)
            print("AUC-PR:", auc_pr)
            average_loss += anomaly_count
            precision, recall, _ = precision_recall_curve(corrected, predicted)
            auc_pr = auc(recall, precision)
            print("AUC-PR:", auc_pr)

            # aggiungo il record contenente l'attuale test mediato su 10 prove:
            results.append({
                'AUC-ROC': auc_roc,
                'AUC-PR': auc_pr,
                'F1-Score': f1,
                'TPR': TPR,
                'FPR': FPR,
                'iterazioni': num_trials,
                'mixed_data': 'Sì'
            })

            TPR_results = np.append(TPR_results, TPR)
            FPR_results = np.append(FPR_results, FPR)
            AUC_ROC_results = np.append(AUC_ROC_results, auc_roc)
            F1_results = np.append(F1_results, f1)
            AUC_PR_results = np.append(AUC_PR_results, auc_pr)

            # Creo un DataFrame con Pandas che contenga i risultati:
    result_df = pd.DataFrame(results)
    filename = 'trial_results_NOTCOMPLETE_naiive_{}_{}.txt'.format(len(percorso), j)
    # Salvo il DataFrame in un file
    with open(filename, 'w') as f:
        headers = result_df.columns.tolist()
        formatted_headers = ["{:^20}".format(header) for header in headers]

        f.write('\t'.join(formatted_headers) + '\n')

        for _, row in result_df.iterrows():
            formatted_row = ["{:^20}".format(str(item)) for item in row]
            f.write('\t'.join(formatted_row) + '\n')
            # result_df.to_csv(filename, sep='\t', index=False)


    #OLTRE al file illustrato sopra, memorizzo le medie e le deviazioni standard delle misure delle
    #valutazioni:
    # media e deviazione standard delle misure di valutazione:
    auc_roc_mean = round(np.mean(AUC_ROC_results), 2)
    auc_roc_std = round(np.std(AUC_ROC_results), 2)
    auc_pr_mean = round(np.mean(AUC_PR_results), 2)
    auc_pr_std = round(np.std(AUC_PR_results), 2)
    f1_mean = round(np.mean(F1_results), 2)
    f1_std = round(np.std(F1_results), 2)
    tpr_mean = round(np.mean(TPR_results), 2)
    tpr_std = round(np.std(TPR_results), 2)
    fpr_mean = round(np.mean(FPR_results), 2)
    fpr_std = round(np.std(FPR_results), 2)

    risultatiMedie = []

    risultatiMedie.append({
        'media AUC-ROC': auc_roc_mean,
        'std AUC-ROC': auc_roc_std,
        'media AUC-PR': auc_pr_mean,
        'std AUC-PR': auc_pr_std,
        'media F1-Score': f1_mean,
        'std F1-Score': f1_std,
        'media FPR': fpr_mean,
        'std FPR': fpr_std,
        'media TPR': tpr_mean,
        'std TPR': tpr_std,
    })

    result_df = pd.DataFrame(risultatiMedie)
    base_nome = 'risultati_ImplUnica_medie_'+str(len(percorso))+'.txt'
    # Salvo il DataFrame in un file
    with open(base_nome, 'w') as f:
        headers = result_df.columns.tolist()
        formatted_headers = ["{:^20}".format(header) for header in headers]

        f.write('\t'.join(formatted_headers) + '\n')

        for _, row in result_df.iterrows():
            formatted_row = ["{:^20}".format(str(item)) for item in row]
            f.write('\t'.join(formatted_row) + '\n')



def unifica_dati_anomali(percorso):
    autoencoder = Autoencoder()
    somma = 0
    lunghezza = 0
    #scengo la metà dei dstaset su cui fare l'addestramento
    lista_indici = []
    meta = math.ceil(len(percorso)-1)
    print(meta)
    while len(lista_indici)<meta:
        i = random.randint(0, len(percorso) - 1)
        if i not in lista_indici:
            lista_indici.append(i)

    for i in lista_indici:
        data_path = f"{percorso[i]}/benign_traffic.csv"
        data = pd.read_csv(data_path)
        casuale = random.randint(10, 100)
        data = data.sample(frac=0.1, random_state=casuale)
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
        data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
        stimaMiglioriParametri(data_tensor)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=best_lr)
            # optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
        num_epochs = best_epochs
            # num_epochs = 150
        losses = []

        # addestramento della rete neurale:
        for epoch in range(num_epochs):
                outputs = autoencoder(data_tensor)
                loss = criterion(outputs, data_tensor)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        somma += sum(losses[-int(num_epochs* 1):])#1 (meglio uno
        lunghezza += len(losses[-int(num_epochs* 1):]) #1

    media = somma / lunghezza
    multiplo = 2
    deviazione = np.std(losses)
    threshold = media + multiplo * deviazione
    prepara_dati_addestramento(percorso, autoencoder, criterion, threshold, scaler)
    results = []

    attacks = ["combo", "udp", "tcp", "junk", "scan"]
    miraiAttacks = ["ack", "scan", "syn", "udp", "udpplain"]
    # test

    # SI MANTENGONO LE STRUTTURE DATI PER MEMORIZZARE I DATI PER MEMORIZZARE LE
    # MISURE DI VALUTAZIONE:
    AUC_ROC_results = np.array([])
    AUC_PR_results = np.array([])
    F1_results = np.array([])
    TPR_results = np.array([])
    FPR_results = np.array([])

    corrected = np.array([])
    predicted = np.array([])

    num_mixedTrial = 1
    for _ in range(num_mixedTrial):
        for j in range(len(percorso)):

            lista_dati_anomali = []
            for attack in attacks:
                data_path = f"{percorso[j]}/gafgyt_attacks/{attack}.csv"
                data = pd.read_csv(data_path)
                casuale = random.randint(10, 100)
                sampled_data = data.sample(frac=0.1, random_state=casuale)
                lista_dati_anomali.append(sampled_data)
            if percorso[j] != 'D:/dataset and/Ennio_Doorbell' and percorso[j] != ('D:/dataset and/Samsung_SNH_1011_N_Webc'
                                                                                  ''
                                                                                  ''
                                                                                  'am'):
                for attack in miraiAttacks:
                    data_path = f"{percorso[j]}/mirai_attacks/{attack}.csv"
                    data = pd.read_csv(data_path)
                    casuale = random.randint(10, 100)
                    sampled_data = data.sample(frac=0.1, random_state=casuale)
                    lista_dati_anomali.append(sampled_data)

            combined_data = pd.concat(lista_dati_anomali)
            # Salva il DataFrame combinato in un file
            combined_data.to_csv('lista_dati_anomali.csv', index=False)
            data = pd.read_csv('lista_dati_anomali.csv')
            casuale = random.randint(10, 100)
            data = data.sample(frac=0.1, random_state=casuale)
            data_normalized = scaler.transform(data)
            data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
            outputs = autoencoder(data_tensor)
            loss = criterion(outputs, data_tensor)

            corrected = np.append(corrected, 1)

            ris = testSecondaRetethreshold(threshold, loss)
            if (ris >= 0.5):
                # è stata riconosciuta l'anomalia DEI DATI COMBINATI:
                print(f"Anomaly COMPLETE detected correctly")
                predicted = np.append(predicted, 1)
            else:
                print(f"Anomaly COMPLETE NOT detected")
                predicted = np.append(predicted, 0)

    #unifico dati corretti:
    lista_dati_corretti = []
    for j in range(len(percorso)):
        data_path = f"{percorso[j]}/benign_traffic.csv"
        data = pd.read_csv(data_path)
        sample_size = int(len(data) * 0.1)
        sampled_data = data.sample(n=sample_size, random_state=j)
        lista_dati_corretti.append(sampled_data)

    combined_data = pd.concat(lista_dati_corretti)
    # Salva il DataFrame combinato in un file
    combined_data.to_csv('lista_dati_corretti.csv', index=False)
    data = pd.read_csv('lista_dati_corretti.csv')
    #prendo il 30% IN MODO CASUALE:
    data = data.sample(frac=0.1, random_state=10)
    data_normalized = scaler.transform(data)
    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
    outputs = autoencoder(data_tensor)
    loss = criterion(outputs, data_tensor)

    corrected = np.append(corrected, 0)

    ris = testSecondaRetethreshold(threshold, loss)
    if (ris >= 0.5):
        # è stata riconosciuta l'anomalia DEI DATI COMBINATI:
        print(f"FALSE POSITIVE COMPLETE")
        predicted = np.append(predicted, 1)
    else:
        print(f"True negative COMPLETE")
        predicted = np.append(predicted, 0)

    FP = 0
    FN = 0
    TP = 0
    TN = 0
    for i in range(len(corrected)):
        if corrected[i] == 1 and predicted[i] == 1:
            TP += 1
        if corrected[i] == 1 and predicted[i] == 0:
            FN += 1
        if corrected[i] == 0 and predicted[i] == 0:
            TN += 1
        if corrected[i] == 0 and predicted[i] == 1:
            FP += 1
    TPR = round(TP / (TP + FN), 2)
    FPR = round(FP / (FP + TN), 2)
    auc_roc = roc_auc_score(corrected, predicted)
    f1 = f1_score(corrected, predicted)
    precision, recall, _ = precision_recall_curve(corrected, predicted)
    auc_pr = auc(recall, precision)
    precision, recall, _ = precision_recall_curve(corrected, predicted)
    results.append({
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'F1-Score': f1,
        'TPR': TPR,
        'FPR': FPR,
        'iterazioni': '2',
        'mixed_data': 'Sì'
    })

    TPR_results = np.append(TPR_results, TPR)
    FPR_results = np.append(FPR_results, FPR)
    AUC_ROC_results = np.append(AUC_ROC_results, auc_roc)
    F1_results = np.append(F1_results, f1)
    AUC_PR_results = np.append(AUC_PR_results, auc_pr)

#t
    for j in range(len(percorso)):

        num_trials = 1

        average_loss = 0.0
        corrected = np.array([])
        predicted = np.array([])
        anomaly_count = 0

        for i in range(num_trials):
            data_path = f"{percorso[j]}/benign_traffic.csv"
            data = pd.read_csv(data_path)
            casuale = random.randint(10,100)
            data = data.sample(frac=0.3, random_state=casuale)
            data_normalized = scaler.transform(data)
            data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
            outputs = autoencoder(data_tensor)
            loss = criterion(outputs, data_tensor)
            corrected = np.append(corrected, 0)
            ris = testSecondaRetethreshold(threshold, loss)
            if ris >= 0.5:
                # errore nella predizione di rete:
                anomaly_count += 1
                predicted = np.append(predicted, 1)
                print("FALSE POSITIVE!!!")
            else:
                predicted = np.append(predicted, 0)
                # print("TRUE NEGATIVE")

            for attack in attacks:
                data_path = f"{percorso[j]}/gafgyt_attacks/{attack}.csv"
                data = pd.read_csv(data_path)
                casuale = random.randint(10, 100)
                data = data.sample(frac=0.3, random_state=casuale)
                data_normalized = scaler.transform(data)
                data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
                outputs = autoencoder(data_tensor)
                loss = criterion(outputs, data_tensor)
                corrected = np.append(corrected, 1)
                ris = testSecondaRetethreshold(threshold, loss)
                if ris >= 0.5:
                    # errore nella predizione di rete:
                    anomaly_count += 1
                    predicted = np.append(predicted, 1)
                    #print("TRUE POSITIVE!!!")
                else:
                    predicted = np.append(predicted, 0)
                    print("FALSE NEGATIVE!!!!")


            # Si controlla non siano dispositivi senza dataset di attacchi di botnet di tipo MIRAI:
            if percorso[j] != 'D:/dataset and/Ennio_Doorbell' and percorso[j] != 'D:/dataset and/Samsung_SNH_1011_N_Webcam':
                for attack in miraiAttacks:
                    data_path = f"{percorso[j]}/mirai_attacks/{attack}.csv"
                    data = pd.read_csv(data_path)
                    casuale = random.randint(10, 100)
                    data = data.sample(frac=0.3, random_state=casuale)
                    data_normalized = scaler.transform(data)
                    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
                    outputs = autoencoder(data_tensor)
                    loss = criterion(outputs, data_tensor)
                    corrected = np.append(corrected, 1)
                    ris = testSecondaRetethreshold(threshold, loss)
                    if ris >= 0.5:
                        # errore nella predizione di rete:
                        anomaly_count += 1
                        predicted = np.append(predicted, 1)
                        #print("FALSE POSITIVE!!!")
                    else:
                        predicted = np.append(predicted, 0)
                        print("FALSE NEGATIVE!!!")

        # si calcolano i valodi di auc e f-measure:
        FP = 0
        FN = 0
        TP = 0
        TN = 0
        for i in range(len(corrected)):
            if corrected[i] == 1 and predicted[i] == 1:
                TP += 1
            if corrected[i] == 1 and predicted[i] == 0:
                FN += 1
            if corrected[i] == 0 and predicted[i] == 0:
                TN += 1
            if corrected[i] == 0 and predicted[i] == 1:
                FP += 1
        TPR = round(TP / (TP + FN), 2)
        FPR = round(FP / (FP + TN), 2)
        auc_roc = roc_auc_score(corrected, predicted)
        print("AUC-ROC:", auc_roc)

        f1 = f1_score(corrected, predicted)
        print("F-measure:", f1)
        # print(f"Anomaly count for {num_trials} trials: {anomaly_count}")
        # auc-pr:
        precision, recall, _ = precision_recall_curve(corrected, predicted)
        auc_pr = auc(recall, precision)
        print("AUC-PR:", auc_pr)
        average_loss += anomaly_count
        precision, recall, _ = precision_recall_curve(corrected, predicted)
        auc_pr = auc(recall, precision)
        print("AUC-PR:", auc_pr)

        # aggiungo il record contenente l'attuale test mediato su 10 prove:
        results.append({
                'AUC-ROC': auc_roc,
                'AUC-PR': auc_pr,
                'F1-Score': f1,
                'TPR': TPR,
                'FPR': FPR,
                'iterazioni': num_trials,
                'mixed_data': 'No'
        })

        TPR_results = np.append(TPR_results, TPR)
        FPR_results = np.append(FPR_results, FPR)
        AUC_ROC_results = np.append(AUC_ROC_results, auc_roc)
        F1_results = np.append(F1_results, f1)
        
        AUC_PR_results = np.append(AUC_PR_results, auc_pr)

    # Creo un DataFrame con Pandas che contenga i risultati:
    result_df = pd.DataFrame(results)
    filename = 'trial_results_completo_naiive_TRAIN_HALF_{}.txt'.format(len(percorso))
    # Salvo il DataFrame in un file
    with open(filename, 'w') as f:
        headers = result_df.columns.tolist()
        formatted_headers = ["{:^20}".format(header) for header in headers]

        f.write('\t'.join(formatted_headers) + '\n')

        for _, row in result_df.iterrows():
            formatted_row = ["{:^20}".format(str(item)) for item in row]
            f.write('\t'.join(formatted_row) + '\n')
            # result_df.to_csv(filename, sep='\t', index=False)

#OLTRE al file illustrato sopra, memorizzo le medie e le deviazioni standard delle misure delle
    #valutazioni:
    # media e deviazione standard delle misure di valutazione:
    auc_roc_mean = round(np.mean(AUC_ROC_results), 2)
    auc_roc_std = round(np.std(AUC_ROC_results), 2)
    auc_pr_mean = round(np.mean(AUC_PR_results), 2)
    auc_pr_std = round(np.std(AUC_PR_results), 2)
    f1_mean = round(np.mean(F1_results), 2)
    f1_std = round(np.std(F1_results), 2)
    tpr_mean = round(np.mean(TPR_results), 2)
    tpr_std = round(np.std(TPR_results), 2)
    fpr_mean = round(np.mean(FPR_results), 2)
    fpr_std = round(np.std(FPR_results), 2)

    risultatiMedie = []

    risultatiMedie.append({
        'media AUC-ROC': auc_roc_mean,
        'std AUC-ROC': auc_roc_std,
        'media AUC-PR': auc_pr_mean,
        'std AUC-PR': auc_pr_std,
        'media F1-Score': f1_mean,
        'std F1-Score': f1_std,
        'media FPR': fpr_mean,
        'std FPR': fpr_std,
        'media TPR': tpr_mean,
        'std TPR': tpr_std,
    })

    result_df = pd.DataFrame(risultatiMedie)
    variabile = len(percorso)
    filename = 'risultati_ImplUnica_medie2_TRAIN_HALF_'+str(variabile)+'.txt'
    # Salvo il DataFrame in un file
    with open(filename, 'w') as f:
        headers = result_df.columns.tolist()
        formatted_headers = ["{:^20}".format(header) for header in headers]

        f.write('\t'.join(formatted_headers) + '\n')

        for _, row in result_df.iterrows():
            formatted_row = ["{:^20}".format(str(item)) for item in row]
            f.write('\t'.join(formatted_row) + '\n')






basePercorso = "D:/dataset and"
#print("\nora 1\n0")
#listaPercorsi = [f"{basePercorso}/Danmini_Doorbell"]


#print("\nora 1:\n")
#listaPercorsi = [f"{basePercorso}/Danmini_Doorbell"]
#buildAutoencoder(listaPercorsi, True)


print("\nora 2:\n")
listaPercorsi = [f"{basePercorso}/Danmini_Doorbell",f"{basePercorso}/Samsung_SNH_1011_N_Webcam"]
buildAutoencoder(listaPercorsi, True)


#print("\nora 3:\n")
#listaPercorsi = [f"{basePercorso}/Danmini_Doorbell", f"{basePercorso}/Ecobee_Thermostat", f"{basePercorso}/Philips_B120N10_Baby_Monitor"]
#buildAutoencoder(listaPercorsi, True)

#print("\nora 4:\n")
#listaPercorsi = [f"{basePercorso}/Danmini_Doorbell", f"{basePercorso}/Ecobee_Thermostat", f"{basePercorso}/Philips_B120N10_Baby_Monitor", f"{basePercorso}/Provision_PT_737E_Security_Camera"]
#buildAutoencoder(listaPercorsi, True)

#print("\nora 5:\n")
#listaPercorsi = [f"{basePercorso}/Danmini_Doorbell", f"{basePercorso}/Ecobee_Thermostat", f"{basePercorso}/Philips_B120N10_Baby_Monitor", f"{basePercorso}/Provision_PT_737E_Security_Camera", f"{basePercorso}/Provision_PT_838_Security_Camera"]
#buildAutoencoder(listaPercorsi, True)

#print("\nora 6:\n")
#listaPercorsi = [f"{basePercorso}/Danmini_Doorbell", f"{basePercorso}/Ecobee_Thermostat", f"{basePercorso}/Philips_B120N10_Baby_Monitor",f"{basePercorso}/SimpleHome_XCS7_1003_WHT_Security_Camera", f"{basePercorso}/Ennio_Doorbell", f"{basePercorso}/Provision_PT_838_Security_Camera"]
#buildAutoencoder(listaPercorsi, True)

print("\nora 7:\n")
listaPercorsi = [f"{basePercorso}/Danmini_Doorbell", f"{basePercorso}/Ecobee_Thermostat", f"{basePercorso}/Philips_B120N10_Baby_Monitor", f"{basePercorso}/SimpleHome_XCS7_1003_WHT_Security_Camera", f"{basePercorso}/SimpleHome_XCS7_1002_WHT_Security_Camera", f"{basePercorso}/Ennio_Doorbell", f"{basePercorso}/Provision_PT_838_Security_Camera"]
buildAutoencoder(listaPercorsi, True)

print("\nora 8:\n")
listaPercorsi = [f"{basePercorso}/Danmini_Doorbell", f"{basePercorso}/Ecobee_Thermostat", f"{basePercorso}/Philips_B120N10_Baby_Monitor", f"{basePercorso}/SimpleHome_XCS7_1003_WHT_Security_Camera", f"{basePercorso}/SimpleHome_XCS7_1002_WHT_Security_Camera", f"{basePercorso}/Ennio_Doorbell", f"{basePercorso}/Provision_PT_838_Security_Camera", f"{basePercorso}/Provision_PT_737E_Security_Camera"]
buildAutoencoder(listaPercorsi, True)

print("\nora 9:")
listaPercorsi = [f"{basePercorso}/Danmini_Doorbell", f"{basePercorso}/Ecobee_Thermostat", f"{basePercorso}/Philips_B120N10_Baby_Monitor", f"{basePercorso}/SimpleHome_XCS7_1003_WHT_Security_Camera", f"{basePercorso}/SimpleHome_XCS7_1002_WHT_Security_Camera", f"{basePercorso}/Ennio_Doorbell", f"{basePercorso}/Provision_PT_838_Security_Camera", f"{basePercorso}/Provision_PT_737E_Security_Camera", f"{basePercorso}/Samsung_SNH_1011_N_Webcam"]


buildAutoencoder(listaPercorsi, True)
#buildAutoencoder(listaPercorsi, False)








