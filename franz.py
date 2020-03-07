# Datenset, Neuronale Netze, andere KI's importieren

import sklearn.datasets
import sklearn.neighbors
import sklearn.svm
import sklearn.neural_network

import matplotlib.pyplot as plt
import numpy as np

# daten speichern
digits = sklearn.datasets.load_digits()


# Bilder speichern und Eindimensional machen
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Testdaten sind die Bilder von 1500 bis ende
# mit den dazugehörigen Vergleichsnummern (Y_test)
X_test = data[1500:]
Y_test = digits.target[1500:]

# alle prozentzahlen werden gespeichert (Y Achse des Graphen)
prozent_ges = []

# es kommen immer mehr Daten dazu, die zum lernen benutzt werden
for lerndaten in range(1500):
    #?
    if lerndaten == 0 or lerndaten == 1:
        pass

    else:
        
        # KI's erzeugen
        clf_svc = sklearn.svm.SVC(gamma=0.001)
        clf_mlp = sklearn.neural_network.MLPClassifier()
        
        X = data[:lerndaten]
        Y = digits.target[:lerndaten]

        # mit den vorhandenen Bildern wird gelernt
        clf_svc.fit(X, Y)
        clf_mlp.fit(X, Y)

        # Das Neuronale Netz wird an den Testdaten getestet
        # Jedem Bild wird eine Zahl zugeordnet
        y = clf_mlp.predict(X_test)

        richtig = 0

        # alle Output Zahlen werden mit der richtigen Lösung verglichen
        # dadurch wird eine Fehlerquote erstelt(siehe unten)
        for i in range(lerndaten-10):
            if y[i] == Y_test[i]:
                richtig += 1

        # Der Prozentsatz wird ausgerechnet und gespeichert
        prozent = richtig/lerndaten
        prozent_ges.append(prozent)


plt.show()

