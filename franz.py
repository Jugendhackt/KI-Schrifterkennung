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

testdaten = len(Y_test)

# alle prozentzahlen werden gespeichert (Y Achse des Graphen)
prozent_svc_ges = []
prozent_nn_ges = []

x_ticks = []
# es kommen immer mehr Daten dazu, die zum lernen benutzt werden
for lerndaten in range(1500):

    if lerndaten == 0 or lerndaten == 1 or lerndaten % 5 != 2:
        pass

    else:
        print("Iteration %d von %d" % (lerndaten + 1, 1500))
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
        y_svc = clf_svc.predict(X_test)
        y_nn = clf_mlp.predict(X_test)

        richtig_svc = 0
        richtig_nn = 0

        # alle Output Zahlen werden mit der richtigen Lösung verglichen
        # dadurch wird eine Fehlerquote erstelt(siehe unten)
        for i in range(testdaten):
            if y_svc[i] == Y_test[i]:
                richtig_svc += 1.

        for i in range(testdaten):
            if y_nn[i]:
                richtig_nn += 1.
        # Der Prozentsatz wird ausgerechnet und gespeichert
        prozent_svc = 100. * richtig_svc / testdaten
        prozent_svc_ges.append(prozent_svc)

        prozent_nn = 100. * richtig_nn / testdaten
        prozent_nn_ges.append(prozent_nn)

        x_ticks.append(lerndaten)

plt.title("Erkennungsrate")
plt.xlabel("Anzahl der Lerndaten")
plt.ylabel("Trefferquote")
plt.xticks()
plt.ylim(0, 100)
plt.plot(x_ticks, prozent_svc_ges, label="svc")
plt.plot(x_ticks, prozent_nn_ges, label="nn")
plt.legend()
plt.show()
