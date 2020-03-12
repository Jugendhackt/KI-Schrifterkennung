# Diese Datei ist nach dem Jugend Hackt Event entstanden
# Es wurde von Franz Schlicht programmieret
# Änderungen: EMNIST Dataset wird verwendet(auch Buchstaben)


from emnist import extract_training_samples
import numpy as np

# daten speichern
digits = np.array(extract_training_samples('digits'))
print(digits)

# daten eindemensional machen
n_samples = len(digits)
#data = digits.reshape(n_sampels, -1)
