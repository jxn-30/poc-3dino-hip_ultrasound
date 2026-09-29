# 3D-Ultraschall-Segmentierung mit 3DINO

## Aufgabe / Experiment

In diesem Experiment wird untersucht, ob sich das Standard vortrainierte 3DINO-Modell für die Mehrklassen-Segmentierung von 3D-Ultraschallvolumina eignen kann. Segmentiert werden vier anatomische Vordergrundklassen:

1. `bone_cortex`
2. `femur_head`
3. `ilium`
4. `chondro_osseous_border`

Der Hintergrund wird bei der Dice-Berechnung nicht berücksichtigt.

## Methode

Als Grundlage dient ein vortrainierter 3DINO-ViT-Encoder. Der Encoder wurde eingefroren; trainiert wurde ausschließlich ein linearer Segmentierungskopf mit `channel_merge`, Batch Normalization und `conv_seg`.

Für das Training wurden 3D-Patches mit klassenbasiertem Cropping verwendet. Die Optimierung erfolgte mit Dice-Loss:

- `include_background=False`
- `to_onehot_y=True`
- `softmax=True`
- `squared_pred=True`
- `smooth_nr=0`
- `smooth_dr=1e-5`

## Durchführung

Verwendete Parameter:

- Batchgröße: 5
- Epochen: 100
- Epoch Length: 10
- Lernrate: `1e-4`
- Validierung: alle 5 Iterationen
- Trainings-, Validierungs- und Testdaten: 132 / 26 / 32 Volumina

Der Lauf wurde nach 215 Trainingsiterationen dokumentiert. Die Laufzeit betrug ungefähr sieben Stunden.

## Ergebnisse

Der Trainings-Loss sank insgesamt von ungefähr 0,91 zu Beginn auf etwa 0,55 bis 0,70 nach mehr als 200 Iterationen. Trotz deutlicher Schwankungen ist somit ein genereller abnehmender Trend erkennbar.

Der Validierungs-Dice stieg von 0,0068 in Iteration 0 auf einen bisherigen Bestwert von **0,0503 in Iteration 190**. Danach schwankte er; bei Iteration 210 betrug er 0,0463. Das Modell zeigt damit ein Lernsignal, erreicht aber noch keine für die Anwendung ausreichende Segmentierungsqualität.

Am Bestpunkt (Iteration 190) ergaben sich folgende Dice-Werte pro Vordergrundklasse:

| Klasse | Dice |
|---|---:|
| `bone_cortex` | 0,0046 |
| `femur_head` | 0,0809 |
| `ilium` | 0,1147 |
| `chondro_osseous_border` | 0,00005 |

Insbesondere `chondro_osseous_border` wird praktisch nicht segmentiert. Eine mögliche Ursache ist, dass diese kleine Klasse nicht in allen Volumina vorkommt und dadurch nur wenige Trainingsvoxel verfügbar sind. Dies sollte durch eine Klassenverteilungsanalyse überprüft werden.

## Bewertung und nächste Schritte

Der Versuch zeigt, dass der Trainingsablauf stabil funktioniert und der eingefrorene 3DINO-Encoder in Kombination mit einem linearen Kopf bereits ein begrenztes Lernsignal liefert. Die absolute Leistung ist jedoch noch niedrig.

Als nächste Schritte sollten insbesondere der beste Checkpoint aus Iteration 190 auf dem Testsatz evaluiert, Vorhersagen visuell mit den Ground-Truth-Labels verglichen und die Klassenverteilung im Datensatz analysiert werden. Anschließend können Sampling, Augmentierungen, Loss-Gewichtung, Trainingsparameter und die Architektur des Segmentierungskopfs gezielt variiert werden.

## Limitationen

Es wurde ein Jupyter-Notebook mit Python3.8 verwendet, während 3DINO für Python3.9 entwickelt wurde. Hierfür musste 3DINO beim Einlesen der Argumente leicht modifiziert werden. Auf das Training hat dies keinen Einfluss.
Der Jupyter-Server ist beim Training regelmäßig abgestürzt, weshalb das Experiment in eine Python-Datei ausgelagert und ohne Overhead / Wrapping durch Jupyter durchgeführt wurde.
