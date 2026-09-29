# Parameter / Training
* batch_size=5, epochs=100, epoch_length=10, learning_rate=1e-4
* Evaluation nach 5 Iterationen
* Laufzeit: ca. 7 Stunden für 215 Iterationen

# Lernerfolg
Der Loss sinkt von Anfangs > 0.9 auf einen Bereich zwischen 0.55 und 0.70 nach über 200 Iterationen.
Ein genereller Abwärtstrend ist erkennbar, auch wenn dieser fluktuiert und stagniert.

# Qualität
Der Validierungs-Dice ist konstant schlecht. Ein Maximum von 0.0503 wurde nach 190 Iterationen erreicht.
Im Vergleich zum Start (0.0068) ist das zwar sehr gut, aber dennoch insgesamt sehr schlecht.

# Verlauf
Starker Start mit relativ gesehen starken Verbesserungen. Dann lange keine spürbare Verbesserung im Dice. Ab Iteration 175 wieder ein verstärkter Anstieg im Dice, anschließend wieder Stagnation.

# Klassenweise Betrachtung
Keine Klasse wird besonders gut segmentiert (Beispiel bei Iteration 190: [0.004581678134854883, 0.08091930085076736, 0.11467958571246037, 4.8521508242279546e-05]). Besonders schlecht schneidet Klasse 4 (chondro_osseous_border) ab. Diese ist nicht in allen Volumina vorhanden und besteht aus kleinen Strukturen, das muss aber nicht zwingend der Grund für das Abschneiden sein.

# Bewertung
Positiv: Es funktioniert
Der Dice ist aber sehr verbesserungswürdig, bislang ist lediglich gezeigt, dass hier ein mögliches Potential weiter erkundet werden muss.

# Mögliche weitere Schritte
Verschiedene Parameter anpassen und Variationen austesten:
* Segmentierungskopf (ist: Linear)
* Batch Size (ist: 5)
* Zahl der Epochen (ist: 100)
* Länge der Epochen (ist: 10)
* Lernrate (ist: 1e-4)
* *Validierungsiterationen (ist: 5), ändert nichts an den Ergebnissen, die Validierung braucht aber mit Abstand am längsten, hierüber liese sich also steuern, wie schnell höhere Iterationen erreicht werden*
* Parameter des DiceLoss anpassen (ist: include_background=False, to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-5)
* Weitere / andere Augmentierungen implementieren (dinov2/eval/segmentation_3d/augmentations.py)

# Fazit
Die Ergebnisse sind nicht befriedigend, aber immerhin scheint 3DINO mit den Daten arbeiten zu können. Es gibt noch viele Parameter, die angepasst und getestet werden können, daher ist es noch zu früh, um "für diesen Datensatz auf jeden Fall nicht sinnvoll" sagen zu können.
