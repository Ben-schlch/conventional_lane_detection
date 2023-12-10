# Lane Detection

Dieses Dokument beschreibt die Funktionen und die Verwendung des Lane Detection Objekts.

## \_\_init\_\_

    ```python
    def __init__(self)
    ```
Die Initialisierung des Lane Detection Objekts. Hierbei werden die Konfigurationen geladen und einige Werte initialisiert.

## detect_lane

    ```python
    def detect_lane(self, image: Mat) -> Mat
    ```
Diese Funktion nimmt ein Bild entgegen und gibt ein Bild zurück, auf dem die erkannten Spuren eingezeichnet sind. 
Sie ruft dabei alle anderen Funktionen auf, die für die Erkennung der Spuren benötigt werden.

## run
    ```python
    def run(self) -> np.ndarray
    ```
Die run Funktion ist die Funktion, die von außen aufgerufen wird. Sie startet die producer und consumer threads, 
die dafür zuständig sind das Laden der Bilder und das Erkennen der Spuren zu parallelisieren. Sie misst außerdem auch 
die Zeit und die Bilder pro Sekunde, die das Objekt verarbeitet.

## \_producer

    ```python
    def _producer(self)
    ```
Die Producer Funktion läuft in einem eigenen Thread und ist dafür zuständig, die Bilder in die Queue zu laden,
die von der detect_lane Funktion abgearbeitet werden.

## \_consumer

    ```python
    def _consumer(self, batch_size = 8)
    ```
Die Consumer Funktion läuft in einem eigenen Thread und ist dafür zuständig, die Bilder aus der Queue zu laden,
die von der detect_lane Funktion abgearbeitet werden. Dabei kann man die Anzahl der Bilder, die gleichzeitig verarbeitet 
werden, über die Konfiguration einstellen. In diesem Projekt ist diese Anzahl auf 1 gesetzt, da die Erkennung der Spuren
sequenziell abgearbeitet werden muss.

## fit\_lane\_lines

    ```python
    def fit_lane_lines(self, image: np.ndarray) -> np.ndarray
    ```

