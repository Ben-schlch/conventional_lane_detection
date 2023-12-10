# Lane Detection

## Explanation:
Main ist the main file of the project. It contains the LaneDetection class, which is responsible for the lane detection.
Main reads the input video and passes the frames to the LaneDetection class. The LaneDetection class then detects the lanes.   
In terms of what steps it are implemented inside the main.py, we can refer to our [Step by step Guide](step_by_step.ipynb). Main.py uses preprocess.py until [Step 8](step_by_step.ipynb) of our step by step visualization.

Afterwards the lanes are split, the polynomials are fitted, the curve radius is calculated and the lanes are drawn on the image.

For lanes after being drawn are converted back to the original perspective from the birds eye view and then the image is written to the output video.

One of the tricks we used to improve lane detection, is that we split the image in two halves and fit the polynom on each half separately. This is done in the function `_seperate_lane_lines_on_thresh`. This is done to group the lanes into left and right lane.

## Alternatives to our proposed main.py

We could have changed many things:
- We could have split the lanes according to a more intelligent way or by yellow vs white
- Polynomfitting:
- One big possible time improvement we did not try is to use Parallelization. Because we anyway split the image in half and detect one lane left and one right, we could use multiprocessing and do all the steps parallel.

## Funktionen:

### \_\_init\_\_

```python
def __init__(self)
```
Die Initialisierung des Lane Detection Objekts. Hierbei werden die Konfigurationen geladen und einige Werte initialisiert.

### detect_lane

```python
def detect_lane(self, image: Mat) -> Mat
```
Diese Funktion nimmt ein Bild entgegen und gibt ein Bild zurück, auf dem die erkannten Spuren eingezeichnet sind. 
Sie ruft dabei alle anderen Funktionen auf, die für die Erkennung der Spuren benötigt werden.

### run
```python
def run(self) -> np.ndarray
```
Die run Funktion ist die Funktion, die von außen aufgerufen wird. Sie startet die producer und consumer threads, 
die dafür zuständig sind das Laden der Bilder und das Erkennen der Spuren zu parallelisieren. Sie misst außerdem auch 
die Zeit und die Bilder pro Sekunde, die das Objekt verarbeitet.

### \_producer

```python
def _producer(self)
```
Die Producer Funktion läuft in einem eigenen Thread und ist dafür zuständig, die Bilder in die Queue zu laden,
die von der detect_lane Funktion abgearbeitet werden.

### \_consumer

```python
def _consumer(self, batch_size = 8):
```
Die Consumer Funktion läuft in einem eigenen Thread und ist dafür zuständig, die Bilder aus der Queue zu laden,
die von der detect_lane Funktion abgearbeitet werden. Dabei kann man die Anzahl der Bilder, die gleichzeitig verarbeitet 
werden, über die Konfiguration einstellen. In diesem Projekt ist diese Anzahl auf 1 gesetzt, da die Erkennung der Spuren
sequenziell abgearbeitet werden muss.

### fit\_lane\_lines

```python
def _fit_lane_lines(self, right_line: np.ndarray, left_line: np.ndarray) \
    -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
```

Diese Funktion nimmt die erkannten Punkte der Spuren entgegen und berechnet daraus die Koeffizienten der Polynome, 
sowohl für die pixelwerte, als auch für die Realwelt-Werte. Falls das Resultat des Kurvenfittings der Funktion 
`_fit_line` `None` ist, werden die Koefizienten der Polynome der vorherigen Iteration zurückgegeben. Das Polynom wird 
in abhängigkeit von y berechnet, da die Kurven im Bild meist gerade sind und die x-Werte für gerade Kurven nicht 
geeignet sind.

### _\_fit\_line

```python
def __fit_line(self, line: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
```
Diese Funktion übernimmt die eigentliche berechnung der Koeffizienten der Polynome. Dabei wird die Funktion
`np.polyfit` verwendet. Falls die Krümmung der Kurve zu stark ist, oder zu wenige Punkte enthalten sind, 
wird die Funktion nicht ausgeführt und es wird `None` zurückgegeben.

### \_draw\_lane\_lines

```python
def _draw_lines(self, image: Mat, right_line: np.ndarray, left_line: np.ndarray) -> Mat:
```
Diese Funktion nimmt ein Bild und die Koeffizienten der Polynome entgegen und zeichnet die Spuren auf das Bild.
Dabei wird die Funktion `cv2.fillPoly` verwendet und die Linien werden zurück in die Normale Perspektive transformiert.

### \_\_draw\_line
    
```python
def __draw_line(self, line: np.ndarray, y: np.ndarray, color: tuple[int, int, int]) -> Mat:
```
Diese Funktion nimmt die Koeffizienten der Polynome und die y-Werte entgegen und zeichnet die Spuren auf das Bild.


### seperate_lane_lines_on_thresh

```python
def _seperate_lane_lines_on_thresh(self, image: Mat) -> tuple[np.ndarray, np.ndarray]:
```
Diese Funktion nimmt ein Bild entgegen und gibt die erkannten Punkte der Spuren zurück. Dabei wird das Bild in zwei 
Hälften geteilt und für jede Hälfte die Punkte der Spuren zurückgegeben. Dabei müssen die Punkte etwas mehr links, bzw. 
rechts, als die Mitte des Bildes liegen, da in der Mitte der Fahrspur meistens störende Punkte liegen, die die
Erkennung der Spuren stören.

### \_calculate\_curvature

```python
def _calculate_curvature(self, fit_left_line, fit_right_line) -> float:
```
Diese Funktion nimmt die Koeffizienten der Polynome entgegen und berechnet die Krümmung der Kurve. Dabei wird die
Formel aus der Vorlesung verwendet. Die Werte für die Umrechnung von Pixelwerten in Realwelt-Werte wurden selber 
festgelegt.

