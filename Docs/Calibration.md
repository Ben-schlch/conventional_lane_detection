# Calibration

Dieses Dokument beschreibt die Funktionen und die Verwendung des Camera Calibration Objekts.

## \_\_init\_\_

```python
def __init__(self,
             calib_path: str = '../images/Udacity/calib/',
             calib_config_path: str = '../images/Udacity/calib/config.pickle',
             warp_mat_path: str = '../images/Udacity/calib/warp_mat.pickle'):
```

Die Initialisierung des Camera Calibration Objekts. Hierbei werden die Konfigurationen geladen und einige Werte initialisiert.

## \_\_new\_\_

```python
def __new__(cls, *args, **kwargs):
```

Die Erstellung des Camera Calibration Objekts. Hierbei wird überprüft, ob das Objekt bereits existiert und falls ja, 
wird das bereits existierende Objekt zurückgegeben. Falls nicht, wird ein neues Objekt erstellt. Das sorgt dafür, 
dass das Objekt nur einmal pro Programm erstellt wird und somit die Kalibrierung nur einmal durchgeführt wird. (Singleton)

## \_calibrate

```python
def _calibrate(self) -> None:
```
Lädt die Kalibrierungs Einstellungen aus einem Pickle File und führt die Kalibrierung durch, falls diese noch nicht
durchgeführt wurde. Die Kalibrierung wird durchgeführt, indem die Bilder aus dem Ordner `calib_path` geladen werden und
daraus die Kameramatrix und die Verzerrungskoeffizienten berechnet werden. Diese werden dann in einem Pickle File gespeichert.

## \_get_obj_img_points

```python
def _get_obj_img_points(self) -> tuple[Sequence[Mat], Sequence[Mat]]:
```

Lädt die Bilder aus dem Ordner `calib_path` (standardmäßig Schachbrettmuster für Udacity) 
und berechnet daraus die Objekt- und Bildpunkte. Diese werden dann zurückgegeben.

## undistort

```python
def undistort(self, image: Mat) -> Mat:
```
Entzerrt das Bild, indem die Kameramatrix und die Verzerrungskoeffizienten verwendet werden.

## warp\_to\_birdseye

```python
def warp_to_birdseye(self, img: Mat) -> Mat:
```
Wendet die Transformationsmatrix an, um das Bild in Vogelperspektive zu bringen. Rechnet sich die Transformationsmatrix
aus.

## warp\_from\_birdseye

```python
def warp_from_birdseye(self, img: Mat) -> Mat:
```
Wendet die Inverse der Transformationsmatrix an, um das Bild aus der Vogelperspektive zurück zu transformieren.

## \_get\_trans\_mat\_pickle

```python
def _get_trans_mat_pickle(self, warp_mat_path: str) -> tuple[Mat, Mat]:
```
Lädt die Transformationsmatrizen aus einem Pickle File, falls diese bereits existieren.
