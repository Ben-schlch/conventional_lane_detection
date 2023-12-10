# Preprocess

In [preprocess.py](../src/preprocess.py) we implemented the functions to preprocess the images. The functions are called in [main.py](main.py) before the lane detection ( Splitting in lanes + Polynomfitting) is done.

## Explanation:

Following things happen in preprocess.py:
- Undistortion
- Warping to birds eye view
- Warping from birds eye view
- Color filtering
- Edge detection with Canny Edge FIlter  + Gaussian Blur
- Region of Interest / Masking

We do Gaussian Blur before Canny Edge Detection, because it showed us improved results and reduced noise.


## Alternatives to our proposed main.py

We could have changed many things:
- We ended up removing HoughTransform because we had struggles to fit curves and it also increased the calculation costs
- We could have used the yellow color to detect the left lane --> it is always yellow, and the right lane is always white
- We could have maybe also removed Canny Edge Detection

## Funktionen:

### preprocess
```python
def preprocess(img: Mat) -> Mat
```
Die zentrale Funktion: Bereitet das Bild für die Spurerkennung vor. Sie nimmt ein Bild als Eingabe und gibt ein vorverarbeitetes Bild zurück.
1. Filter nach Farben ( nur weiß und gelb bleiben übrig)  
__filter_colors
2. Graustufenbild   
cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
3. Gaußscher Weichzeichner   
cv2.GaussianBlur(img, (5, 5), 0)
4. Canny Edge Detection   
cv2.Canny(img, 50, 150)
5. Region of Interest   
__find_and_cut_region
6. Warp to Birdseye    
Calibration().warp_to_birdseye(img_roi) with [Calibration](Calibration.md)

### __filter_colors
```python
def __filter_colors(img: Mat) -> Mat
```
Filtert das Bild nach gelben und weißen Farben. Nur die weißen und gelben Farben bleiben übrig --> idealerweise Fahrspuren

### __find_and_cut_region
```python
def __find_and_cut_region(img: Mat) -> Mat
```
Findet den relevanten Bereich des Bildes und schneidet ihn aus. Hardgecodete Werte, die für die Udacity Videos/Bilder funktionieren.