# Konventionelle Spurenerkennung

## Project requirements

- Minimum is to detect the lanes on the [Project Video](./Docs/project_video.mp4)
- Segmentation of the image
- Preprocessing (also Camera Calibration)
- Use of OpenCV (Color Spaces, Histograms, ...)
- In real time (at least 20 fps) without displaying the image
- Increase speed of lane detection by some way
- Show the curve radius

## Our results
- Lane Detection on not only the [Project Video](./images/Udacity/project_video.mp4) but also on the [Challenge Video](./images/Udacity/challenge_video.mp4)
  - [Project Video with Lane Detection](./Docs/Videos/detected_project_video.mp4)
  - [Challenge Video with Lane Detection](./Docs/Videos/detected_challenge_video.mp4)
- Over 70 FPS without visualization:   
Frames: 1260   
Time: 17.145667791366577   
FPS: 73.48795131995107
- Over 30 FPS with visualization
- We also implemented Lane Detection on the Harder Challenge Video with Machine Learning, aswell as License Plate Detection, in our other repository: [Additional Tasks](https://github.com/Ben-schlch/additions_to_conv_lane_detect)

## Installation

Please use Python [Venv](https://docs.python.org/3/library/venv.html) to install the dependencies.

Linux:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows: 
```bash
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Usage

To run the lane detection on the [Project Video](./images/Udacity/project_video.mp4) please stay in the root directory of the project and run the following command:

Without visualization:
```bash
python ./src/main.py --video ./Images/Udacity/challenge_video.mp4
```

With visualization:
```bash
python ./src/main.py --visualize --video ./Images/Udacity/challenge_video.mp4
```

## Step by Step Image Processing

To visually follow the steps of the image processing, check our [Step by Step visualization Jupyter Notebook](./Docs/step_by_step.ipynb).    
By doing so we also visually present most of the functions we calll and use.

## More Details

For more details on the implementation and the functions we used, check our:
- [Camera Calibration and Warp](./Docs/Calibration.md)
- [Preprocess](./Docs/Preprocessing.md)
- [Main Documentation](./Docs/Main.md)

## Discussion of the results

The results on the Project Video and the Challenge Video are quite good. Even curves, shadows, etc. are no problem for our algorithm.   
This was achieeved by using a very exact ROI and good preprocessing. The parameters were tuned in a long iterative process.

The algorithm is also quite fast. We can process the Project Video with over 70 FPS without visualization.    
This is achieved by leaving out the formerly included Hough Transform and using a very exact ROI.   
The call graph enabled us to find the biggest slow down which was reading the image ino memory. Therefore we parallelize the Input reading of the next image with the processing of the current image.

However the algorithm is not perfect. It is not able to detect the lanes on the [Harder Challenge Video](./images/Udacity/harder_challenge_video.mp4).
The reasons for this are the following:
- The algorithm is not able to detect lanes on the very different lighting conditions
- We split the image into two parts and detect the lanes on each part. This is not possible on the Harder Challenge Video because the curves are too sharp.    
  Therefore we would need to implement another way of grouping the lanes, e.g. by using the color information (--> Left Lane is always yellow ), or another more sophisticated way.

Some of our ideas did not find their way into the final algorithm:
- Hough Transform: We tried to use the Hough Transform to detect the lanes. However this ended us up with only lines and not with a curve.   
  Also we reduced calculation cost by removing the Hough Transform.
- We tried to not use Birds Eye View, which ended up working. But for the curve radius calculation we needed to use itanyway so we decided to use it for the whole algorithm.

## Lessons Learned

Do not underestimate the time needed for the project. Especially for finetuning parameters.   
This project is extremely time consuming, although it made fun. Sometimes it is exhausting when you spend hours on finetuning parameters and the result is not as expected.

Also we learned a lot about the OpenCV library and how to use it.   

We also learned how to use the call graph to find the slowest functions,....


## What we would do differently next time and what we want to try in the future

- Try the Sliding Window approach to detect the lanes. We did not use it because we thought it would be too boring. We got it recommended but thought it would be nice to try to use our knowledge from the course.
- Try to search for the lane only close to the lane from the previous frame. This would be a good idea to speed up the algorithm and improve results.
- Try to use the color information to group the lanes. This would be a good idea to improve the results on the Harder Challenge Video.

## References
https://medium.com/analytics-vidhya/building-a-lane-detection-system-f7a727c6694

https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132

https://automaticaddison.com/the-ultimate-guide-to-real-time-lane-detection-using-opencv/

https://www.analyticsvidhya.com/blog/2020/05/tutorial-real-time-lane-detection-opencv/

https://www.geeksforgeeks.org/opencv-real-time-road-lane-detection/

https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0