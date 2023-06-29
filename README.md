# TFLite-Object-Detection-Android-App-YOLOv5

### Pineline

- Train yolov5 model

- Convert yolov5 (.pt model) into a tensorflow model(.pb file)

- Convert tensorflow model (.pb model) to tflite model.

- Download and install Android Studio

- Build and run your Object detection App.

### Detail Pineline

- when building app, first jump in `DetectorActivity`:
    - Go to `manifests` to see in `activity` blocks, the first layout is `DetectorActivity`.
    - The class `DetectorActivity` extends from `CameraActivity` and implements `OnImageAvailableListener`
    - Define some parameters and define `detector` object from `DetectorFactory.getDetector` (in `tflite/DetectorFactory`). In `DetectorFactory`, we implement `getDectector` function by calling some code from `YoloV5Classifier`
    - In `processImage`, process image input and get results of detection by `detector.recognizeImage`
        


