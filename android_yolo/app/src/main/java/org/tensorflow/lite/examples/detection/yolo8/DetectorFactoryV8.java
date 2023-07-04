package org.tensorflow.lite.examples.detection.yolo8;

import android.content.res.AssetManager;

import org.tensorflow.lite.examples.detection.yolo8.YoloV8Classifier;

import java.io.IOException;

public class DetectorFactoryV8 {
    public static YoloV8Classifier getDetector(
            final AssetManager assetManager,
            final String modelFilename)
            throws IOException {
        String labelFilename = null;
        boolean isQuantized = false;
        int inputSize = 0;
        int[] output_width = new int[]{0};
        int[][] masks = new int[][]{{0}};
        int[] anchors = new int[]{0};

        if (modelFilename.equals("yolov8n_float16.tflite")) {
            labelFilename = "file:///android_asset/coco.txt";
            isQuantized = false;
            inputSize = 640;
            output_width = new int[]{80, 40, 20};
            masks = new int[][]{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
            anchors = new int[]{
                    10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
            };
        }
        return YoloV8Classifier.create(assetManager, modelFilename, labelFilename, isQuantized,
                inputSize);
    }

}
