package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetManager;

import java.io.IOException;

public class DetectorFactory {
    public static YoloV5Classifier getDetector(
            final AssetManager assetManager,
            final String modelFilename)
            throws IOException {
        String labelFilename = null;
        boolean isQuantized = false;
        int inputSize = 0;
        int[] output_width = new int[]{0};
        int[][] masks = new int[][]{{0}};
        int[] anchors = new int[]{0};

        if (modelFilename.equals("a_yolov8n_0.5_face_float16.tflite")) {
            labelFilename = "file:///android_asset/face_classes.txt";
            isQuantized = false;
            inputSize = 128;
            output_width = new int[]{80, 40, 20};
            masks = new int[][]{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
            anchors = new int[]{
                    10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
            };
        }
        else if (modelFilename.equals("yolov8n_1.0_face_float16.tflite")) {
            labelFilename = "file:///android_asset/face_classes.txt";
            isQuantized = false;
            inputSize = 128;
            output_width = new int[]{40, 20, 10};
            masks = new int[][]{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
            anchors = new int[]{
                    10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
            };
        }
        else if (modelFilename.equals("yolov8n_0.125_face_float16.tflite")) {
            labelFilename = "file:///android_asset/face_classes.txt";
            isQuantized = false;
            inputSize = 128;
            output_width = new int[]{40, 20, 10};
            masks = new int[][]{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
            anchors = new int[]{
                    10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
            };
        }
        return YoloV5Classifier.create(assetManager, modelFilename, labelFilename, isQuantized,
                inputSize);
    }

    public static YoloV5Combine getDetCls(
            final AssetManager assetManager,
            final String modelFilename)
            throws IOException {
        String labelFilename = null;
        boolean isQuantized = false;
        int inputSize = 0;
        int[] output_width = new int[]{0};
        int[][] masks = new int[][]{{0}};
        int[] anchors = new int[]{0};
        final String TF_CLS_API_MODEL_FILE = "mask_float16.tflite";
        final String TF_CLS_API_LABELS_FILE = "file:///android_asset/facemask.txt";

        if (modelFilename.equals("a_yolov8n_0.5_face_float16.tflite")) {
            labelFilename = "file:///android_asset/face_classes.txt";
            isQuantized = false;
            inputSize = 128;
            output_width = new int[]{80, 40, 20};
            masks = new int[][]{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
            anchors = new int[]{
                    10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
            };
        }
        else if (modelFilename.equals("yolov8n_1.0_face_float16.tflite")) {
            labelFilename = "file:///android_asset/face_classes.txt";
            isQuantized = false;
            inputSize = 128;
            output_width = new int[]{40, 20, 10};
            masks = new int[][]{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
            anchors = new int[]{
                    10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
            };
        }
        else if (modelFilename.equals("yolov8n_0.125_face_float16.tflite")) {
            labelFilename = "file:///android_asset/face_classes.txt";
            isQuantized = false;
            inputSize = 128;
            output_width = new int[]{40, 20, 10};
            masks = new int[][]{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
            anchors = new int[]{
                    10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
            };
        }
        else if (modelFilename.equals("mask_float16.tflite")) {
            labelFilename = "file:///android_asset/face_classes.txt";
            isQuantized = false;
            inputSize = 96;
            output_width = new int[]{40, 20, 10};
            masks = new int[][]{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
            anchors = new int[]{
                    10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
            };
        }
        return YoloV5Combine.create(assetManager, modelFilename, labelFilename, isQuantized,
                inputSize, TF_CLS_API_MODEL_FILE, TF_CLS_API_LABELS_FILE );
    }

}
