#Code to convert h5 to tflite
import tensorflow as tf

model =tf.keras.models.load_model("keypoint_classifier.h5")

#Convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
open("keypoint_classifier.tflite", "wb").write(tflite_model)

#Implement optimization strategy for smaller model sizes
converter.optimizations = [tf.lite.Optimize.DEFAULT] #Uses default optimization strategy to reduce the model size
tflite_model_optimized = converter.convert()
open("keypoint_classifier_optimized.tflite", "wb").write(tflite_model_optimized)