import tensorflow as tf
import os

# Chemins
MODEL_DIR = "model"
H5_MODEL_PATH = os.path.join(MODEL_DIR, "production_model.h5")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "production_model.tflite")

if __name__ == "__main__":
    # Charge le modèle sans recompilation
    model = tf.keras.models.load_model(H5_MODEL_PATH, compile=False)

    # Initialisation du convertisseur TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimisation par défaut 
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Autoriser des ops TensorFlow pour les TensorList
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    # Désactiver l'abaissement expérimental de TensorList
    converter._experimental_lower_tensor_list_ops = False

    # Conversion
    tflite_data = converter.convert()

    # Sauvegarde du modèle TFLite
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_data)

    print(f"[+] Export TFLite terminé : {TFLITE_MODEL_PATH}")