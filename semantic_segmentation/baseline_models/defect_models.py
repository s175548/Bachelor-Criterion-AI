import tensorflow as tf,os
from tensorflow import keras


def extract_model(path,model_name):
    dst = os.path.join(data_path, model_name)
    return tf.keras.models.load_model(dst, custom_objects={"tf": tf, **tf.keras.utils.get_custom_objects()}, compile=False)
#model = keras.models.load_model('path/to/location')


if __name__ == "__main__":
    data_path = r'C:\Users\Mads-\Desktop\leather_patches'
    model_name1 = r'defect_model_4.h5'
    model_name2 = r'defect_model_5.h5'

    defect_model_4 = extract_model(data_path,model_name1)
    defect_model_5 = extract_model(data_path,model_name2)

    print(defect_model_4.summary())
    print(defect_model_5.summary())

