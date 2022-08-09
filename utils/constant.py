from keras.models import load_model
from keras.models import model_from_json


brain_tumor_stoke_model_json_file = open("./model/brain_tumor_model.json", "r")
brain_tumor_stoke_model = brain_tumor_stoke_model_json_file.read()
brain_tumor_stoke_model_json_file.close()

brain_tumor_stoke_model = model_from_json(brain_tumor_stoke_model)
brain_tumor_stoke_model.load_weights("./model/brain_tumor_model_weights.h5")




tumor_or_normal_json_file = open("./model/tumor_or_normal.json")
tumor_or_normal = tumor_or_normal_json_file.read()
tumor_or_normal_json_file.close()

tumor_or_normal = model_from_json(tumor_or_normal)
tumor_or_normal.load_weights("./model/tumor_or_normal_weights.h5")

