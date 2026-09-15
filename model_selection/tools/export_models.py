"""Copy the four final models from models/trained_chiTest4 to USER/models in the .keras format."""
import tensorflow as tf

model_path = "../models/trained_chiTest4"
model_names = ["photozannv10p_trained", "photozannv14p_trained", "photozannv20p_trained", "photozannv24p_trained"]
model_output_path = "../USER/models/"
model_output_names = ["SimpleSingle", "SimpleMulti", "NaiveSingle", "NaiveMulti"]
for i in range(len(model_names)):
    model = tf.keras.models.load_model(model_path + "/" + model_names[i], compile=False)
    model.save(model_output_path + model_output_names[i]+".keras", include_optimizer=False)