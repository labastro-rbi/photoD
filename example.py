import numpy as np
import photod
import tensorflow as tf

# print gpus used
print(tf.config.list_physical_devices('GPU'))

# Create a photometric distance model
photod_model = photod.PhotoD(batch_size=1024)

# Import a catalog with colors, their errors and the Bayesian estimates of Mr, Ar, FeH with uncertainties.
# 10000 stars are used for training, 20% of the catalog is kept for testing.
photod_model.import_csv("./data/BayesMethod3D_SDSSpatchRA340-350_KarloTest.txt", reduce=10000, test_split=0.2)

# Create the networks with the default architecture
photod_model.create_model()

# Train the network and the network for the uncertainties of the Bayesian estimates
training_time = photod_model.train_model(epochs=1024, iterations=2, decay_epochs=10, decay_rate=0.7)
training_time += photod_model.train_error_model(epochs=256, iterations=1, decay_epochs=10, decay_rate=0.7)
print("Training time: ", training_time)

# Test a model performance
x, y, y_error, p, sigma_p, bayes_sigma = photod_model.test_model()
# Print model metrics
print(photod_model.metrics)

# Plot model performance
x = tuple(np.array(v) for v in x)
photod.plot_tools.get_model_metrics(x, y, p, sigma_p)

# Predict for new stars: rmag, u-g, g-r, r-i, i-z and their errors
x_new = np.array([[20.43, 2.497, 1.435, 1.378, 0.724]])
x_new_error = np.array([[0.005, 0.060, 0.008, 0.007, 0.007]])
p_predicted, sigma_p_predicted, bayes_sigma_predicted = photod_model.predict((x_new, x_new_error))
print(p_predicted, sigma_p_predicted)

# Save the models
photod_model.save_model("./PhotoD.keras")
photod_model.save_error_model("./PhotoD_error.keras")
