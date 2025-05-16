import pandas as pd
import numpy as np
import tensorflow as tf
import gc

# Load Train and Validation data. These can be found on: https://huggingface.co/datasets/TSaala/CollidingAdversaries/tree/main
# Assumes Train.feather and Val.feather are in same directory as this script
train_df = pd.read_feather('Train.feather')
val_df = pd.read_feather('Val.feather')

# Split Train and Validation DataFrames into x and y
x_train = train_df.loc[:, train_df.columns != 'Label'].to_numpy()
y_train = train_df["Label"].to_numpy()
x_val = val_df.loc[:, val_df.columns != 'Label'].to_numpy()
y_val = val_df["Label"].to_numpy()

# Optional: Delete Train and Validation DataFrames after splitting
del train_df
del val_df
gc.collect()


# Define your model architecture, as well as hyperparameters etc. if wanted
class TopoDNN:
    def __init__(self, input_dim, activation='relu', final_activation='sigmoid',
                 loss='binary_crossentropy', lr=0.00005):
        self.input_dim = input_dim
        self.activation = activation
        self.final_activation = final_activation
        self.loss = loss
        self.lr = lr
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=300, input_dim=self.input_dim, activation=self.activation,
                                        kernel_regularizer=tf.keras.regularizers.L1(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(units=102, activation=self.activation, kernel_regularizer=tf.keras.regularizers.L1(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(units=12, activation=self.activation, kernel_regularizer=tf.keras.regularizers.L1(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(units=6, activation=self.activation, kernel_regularizer=tf.keras.regularizers.L1(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(units=1, activation=self.final_activation))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss=self.loss,
                      metrics=['accuracy', tf.keras.metrics.AUC()])
        return model

    def train(self, x_train, y_train, x_val, y_val, batch_size=200, epochs=100):
        save_model = tf.keras.callbacks.ModelCheckpoint(f'model.hdf5',
                                                        save_best_only=True,
                                                        monitor='val_loss',
                                                        mode='min')
        history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', verbose=True), save_model]
        )
        return history

    def load_best_model(self):
        self.model = tf.keras.models.load_model(f'model.hdf5')

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)
    
    def predict(self, x_input):
        predictions = self.model.predict(x_input)
        return predictions

    def get_model(self):
        return self.model
    

# Optional: Define some form of robustness improvements on the model
def make_model_robust(model):
    # Do your magic here
    robust_model = model
    return robust_model


# Wrap model and train it
model_wrapper = TopoDNN(input_dim=x_train.shape[1])
model_wrapper.train(x_train, y_train, x_val, y_val, batch_size=256, epochs=100)
model_wrapper.load_best_model()

# Optional: make model robust with explicit method
model_wrapper = make_model_robust(model_wrapper)


# Load accompanying clean test data and split it into x and y
clean_test_df = pd.read_feather('Clean_Test_Public.feather')
x_test_clean = clean_test_df.loc[:, clean_test_df.columns != 'Label'].to_numpy()
y_test_clean = clean_test_df["Label"].to_numpy()

# Predict on the clean test samples using the model
clean_pred = (np.squeeze(model_wrapper.predict(x_test_clean)) >= 0.5).astype(int)


# Load accompanying adversarial test data and split it into x and y
adv_test_df = pd.read_feather('Adversaries_Test_Public.feather')
x_test_adv = adv_test_df.loc[:, adv_test_df.columns != 'Label'].to_numpy()
y_test_adv = adv_test_df["Label"].to_numpy()

# Predict on the adversarial test samples using the model
adv_pred = (np.squeeze(model_wrapper.predict(x_test_adv)) >= 0.5).astype(int)


# Get performance (Accuracy) on the clean and adversarial samples
from sklearn.metrics import accuracy_score
clean_acc = accuracy_score(y_test_clean, clean_pred)
print(f"Accuracy on clean samples: {clean_acc}")

adv_acc = accuracy_score(y_test_clean, adv_pred)
print(f"Accuracy on adversarial samples: {adv_acc}")

# Get combined score S = (Clean Accuracy + Adversarial Accuracy) / 2
score = (clean_acc + adv_acc) / 2
print(f"Final Score: {score}")


# For the final submission: Take the "model.hdf5" and zip it into a submission.zip
import zipfile
model_filename = 'model.hdf5'
submission_zip = 'submission.zip'

# Create a zip file containing only the model file
with zipfile.ZipFile(submission_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(model_filename, arcname='model.hdf5')
