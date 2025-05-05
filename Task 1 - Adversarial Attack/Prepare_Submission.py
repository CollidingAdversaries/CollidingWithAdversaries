import pandas as pd
import numpy as np
import tensorflow as tf
import zipfile
import os

# Load given test samples, assumes "Clean_Test_Public.feather" is in same folder
clean_test_df = pd.read_feather('Clean_Test_Public.feather')

# Separate clean test dataframe into x and y
x_clean = clean_test_df.loc[:, clean_test_df.columns != 'Label'].to_numpy()
y_clean = clean_test_df['Label'].to_numpy()

# Insert here your logic to create adversaries, more parameters might be required
def gen_advs(samples, model_path):
    # Iff your adversaries rely on model information, load given pre-trained model, assumes "best_model.hdf5" is in same folder
    model = tf.keras.models.load_model(model_path)

    # Do the actual adversarial generation here...
    adv_samples = samples.copy()

    return adv_samples


x_adv = gen_advs(x_clean, model_path='best_model.hdf5')


# Create  list of triples (clean, adv, label)
triples = [(x_clean[i], x_adv[i], y_clean[i]) for i in range(len(y_clean))]

x_clean = np.array([t[0] for t in triples])
x_adv = np.array([t[1] for t in triples])
y = np.array([t[2] for t in triples])

# Save all three arrays together as a single submission.npz file
np.savez('submission.npz', clean=x_clean, adv=x_adv, labels=y)

# Create a zip file and add submission.npz into it
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('submission.npz', arcname='submission.npz')

# Delete un-zipped submission.npz file
os.remove('submission.npz')