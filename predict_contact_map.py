import os
import sys
import numpy as np
import pconsc4
import tensorflow as tf
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def predict_single_protein(input_file, output_file):
    # Load the pconsc4 model
    model = pconsc4.get_pconsc4()

    try:
        print(f'Processing {input_file}')
        pred = pconsc4.predict(model, input_file)  # Predict the contact map
        np.save(output_file, pred['cmap'])         # Save the contact map in .npy format
        print(f'{output_file} saved.')

    except Exception as e:
        print(f'Error processing {input_file}: {e}')
    finally:
        # Clear the Keras session to release memory
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

if __name__ == "__main__":
    # Check if input and output file paths are provided
    if len(sys.argv) != 3:
        print("Usage: python predict_contact_map.py input_file output_file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    predict_single_protein(input_file, output_file)
