import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# Directory containing the CSV files
directory_path = '/home/ihossain/ISMAIL/SSL-malware/pseudo_labels/csv/'

# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

# Dictionary to store FNR values for each k and similarity measure
fnr_dict = {}

# Iterate over each CSV file and calculate the FNR
for file_path in csv_files:
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Extract the Pseudo Label and Actual Label columns
    pseudo_labels = df['Pseudo Label']
    actual_labels = df['Actual Label']

    # Calculate the False Negatives and True Positives
    false_negatives = ((pseudo_labels == 0) & (actual_labels == 1)).sum()
    true_positives = ((pseudo_labels == 1) & (actual_labels == 1)).sum()

    # Calculate the False Negative Rate (FNR)
    fnr = false_negatives / (false_negatives + true_positives)

    # Extract k and similarity measure from the file name
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    k = int(parts[-1].split('.')[0])
    similarity_measure = parts[-2]

    # Store the FNR value in the dictionary
    if k not in fnr_dict:
        fnr_dict[k] = {}
    fnr_dict[k][similarity_measure] = fnr

    # Print the FNR for the current file
    print(f'File: {file_name} - k: {k} - Similarity: {similarity_measure} - False Negative Rate (FNR): {fnr:.4f}')

# Plot the FNR values using a bar graph
ks = sorted(fnr_dict.keys())
cosine_fnr = [fnr_dict[k].get('cosine', 0) for k in ks]
euclidean_fnr = [fnr_dict[k].get('euclidean', 0) for k in ks]

bar_width = 0.35
index = np.arange(len(ks))

plt.figure(figsize=(12, 6))
plt.bar(index, cosine_fnr, bar_width, label='Cosine', color='skyblue')
plt.bar(index + bar_width, euclidean_fnr, bar_width, label='Euclidean', color='salmon')

plt.xlabel('k')
plt.ylabel('False Negative Rate (FNR)')
plt.title('False Negative Rate (FNR) for Each k and Similarity Measure')
plt.xticks(index + bar_width / 2, ks)
plt.legend()

plt.tight_layout()

# Save the plot as an image file
output_image_path = '/home/ihossain/ISMAIL/SSL-malware/pseudo_labels/fnr_plot.png'
plt.savefig(output_image_path)

# Show the plot
plt.show()

print(f'Plot saved to {output_image_path}')