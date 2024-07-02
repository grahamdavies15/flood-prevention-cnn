import os
import pandas as pd

# viewing images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from datetime import datetime


def label_images(base_dir):
    image_data = []

    # Traverse the base directory to find all site directories
    for site in os.listdir(base_dir):
        site_path = os.path.join(base_dir, site)
        if os.path.isdir(site_path):
            # Define the subdirectories within each site
            subdirs = ['blocked', 'clear', 'other']

            # Traverse each subdirectory
            for subdir in subdirs:
                subdir_path = os.path.join(site_path, subdir)

                if os.path.exists(subdir_path):
                    # Walk through the subdirectory
                    for root, _, files in os.walk(subdir_path):
                        for file in files:
                            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                                # Full file path
                                file_path = os.path.join(root, file)

                                # Append the data to the list
                                image_data.append({
                                    'file_path': file_path,
                                    'site': site,
                                    'label': subdir
                                })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(image_data)

    return df

# Example usage
base_dir = 'Data/blockagedetection_dataset/images'
df_images = label_images(base_dir)

# Filter out the 'other' label
df_filtered = df_images[df_images['label'].isin(['blocked', 'clear'])]

# Group by site and label, then count the images
summary = df_filtered.groupby(['site', 'label']).size().unstack(fill_value=0)

# Determine the minimum count between blocked and clear labels for each site
summary['balanced'] = summary[['blocked', 'clear']].min(axis=1)
