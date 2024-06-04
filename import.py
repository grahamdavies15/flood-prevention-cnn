import os
import pandas as pd


def label_images(base_dir):
    # Initialize an empty list to store the image information
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
                            if file.lower().endswith(
                                    ('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                                # Full file path
                                file_path = os.path.join(root, file)

                                # Extract location from the file path
                                location = os.path.relpath(root, base_dir)

                                # Append the data to the list
                                image_data.append({
                                    'file_path': file_path,
                                    'label': subdir,
                                    'location': location
                                })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(image_data)

    return df


# Example usage
base_dir = 'Data/blockagedetection_dataset/images'  # Replace with your actual base directory path
df_images = label_images(base_dir)

# Get the count for each label
label_counts = df_images['label'].value_counts()

# Print the summary
print(label_counts)
