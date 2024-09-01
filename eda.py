import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def label_images(base_dir):
    image_data = []

    # Traverse the base directory to find all site directories
    for site in os.listdir(base_dir):
        site_path = os.path.join(base_dir, site)
        if os.path.isdir(site_path):
            subdirs = ['blocked', 'clear', 'other']

            # Traverse each subdirectory
            for subdir in subdirs:
                subdir_path = os.path.join(site_path, subdir)
                if os.path.exists(subdir_path):
                    for root, _, files in os.walk(subdir_path):
                        for file in files:
                            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                                # Full file path
                                file_path = os.path.join(root, file)
                                image_data.append({
                                    'file_path': file_path,
                                    'site': site,
                                    'label': subdir
                                })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(image_data)
    return df


def assign_season(date):
    year = date.year
    seasons = {
        'Spring': (datetime(year, 3, 20), datetime(year, 6, 21)),
        'Summer': (datetime(year, 6, 21), datetime(year, 9, 23)),
        'Autumn': (datetime(year, 9, 23), datetime(year, 12, 22)),
        'Winter_1': (datetime(year, 1, 1), datetime(year, 3, 20)),
        'Winter_2': (datetime(year, 12, 22), datetime(year, 12, 31))
    }
    for season, (start, end) in seasons.items():
        if start <= date <= end:
            return 'Winter' if season in ['Winter_1', 'Winter_2'] else season


# Determine the base directory dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, 'Data/blockagedetection_dataset/images')
df_images = label_images(base_dir)

# Extract the datetime string from the filenames by removing the extension
df_images['datetime_str'] = df_images['file_path'].apply(
    lambda x: '_'.join(os.path.basename(x).split('.')[0].split('_')[:5])
)

# Convert the datetime string to datetime objects
df_images['date'] = pd.to_datetime(df_images['datetime_str'], format='%Y_%m_%d_%H_%M')

# Drop the intermediate datetime string column
df_images.drop(columns=['datetime_str'], inplace=True)

# Apply the function to assign seasons
df_images['season'] = df_images['date'].apply(assign_season)


missing_dates = df_images['date'].isna().sum()
print(f"Number of images with missing or invalid dates: {missing_dates}")

missing_seasons = df_images[df_images['season'].isna()]
print(f"Number of images without an assigned season: {len(missing_seasons)}")
print(missing_seasons[['site', 'file_path']])

# Group by season and label, then count the images
season_summary = df_images.groupby(['season', 'label']).size().unstack(fill_value=0)

# Plotting the bar plot with stacked sections for each label
season_summary.plot(kind='bar', stacked=True, figsize=(10, 7), color=['blue', 'green', 'red'])
plt.title('Distribution of Blocked, Clear, and Other Images by Season')
plt.xlabel('Season')
plt.ylabel('Number of Images')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'plots/season_distribution.png'))
plt.show()

# Print the summary for reference
print("Image Distribution by Season and Label:")
print(season_summary)