import os
import pandas as pd
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

# Extract the datetime string from the filenames by removing the extension
df_images['datetime_str'] = df_images['file_path'].apply(lambda x: '_'.join(os.path.basename(x).split('.')[0].split('_')[:5]))

# Convert the datetime string to datetime objects
df_images['date'] = pd.to_datetime(df_images['datetime_str'], format='%Y_%m_%d_%H_%M')

# Print the results
print(df_images[['file_path', 'datetime_str', 'date']])

df_images.drop(columns=['datetime_str'], inplace=True)

def assign_season(date):
    year = date.year
    seasons = {
        'Spring': (datetime(year, 3, 20), datetime(year, 6, 20)),
        'Summer': (datetime(year, 6, 21), datetime(year, 9, 22)),
        'Autumn': (datetime(year, 9, 23), datetime(year, 12, 21)),
        'Winter': (datetime(year, 1, 1), datetime(year, 3, 19)),
        'Winter_2': (datetime(year, 12, 22), datetime(year, 12, 31))
    }
    for season, (start, end) in seasons.items():
        if start <= date <= end:
            return 'Winter' if season == 'Winter_2' else season
    # Handle edge case for December 31 to next year's March 19
    if date >= datetime(year, 12, 22) or date <= datetime(year, 3, 19):
        return 'Winter'

# Apply the function to assign seasons
df_images['season'] = df_images['date'].apply(assign_season)

# Separate the data into different seasons
winter_data = df_images[df_images['season'] == 'Winter']
spring_data = df_images[df_images['season'] == 'Spring']
summer_data = df_images[df_images['season'] == 'Summer']
autumn_data = df_images[df_images['season'] == 'Autumn']


print(summary)

###
import matplotlib.pyplot as plt

# Assuming df_images is already defined and has a 'date' column
# Create a histogram to show the count of images per date
plt.figure(figsize=(12, 6))
df_images['date'].hist(bins=50, edgecolor='black')
plt.xlabel('Date')
plt.ylabel('Number of Images')
plt.title('Histogram of Image Count by Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Define top sites
top_sites = ['sites_corshamaqueduct_cam1', 'Cornwall_BudeCedarGrove', 'Cornwall_Crinnis', 'Devon_BarnstapleConeyGut_Scree', 'Cornwall_Mevagissey_PreScree', 'sites_sheptonmallet_cam2', 'Cornwall_PenzanceCC']

# Filter data for each season and top sites
filtered_data = {
    'Winter': winter_data[winter_data['site'].isin(top_sites)],
    'Spring': spring_data[spring_data['site'].isin(top_sites)],
    'Summer': summer_data[summer_data['site'].isin(top_sites)],
    'Autumn': autumn_data[autumn_data['site'].isin(top_sites)]
}

balanced_data = []

# Function to balance data within each group
def balance_group(data):
    sampled_dfs = []
    for site, group in data.groupby('site'):
        blocked_df = group[group['label'] == 'blocked']
        clear_df = group[group['label'] == 'clear']
        if not blocked_df.empty and not clear_df.empty:
            sample_size = min(len(blocked_df), len(clear_df))
            blocked_sample = blocked_df.sample(n=sample_size, random_state=1)
            clear_sample = clear_df.sample(n=sample_size, random_state=1)
            sampled_df = pd.concat([blocked_sample, clear_sample])
            sampled_dfs.append(sampled_df)
    return pd.concat(sampled_dfs).sample(frac=1, random_state=1).reset_index(drop=True)

# Balance data for each season
for season, data in filtered_data.items():
    balanced_season = balance_group(data)
    balanced_data.append(balanced_season)

# Combine balanced data from all seasons
balanced_dataset = pd.concat(balanced_data).reset_index(drop=True)

print(balanced_dataset)
