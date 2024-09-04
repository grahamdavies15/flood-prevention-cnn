import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split


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

# Determine the base directory dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, 'Data/blockagedetection_dataset/images')
df_images = label_images(base_dir)

print(len(df_images))

# Filter out the 'other' label
df_filtered = df_images[df_images['label'].isin(['blocked', 'clear'])]

# Group by site and label, then count the images
summary = df_filtered.groupby(['site', 'label']).size().unstack(fill_value=0)

# Determine the minimum count between blocked and clear labels for each site
summary['balanced'] = summary[['blocked', 'clear']].min(axis=1)

# Extract the datetime string from the filenames by removing the extension
df_images['datetime_str'] = df_images['file_path'].apply(
    lambda x: '_'.join(os.path.basename(x).split('.')[0].split('_')[:5])
)

# Convert the datetime string to datetime objects
df_images['date'] = pd.to_datetime(df_images['datetime_str'], format='%Y_%m_%d_%H_%M')

# Drop the intermediate datetime string column
df_images.drop(columns=['datetime_str'], inplace=True)


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


# Apply the function to assign seasons
df_images['season'] = df_images['date'].apply(assign_season)

# Separate the data into different seasons
winter_data = df_images[df_images['season'] == 'Winter']
spring_data = df_images[df_images['season'] == 'Spring']
summer_data = df_images[df_images['season'] == 'Summer']
autumn_data = df_images[df_images['season'] == 'Autumn']


# Define top sites
"""top_sites = [
    'sites_corshamaqueduct_cam1', 'Cornwall_BudeCedarGrove', 'Cornwall_Crinnis',
    'Devon_BarnstapleConeyGut_Scree', 'Cornwall_Mevagissey_PreScree',
    'sites_sheptonmallet_cam2', 'Cornwall_PenzanceCC'
]"""


# Function to filter and balance data within each group
def filter_and_balance(data):
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

    return pd.concat(sampled_dfs).sample(frac=1, random_state=1).reset_index(
        drop=True) if sampled_dfs else pd.DataFrame()


# Apply the function to each season's data
balanced_winter = filter_and_balance(winter_data)
balanced_spring = filter_and_balance(spring_data)
balanced_summer = filter_and_balance(summer_data)
balanced_autumn = filter_and_balance(autumn_data)

# Print results
print("Balanced Winter Data:", balanced_winter, sep="\n")
print("\nBalanced Spring Data:", balanced_spring, sep="\n")
print("\nBalanced Summer Data:", balanced_summer, sep="\n")
print("\nBalanced Autumn Data:", balanced_autumn, sep="\n")


# Function to split dataset into training/validation and test sets
def initial_split(season_data, test_size=0.2, random_state=42):
    image_filenames = season_data['file_path'].tolist()
    labels = season_data['label'].apply(lambda x: 1 if x == 'blocked' else 0).tolist()
    train_val_filenames, test_filenames, train_val_labels, test_labels = train_test_split(
        image_filenames, labels, test_size=test_size, random_state=random_state)
    return train_val_filenames, test_filenames, train_val_labels, test_labels


# Function to further split the training/validation set into training and validation sets
def train_val_split(train_val_filenames, train_val_labels, val_size=0.2, random_state=42):
    train_filenames, val_filenames, train_labels, val_labels = train_test_split(
        train_val_filenames, train_val_labels, test_size=val_size, random_state=random_state)
    return train_filenames, val_filenames, train_labels, val_labels


# Splitting data for each season
autumn_train_val, autumn_test, autumn_train_val_labels, autumn_test_labels = initial_split(balanced_autumn)
winter_train_val, winter_test, winter_train_val_labels, winter_test_labels = initial_split(balanced_winter)
spring_train_val, spring_test, spring_train_val_labels, spring_test_labels = initial_split(balanced_spring)

# Further split the training/validation set into training and validation sets
autumn_train, autumn_val, autumn_train_labels, autumn_val_labels = train_val_split(autumn_train_val,
                                                                                   autumn_train_val_labels)
winter_train, winter_val, winter_train_labels, winter_val_labels = train_val_split(winter_train_val,
                                                                                   winter_train_val_labels)
spring_train, spring_val, spring_train_labels, spring_val_labels = train_val_split(spring_train_val,
                                                                                   spring_train_val_labels)


# Combine the balanced datasets for all seasons after balancing
balanced_data = pd.concat([balanced_winter, balanced_spring, balanced_summer, balanced_autumn])
bins = 50

# Plot the original histogram (before balancing)
plt.figure(figsize=(12, 6))
df_images['date'].hist(bins=bins, edgecolor='black', color='skyblue')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.title('Histogram of Image Count by Date (Before Balancing)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'plots/data_dates_before_balancing.png'))
plt.show()

# Plot the second histogram (after balancing)
plt.figure(figsize=(12, 6))
balanced_data['date'].hist(bins=bins, edgecolor='black', color='lightcoral')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.title('Histogram of Image Count by Date (After Balancing)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'plots/data_dates_after_balancing.png'))
plt.show()