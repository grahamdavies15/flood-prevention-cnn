import os
import pandas as pd

# viewing images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from datetime import datetime
import import_data as id


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

# Collect balanced data for each site
balanced_data = []
for site in id.summary.index:
    min_count = summary.loc[site, 'balanced']

    blocked_images = df_filtered[(df_filtered['site'] == site) & (df_filtered['label'] == 'blocked')].sample(min_count)
    clear_images = df_filtered[(df_filtered['site'] == site) & (df_filtered['label'] == 'clear')].sample(min_count)

    balanced_data.append(blocked_images)
    balanced_data.append(clear_images)

# Concatenate the balanced data
balanced_df = pd.concat(balanced_data)

# Aggregate the counts for balanced images
balanced_summary = balanced_df.groupby(['site', 'label']).size().unstack(fill_value=0)
balanced_summary['total'] = balanced_summary['blocked'] + balanced_summary['clear']

# Select the top three sites with the highest number of balanced images
top_balanced_sites = balanced_summary.nlargest(7, 'total')  # chosen 7 as these each have 1000 images
print(top_balanced_sites)


###############


# Select one image per site
example_images = balanced_df[balanced_df['site'].isin(top_balanced_sites.index)].groupby('site').first().reset_index()

# Get the total count
example_images = example_images.merge(top_balanced_sites['total'], on='site')

# Plot the images in a 4x2 grid
fig, axes = plt.subplots(4, 2, figsize=(15, 15))

for i, ax in enumerate(axes.flat):
    if i < len(example_images) and i < 7:  # Ensure we only plot 7 images
        img_path = example_images.iloc[i]['file_path']
        img = mpimg.imread(img_path)
        site = example_images.iloc[i]['site']
        total = example_images.iloc[i]['total']
        ax.imshow(img)
        ax.set_title(f"{site} (Total: {total})")
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()




##############


# Extract the datetime string from the filenames by removing the extension
df_images['datetime_str'] = df_images['file'].apply(lambda x: '_'.join(x.split('.')[0].split('_')[:5]))

# Convert the datetime string to datetime objects
df_images['date'] = pd.to_datetime(df_images['datetime_str'], format='%Y_%m_%d_%H_%M')

# Print the results
print(df_images[['file', 'datetime_str', 'date']])

df_images.drop(columns=['datetime_str'], inplace=True)



############

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



#################

top_sites = ['sites_corshamaqueduct_cam1', 'Cornwall_BudeCedarGrove', 'Cornwall_Crinnis', 'Devon_BarnstapleConeyGut_Scree', 'Cornwall_Mevagissey_PreScree', 'sites_sheptonmallet_cam2', 'Cornwall_PenzanceCC']

filtered_winter = winter_data[winter_data['site'].isin(top_sites)]
filtered_spring = spring_data[spring_data['site'].isin(top_sites)]
filtered_summer = summer_data[summer_data['site'].isin(top_sites)]
filtered_autumn = autumn_data[autumn_data['site'].isin(top_sites)]

print(filtered_winter.count())
print(filtered_summer.count())

##########

sampled_dfs = []
# Only winter done
# Group by 'site' and sample within each group
for site, group in filtered_winter.groupby('site'):
    # Separate 'blocked' and 'clear' samples within each group
    blocked_df = group[group['label'] == 'blocked']
    clear_df = group[group['label'] == 'clear']

    # Ensure there are both 'blocked' and 'clear' samples in the group
    if not blocked_df.empty and not clear_df.empty:
        # Determine the sample size
        sample_size = min(len(blocked_df), len(clear_df))

        # Randomly sample without replacement
        blocked_sample = blocked_df.sample(n=sample_size, random_state=1)
        clear_sample = clear_df.sample(n=sample_size, random_state=1)

        # Concatenate and append
        sampled_df = pd.concat([blocked_sample, clear_sample])
        sampled_dfs.append(sampled_df)

# Concatenate and shuffle
balanced_winter = pd.concat(sampled_dfs)
balanced_winter = balanced_winter.sample(frac=1, random_state=1).reset_index(drop=True)

print(balanced_winter)