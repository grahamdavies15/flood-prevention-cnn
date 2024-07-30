import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# File paths
file_paths = glob.glob('csvs/*_pred_*_full.csv')

# Create an empty list to hold dataframes
dataframes = []

# Load all CSV files and add a column to identify the model
for file_path in file_paths:
    df = pd.read_csv(file_path)
    model = file_path.split('_')[0].split('/')[-1]
    season = file_path.split('_')[2]
    df['model'] = model
    df['season'] = season
    dataframes.append(df)

# Combine all data into a single DataFrame
df = pd.concat(dataframes)

# Calculate the accuracy for each model and season
accuracy_df = df.groupby(['model', 'season']).apply(lambda x: (x['label'] == x['pred']).mean()).reset_index(name='accuracy')

# Plot the accuracy comparison
plt.figure(figsize=(14, 8))
sns.barplot(data=accuracy_df, x='season', y='accuracy', hue='model')

# Add the title and labels
plt.title('Prediction Accuracy by Season and Model')
plt.xlabel('Season')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1

# Save the plot to a file
plt.savefig('plots/accuracy_comparison_all_models.png')

# Show the plot
plt.show()
