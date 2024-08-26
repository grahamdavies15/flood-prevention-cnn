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

# Set up the plot with a larger font size and colour-blind friendly palette
plt.figure(figsize=(14, 8))
sns.set_palette("colorblind")  # Ensure this line is in place
sns.barplot(data=accuracy_df, x='season', y='accuracy', hue='model', width=0.6)  # Adjusted width for spacing

# Add the title and labels with larger font sizes
plt.title('Prediction Accuracy by Season and Model', fontsize=20)  # Slightly increased title font size
plt.xlabel('Season', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1

# Add gridlines for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Improve legend placement (inside bottom-right)
plt.legend(title='Model', fontsize=12, title_fontsize=14, loc='lower right')

# Annotate the bars with accuracy values, ensuring consistent decimal places
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=12, color='black', xytext=(0, 8),  # Adjusted text offset
                       textcoords='offset points')

# Save the plot to a file
plt.savefig('plots/accuracy_comparison_all_models.png', bbox_inches='tight')

# Show the plot
plt.show()