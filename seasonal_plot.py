import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.metrics import f1_score

# File paths
file_paths = glob.glob('csvs/*_pred_*_model.csv')

# Create an empty list to hold dataframes
dataframes = []

# Load all CSV files and add a column to identify the model and season
for file_path in file_paths:
    df = pd.read_csv(file_path)
    season = file_path.split('_pred_')[0].split('/')[-1]
    model = file_path.split('_pred_')[1].split('_model')[0]
    df['model'] = model
    df['season'] = season

    # Convert predictions to numeric form to match labels
    df['pred'] = df['pred'].map({'clear': 0, 'blocked': 1})

    dataframes.append(df)

    print(f'Model: {model}, Season: {season}')
    print(df.head())  # Inspect the first few rows of each DataFrame

# Combine all data into a single DataFrame
df = pd.concat(dataframes, ignore_index=True)
print(df.head())  # Inspect the combined DataFrame

# Calculate the accuracy for each model and season
accuracy_df = df.groupby(['model', 'season']).apply(lambda x: pd.Series({
        'accuracy': (x['label'] == x['pred']).mean(),
        'f1_score': f1_score(x['label'], x['pred']) }))

print(accuracy_df)  # Inspect the accuracy DataFrame

# Set up a custom color palette using 'skyblue' and 'lightcoral'
custom_palette = ['skyblue', 'lightcoral']

# Set up the plot with a larger font size and the custom color palette
plt.figure(figsize=(14, 8))
sns.set_palette(custom_palette)  # Use the custom palette

sns.barplot(data=accuracy_df.reset_index(), x='season', y='accuracy', hue='model', width=0.6)

# Add the title and labels with larger font sizes
plt.title('Prediction Accuracy by Season and Model', fontsize=20, y=1.05)  # Moved the title further up
plt.xlabel('Season', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1

# Add gridlines for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Improve legend placement (inside bottom-right)
plt.legend(title='Model', fontsize=12, title_fontsize=14, loc='lower right')

# Annotate the bars with accuracy values, ensuring consistent decimal places
for p in plt.gca().patches:
    height = p.get_height()
    if height > 0.001:  # Only annotate if height is greater than a very small threshold
        plt.gca().annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='center', fontsize=12, color='black', xytext=(0, 8),
                           textcoords='offset points')

# Save the plot to a file
plt.savefig('plots/accuracy_comparison_all_models.png', bbox_inches='tight')

# Show the plot
plt.show()