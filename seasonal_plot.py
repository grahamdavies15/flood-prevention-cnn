import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
spring_df = pd.read_csv('autumn_pred_spring_model.csv')
autumn_df = pd.read_csv('autumn_pred_autumn_model.csv')

# Combine the data into a single DataFrame

spring_df['season'] = 'autumn_s'
df = pd.concat([autumn_df, spring_df])

# Calculate the accuracy for each season
accuracy_df = df.groupby('season').apply(lambda x: (x['label'] == x['pred']).mean()).reset_index(name='accuracy')

# Plot the accuracy comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=accuracy_df, x='season', y='accuracy')

# Add a red dashed line at 0.86 titled "Winter Validation"
plt.axhline(0.86, color='red', linestyle='--', linewidth=2)
plt.text(2.5, 0.87, 'Winter Validation', color='red', ha='center', va='bottom')

plt.title('Prediction Accuracy by Season')
plt.xlabel('Season')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.show()