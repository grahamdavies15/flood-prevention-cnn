import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
winter_df = pd.read_csv('winter_predictions.csv')
autumn_df = pd.read_csv('autumn_predictions.csv')
spring_df = pd.read_csv('spring_predictions.csv')

# Combine the data into a single DataFrame
df = pd.concat([winter_df, autumn_df, spring_df])

# Calculate the accuracy for each season
accuracy_df = df.groupby('season').apply(lambda x: (x['label'] == x['pred']).mean()).reset_index(name='accuracy')

# Plot the accuracy comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=accuracy_df, x='season', y='accuracy')
plt.title('Prediction Accuracy by Season')
plt.xlabel('Season')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.show()