# Master's Research Project
## MSc Data Science, University of Bath

# Project Overview

This directory contains various scripts, each of which can be executed independently, provided there is a `Data` folder in the same directory. The `Data` folder should contain the `blockagedetection_dataset` as sourced from [Van de Alie et al.]([https://example.com](https://researchdata.reading.ac.uk/498/)).

## Scripts

- **`eda.py`**  
  Performs exploratory data analysis, generating visualisations for the entire dataset. This script is customisable to include additional summary statistics or modifications as required.

- **`seasonal_data_split.py`**  
  Splits the dataset by season and balances it according to the site. This script is essential for all other scripts in the project and should be located in the same directory.

- **`train_seasonal.py`**  
  Trains models on a seasonal basis. The user needs to manually select the model type. The script also provides options to save model weights and track the best validation accuracy during training.

- **`classification_network.py`**  
  Adapted from the script provided by [Van de Alie et al.](https://example.com), this script loads a trained model and classifies images from a test set, outputting a CSV file containing the predictions. This script only needs to be run once to generate results for all models.

- **`seasonal_plot.py`**  
  Uses the CSV files generated by `classification_network.py` to create bar plots, which are featured in the project’s model comparison visualisations.

- **`saliency_mapping.py`**  
  Generates saliency maps to visualise model predictions. The user must manually select the classifier to be used.

- **`grad-cam.py`**  
  Produces smoothgrad-CAM visualisations. The user needs to manually select the classifier for generating these visualisations.

- **`occlusion.py`**  
  Generates occlusion sensitivity maps to highlight which parts of the input data contribute the most to the model's decisions. As with the other visualisation scripts, the classifier must be manually selected.

- **`integrated_gradients.py`**  
  Creates integrated gradient maps, showing which features are most influential in the model’s decision-making process. The user must manually select the classifier.

---

For detailed instructions on how to run each script or make modifications, please refer to the inline comments within the respective scripts.
