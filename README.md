# SHAP-SHAPley-Model-code
This is for educational purposes. This data is synthesized. Do not cite data for publications. Please cite author for code.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt


# 1. Generate some synthetic data for demonstration
np.random.seed(86)
X = pd.DataFrame({
   'Diabetes': np.random.rand(100) * 10,
   'Cancer': np.random.rand(100) * 5,
   'COPD': np.random.randint(0, 2, 100)
})
y = 2 * X['Diabetes'] + 3 * X['Cancer'] - 5 * X['COPD'] + np.random.randn(100) * 2


# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=86)


# 3. Train a machine learning model (e.g., RandomForestRegressor)
model = RandomForestRegressor(n_estimators=100, random_state=86)
model.fit(X_train, y_train)


print("Model training complete. R-squared on test set: {:.2f}".format(model.score(X_test, y_test)))


# 4. Initialize a SHAP explainer
# For tree-based models, shap.TreeExplainer is efficient
explainer = shap.TreeExplainer(model)


# 5. Calculate SHAP values for the test set
# It's often good practice to explain predictions on a subset of the data,
# especially for larger datasets, or specifically the test set.
shap_values = explainer.shap_values(X_test)


# 6. Visualize the SHAP scores
print("\nGenerating SHAP summary plot...")


# Summary plot: illustrates the impact of each feature on the model's output
# Red indicates higher feature value, blue indicates lower feature value
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Mean Absolute SHAP Value)")
plt.show()


shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot")
plt.show()


print("\nSHAP values calculated and visualized. The plots show feature importance and individual contributions to predictions.")
print("You can also inspect individual predictions, for example, the first test instance:")
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
