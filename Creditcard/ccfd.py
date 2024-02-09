import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load the training dataset
df_train = pd.read_csv('C:/Users/Tharaneetharan/PycharmProjects/pythonProject/creditcard/fraudTrain.csv')

# Load the testing dataset
df_test = pd.read_csv('C:/Users/Tharaneetharan/PycharmProjects/pythonProject/creditcard/fraudTest.csv')

# Identify non-numeric columns
non_numeric_columns = ['trans_date_trans_time', 'merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num']

# Explore and preprocess your training data
X_train = df_train.drop(['is_fraud'] + non_numeric_columns, axis=1)
y_train = df_train['is_fraud']

# Explore and preprocess your testing data
X_test = df_test.drop(['is_fraud'] + non_numeric_columns, axis=1)
y_test = df_test['is_fraud']

# Map class names for better clarity
class_names = ['Legitimate', 'Fraudulent']
y_train = y_train.map({0: class_names[0], 1: class_names[1]})
y_test = y_test.map({0: class_names[0], 1: class_names[1]})

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, predictions, labels=class_names)
classification_rep = classification_report(y_test, predictions)

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)

# Visualize the confusion matrix
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = range(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
