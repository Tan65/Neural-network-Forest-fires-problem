# Neural-network-Forest-fires-problem
PREDICT THE BURNED AREA OF FOREST FIRES WITH NEURAL NETWORKS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data from the CSV file
data = pd.read_csv('forestfires.csv')

# Univariate Analysis
plt.figure(figsize=(15, 10))
for i, col in enumerate(['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']):
    plt.subplot(3, 3, i+1)
    sns.histplot(data[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Bivariate Analysis
plt.figure(figsize=(12, 8))
sns.pairplot(data, vars=['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area'], hue='size_category')
plt.show()

# Multivariate Analysis: Box plots
plt.figure(figsize=(15, 10))
for i, col in enumerate(['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='size_category', y=col, data=data)
    plt.title(col)
plt.tight_layout()
plt.show()

# Multivariate Analysis: Correlation Matrix
plt.figure(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Preprocess categorical variables
categorical_cols = ['month', 'day', 'size_category']
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Normalize numerical features
numerical_cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Split the data into features (X) and target variable (y)
X = data.drop(['area'], axis=1)
y = data['area']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model on the testing set
loss = model.evaluate(X_test, y_test)
print('Test loss:', loss)
