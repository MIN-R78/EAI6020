#Title: EAI6020 MIN LI MAR 16

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH =
COL_NAMES = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv(DATA_PATH, sep='\t', names=COL_NAMES, engine='python')


# EDA
print(data.info())
print(data.describe())
print("Missing values:", data.isnull().sum())
data.dropna(inplace=True)

#vis
plt.figure(figsize=(8, 5))
sns.countplot(x='rating', data=data, palette='viridis')
plt.title('Rating Distribution')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='rating', data=data)
plt.title('Boxplot of Ratings')
plt.show()
data = data[(data['rating'] >= 1) & (data['rating'] <= 5)]

# label
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
data['user_id'] = user_encoder.fit_transform(data['user_id'])
data['item_id'] = item_encoder.fit_transform(data['item_id'])

# 70.15.15
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print("Training set size:", len(train_data))
print("Validation set size:", len(val_data))
print("Test set size:", len(test_data))


import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Dot, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

num_users = data['user_id'].nunique()
num_items = data['item_id'].nunique()
embedding_dim = 50
user_input = Input(shape=(1,))
item_input = Input(shape=(1,))
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim)(user_input)
item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim)(item_input)
user_flatten = Flatten()(user_embedding)
item_flatten = Flatten()(item_embedding)
predicted_rating = Dot(axes=1)([user_flatten, item_flatten])

# model
model = Model(inputs=[user_input, item_input], outputs=predicted_rating)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.summary()

train_users = train_data['user_id'].values
train_items = train_data['item_id'].values
train_ratings = train_data['rating'].values

val_users = val_data['user_id'].values
val_items = val_data['item_id'].values
val_ratings = val_data['rating'].values

# train model
history = model.fit(
    [train_users, train_items], train_ratings,
    validation_data=([val_users, val_items], val_ratings),
    epochs=10,
    batch_size=64,
    verbose=1
)
# test model
test_users = test_data['user_id'].values
test_items = test_data['item_id'].values
test_ratings = test_data['rating'].values

# pre
predicted_ratings = model.predict([test_users, test_items])

# rmse
from sklearn.metrics import mean_squared_error
import numpy as np
rmse = np.sqrt(mean_squared_error(test_ratings, predicted_ratings))
print("Test Set RMSE:", rmse)

model.save('movie_recommendation_model.keras')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# pre
test_users = test_data['user_id'].values
test_items = test_data['item_id'].values
test_ratings = test_data['rating'].values
predicted_ratings = model.predict([test_users, test_items])

# mse mae
mse = mean_squared_error(test_ratings, predicted_ratings)
mae = mean_absolute_error(test_ratings, predicted_ratings)
r2 = r2_score(test_ratings, predicted_ratings)

print("Test Set MSE:", mse)
print("Test Set MAE:", mae)
print("Test Set R² Score:", r2)

residuals = test_ratings - predicted_ratings.flatten()
plt.figure(figsize=(8, 6))
plt.scatter(predicted_ratings, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Ratings')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(test_ratings, predicted_ratings, alpha=0.5)
plt.plot([min(test_ratings), max(test_ratings)], [min(test_ratings), max(test_ratings)], color='r', linestyle='--')
plt.xlabel('True Ratings')
plt.ylabel('Predicted Ratings')
plt.title('True vs Predicted Ratings')
plt.show()

#save model
import joblib
import os

save_dir = r''
os.makedirs(save_dir, exist_ok=True)

model.save(os.path.join(save_dir, 'movie_recommendation_model.h5'))
joblib.dump(user_encoder, os.path.join(save_dir, 'user_encoder.joblib'))
joblib.dump(item_encoder, os.path.join(save_dir, 'item_encoder.joblib'))