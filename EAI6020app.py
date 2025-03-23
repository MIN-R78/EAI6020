#Title: EAI6020 MIN LI MAR 16
from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)

# model
model = tf.keras.models.load_model('movie_recommendation_model.h5')
user_encoder = joblib.load('user_encoder.joblib')
item_encoder = joblib.load('item_encoder.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        user_id = user_encoder.transform([data['user_id']])[0]
        item_id = item_encoder.transform([data['item_id']])[0]

        prediction = model.predict([np.array([user_id]), np.array([item_id])])

        return jsonify({
            'user_id': data['user_id'],
            'item_id': data['item_id'],
            'predicted_rating': float(prediction[0][0])
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)