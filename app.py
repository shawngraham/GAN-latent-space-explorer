from flask import Flask, send_file, jsonify, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Global variables
generator = None
training_images = []
latent_vectors = []
NOISE_DIM = 100
corner_vectors = {'top': None, 'right': None, 'bottom': None, 'left': None}

def load_model_and_images(model_path, image_folder):
    """Load the trained generator and training images"""
    global generator, training_images, latent_vectors
    
    # Load the generator
    generator = tf.keras.models.load_model(model_path)
    
    # Load and store training images
    training_images = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert('RGB').resize((64, 64))
            training_images.append(img)
    
    # Generate and store random latent vectors for each training image
    latent_vectors = [
        tf.random.normal([1, NOISE_DIM]) for _ in range(len(training_images))
    ]

def generate_interpolated_image(weights):
    """Generate an image from interpolated latent vectors"""
    # Check if all corner vectors are selected
    if any(v is None for v in corner_vectors.values()):
        return None
    
    # Create weighted sum of latent vectors
    combined_vector = tf.zeros([1, NOISE_DIM])
    for corner, weight in zip(corner_vectors.values(), weights):
        combined_vector += weight * corner
    
    # Generate image
    generated = generator(combined_vector, training=False)
    
    # Convert to PIL Image
    generated = ((generated[0].numpy() + 1) * 127.5).astype(np.uint8)
    img = Image.fromarray(generated)
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str

@app.route('/')
def serve_interface():
    """Serve the HTML interface"""
    with open('templates/interface.html', 'r') as f:
        return f.read()

@app.route('/api/training-images')
def get_training_images():
    """Return list of training images as base64"""
    image_data = []
    for img in training_images:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_data.append(img_str)
    return jsonify(image_data)

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate image from interpolation weights"""
    weights = request.json['weights']
    img_str = generate_interpolated_image(weights)
    if img_str is None:
        return jsonify({'message': 'Please select images for all corners'})
    return jsonify({'image': img_str})

@app.route('/api/select-image/<int:index>')
def select_image(index):
    """Set the selected image for a corner position"""
    position = request.args.get('position')
    if position not in corner_vectors:
        return jsonify({'error': 'Invalid position'})
    
    if 0 <= index < len(training_images):
        corner_vectors[position] = latent_vectors[index]
        return jsonify({'message': 'Image selected', 'index': index})
    else:
        return jsonify({'error': 'Invalid image index'})

if __name__ == '__main__':
    # Load model and images before starting server
    load_model_and_images(
        model_path='generator_model_final.keras',
        image_folder='downloaded_images'
    )
    app.run(debug=True)