import os
import numpy as np
from PIL import Image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU for inference.")
    except RuntimeError as e:
        print(e)
else:
    print("GPU not available; using CPU.")

# Load the InceptionV3 model (pretrained on ImageNet) in inference mode.
model = InceptionV3(weights='imagenet')
model.trainable = False

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((299,299))
    img_array = np.array(img)
    # Expand dims to get shape (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image (scaling pixel values etc.)
    img_array = preprocess_input(img_array)
    return img_array

def get_predictions(image_paths, batch_size=32):
    predictions = []
    n = len(image_paths)
    for i in range(0, n, batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for path in batch_paths:
            img_array = load_and_preprocess_image(path)
            batch_images.append(img_array)
        # Stack the batches
        batch_images = np.vstack(batch_images)
        preds = model.predict(batch_images)
        predictions.append(preds)
    predictions = np.vstack(predictions)
    return predictions

def calculate_inception_score(predictions, eps=1e-16):
    py = np.mean(predictions, axis=0, keepdims=True) 
    KL = predictions * (np.log(predictions + eps) - np.log(py + eps))
    kL = np.sum(KL, axis=1)
    mean_KL = np.mean(KL)
    return np.exp(mean_KL)

def get_image_paths(directory, extensions={'.png', '.jpg', '.jpeg', '.bmp'}):
    image_paths = []
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return image_paths
    for fname in os.listdir(directory):
        if any(fname.lower().endswith(ext) for ext in extensions):
            image_paths.append(os.path.join(directory, fname))
    return image_paths

def evaluate_category(category, original_base_dir, generated_base_dir):
    # Evaluate real images
    real_dir = os.path.join(original_base_dir, category)
    real_image_paths = get_image_paths(real_dir)
    real_preds = get_predictions(real_image_paths)
    real_inception_score = calculate_inception_score(real_preds)
    
    # Evaluate the generated image
    gen_image_path = os.path.join(generated_base_dir, f'{category}_generated.png')
    gen_preds = get_predictions([gen_image_path])
    gen_inception_score = calculate_inception_score(gen_preds)
    
    return real_inception_score, gen_inception_score

def main():
    # Define categories
    categories = ["CC", "EC", "HGSC", "LGSC", "MC"]
    original_base_dir = 'patch_class'
    generated_base_dir = 'generated_images'
    
    results = {}
    for category in categories:
        print(f"\nEvaluating category: {category}")
        real_is, gen_is = evaluate_category(category, original_base_dir, generated_base_dir)
        results[category] = {
            'real_inception_score': real_is,
            'generated_inception_score': gen_is
        }
        print(f"Category: {category}")
        print(f"  Average Inception Score for real images: {real_is}")
        print(f"  Inception Score for generated image: {gen_is}")
    
    # compute overall averages
    real_scores = [r['real_inception_score'] for r in results.values() if r['real_inception_score'] is not None]
    gen_scores = [r['generated_inception_score'] for r in results.values() if r['generated_inception_score'] is not None]
    if real_scores:
        overall_real_avg = np.mean(real_scores)
        print(f"\nOverall average Inception Score for real images: {overall_real_avg}")
    if gen_scores:
        overall_gen_avg = np.mean(gen_scores)
        print(f"Overall average Inception Score for generated images: {overall_gen_avg}")

if __name__ == '__main__':
    main()