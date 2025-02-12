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
        print("Using GPU")
    except RuntimeError as e:
        print(e)
else:
    print("using CPU")

model = InceptionV3(weights='imagenet')
model.trainable = False

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((299, 299))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

def get_predictions(image_paths, batch_size=32):
    if not image_paths:
        return np.array([])
    preds_list = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [load_image(p) for p in batch_paths]
        batch_images = np.vstack(batch_images)
        preds = model.predict(batch_images)
        preds_list.append(preds)
    return np.vstack(preds_list) if preds_list else np.array([])

def calculate_inception_score(preds, eps=1e-16):
    if preds.size == 0:
        return None
    if preds.ndim == 1:
        preds = preds[np.newaxis, :]
    py = np.mean(preds, axis=0, keepdims=True)
    kl = preds * (np.log(preds + eps) - np.log(py + eps))
    kl_mean = np.mean(np.sum(kl, axis=1))
    return np.exp(kl_mean)

def get_image_paths(directory):
    paths = []
    if not os.path.isdir(directory):
        print(f"NOT FOUND: {directory}")
        return paths
    for fname in os.listdir(directory):
        if fname.lower().endswith('.png'):
            paths.append(os.path.join(directory, fname))
    return paths


def evaluate(cat, orig_dir, gen_dir):
    real_dir = os.path.join(orig_dir, cat)
    real_paths = get_image_paths(real_dir)
    real_preds = get_predictions(real_paths)
    real_is = calculate_inception_score(real_preds)

    gen_path = os.path.join(gen_dir, cat)
    gen_paths = get_image_paths(gen_path)
    gen_preds = get_predictions(gen_paths)
    gen_is = calculate_inception_score(gen_preds)

    return real_is, gen_is

def main():
    categories = ["CC","EC","HGSC","LGSC","MC"]
    orig_dir = 'patch_class'
    gen_dir = 'generated_images'
    results = {}

    for c in categories:
        print(f"\nEvaluating IS for: {c}")
        real_is, gen_is = evaluate(c, orig_dir, gen_dir)
        results[c] = {'real': real_is, 'generated': gen_is}
        print(f"Category: {c}")
        print(f"  Real images Inception Score: {real_is}")
        print(f"  Generated images Inception Score: {gen_is}")

    real_scores = [v['real'] for v in results.values() if v['real'] is not None]
    gen_scores = [v['generated'] for v in results.values() if v['generated'] is not None]
    
    if real_scores:
        print(f"\nOverall average real score: {np.mean(real_scores)}")
    if gen_scores:
        print(f"Overall average generated score: {np.mean(gen_scores)}")

if __name__ == '__main__':
    main()
