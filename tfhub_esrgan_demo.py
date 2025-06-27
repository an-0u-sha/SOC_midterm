import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_PATH = '/content/1b3b0a5e1ab348ccae48a148b7edb167.png'
MODEL_HANDLE = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def load_img(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((img.width // 4, img.height // 4), Image.BICUBIC)
    return np.array(img)

def preprocess(img):
    hr = img.astype(np.float32) / 255.0
    return tf.expand_dims(hr, 0)

def postprocess(sr):
    sr_img = tf.clip_by_value(sr[0], 0, 1)
    return (sr_img * 255).numpy().astype(np.uint8)

def visualize(lr, sr):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.title('Original LR'); plt.imshow(lr)
    plt.subplot(1,2,2); plt.title('SR via TF‑Hub ESRGAN'); plt.imshow(sr)
    plt.axis('off')
    plt.show()

def main():
    img = load_img(IMAGE_PATH)
    img_lr = img  # using the same image as LR for demo
    img_input = preprocess(img_lr)
    model = hub.load(MODEL_HANDLE)
    sr = model(img_input)
    sr_img = postprocess(sr)
    visualize(img_lr, sr_img)

if __name__ == '__main__':
    main()
