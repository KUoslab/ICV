#flask_test.py
import json
import requests
import numpy as np
from PIL import Image

image = Image.open('test_image.jpg')
pixels = np.array(image)

headers = {'Content-Type':'application/json'}
address = "http://127.0.0.1:2431/inference"
data = {'images':pixels.tolist()}

result = requests.post(address, data=json.dumps(data), headers=headers)

print(str(result.content, encoding='utf-8'))