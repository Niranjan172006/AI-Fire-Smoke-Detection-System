# test_client.py
import requests, base64, json

# your test image
img_path = "test.jpg"

# read file
with open(img_path, "rb") as f:
    img_bytes = f.read()

# encode to base64
payload = {
    "image_base64": base64.b64encode(img_bytes).decode()
}

# send request
res = requests.post("http://127.0.0.1:5000/detect-image", json=payload)

print("Status code:", res.status_code)
print("Response:")
print(json.dumps(res.json(), indent=2))
