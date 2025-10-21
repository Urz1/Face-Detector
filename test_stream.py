import requests

stream_url = "http://192.168.137.164/mjpeg/1"
try:
    response = requests.get(stream_url, stream=True, timeout=5)
    print(f"Status Code: {response.status_code}")
    print("Headers:")
    for key, value in response.headers.items():
        print(f"  {key}: {value}")
    print("First 100 bytes:", next(response.iter_content(100)))
except Exception as e:
    print(f"Error: {str(e)}")