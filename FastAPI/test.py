import requests

url = "http://127.0.0.1:8000/analyze"
data = {"texts": ["Чудовий продукт!", "Цей продукт має нормальні характеристики.", "Жах! Не купуйте в цього продавця!", ""]}

response = requests.post(url, json=data)
print(response.json())