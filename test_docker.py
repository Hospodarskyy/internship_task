import requests
import json

# Use localhost instead of 0.0.0.0 to connect from outside the container
url = "http://127.0.0.1:8000/analyze"

data = {
    "texts": [
        "Приємна продавець. Жалюзі ідеально підійшли. Згодна що трохи кривий крипіж. Ковпачки з боків так і не натягнула. Там все складно. Але це не заважає функціонуванню. І все інше супер",
        "Не можу оцінити відповідно товар, шнур не підходить. Шкода що не читала відгуки про товар заздалегідь .",
        "Жах! Не купуйте в цього продавця!",
        "Чому розетка не реагує на негативні відгуки та запитання? Чому не відповідає? Дуже швидко перестав працювати основний брелок. Сенс двосторонньої сигналізації зник",
        "",
        ""
    ]
}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raises an error for bad responses (4xx and 5xx)

    # Save the response to a JSON file
    with open("api_response.json", "w", encoding="utf-8") as f:
        json.dump(response.json(), f, ensure_ascii=False, indent=4)

    print(f"Response saved to api_response.json")

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
