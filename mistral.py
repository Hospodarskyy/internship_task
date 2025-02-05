from pydantic import BaseModel
from mistralai import Mistral
import json

# content = 'Недостатньо інформації про контрольний розчин. Якщо його перевірити на апараті і він показує якусь цифру, то з якою цифрою цей показник порівнювати? Хочеться побачити відповідь'
# content = 'Отримала сьогодні тискомір не той що замовила?!Коли передзвонила ,то мене ще звинуватили ,що не вмію з ним працювати.Люди не купляйтесь на таких продавців ,що обманюють людей'
# content = 'Почав встановлювати, а мийка вигнута дугою. По краях лежить ідеально на столешні, а посередині щілина майже в півсантиметри. Як так???'
content = 'Купили по акції і купились на crema , висновки:пінка не дуже зразу осідає,і смак в порівнянні з іншими ambassador для мене ця найгірша'


class Book(BaseModel):
    item: str
    sale_info: str
    complaints: str


import os
from mistralai import Mistral

api_key = "fb3VxADhedRewoBn5wblVrzxWAg9l3bo"
model = "mistral-small"

client = Mistral(api_key=api_key)

chat_response = client.chat.parse(
    model=model,
    messages=[
        {
            "role": "system", 
            "content": "Витягни назву товару, інформацію про наявність акції чи знижки (якщо інформації немає, виведи None) та в чому полягає скарга з відгуку."
        },
        {
            "role": "user", 
            "content": content
        },
    ],
    response_format=Book,
    max_tokens=256,
    temperature=0
)

print(type(chat_response))
print(chat_response.choices[0].message.content)