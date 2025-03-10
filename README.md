Тиждень 1: Збір та попередня обробка даних

1. Збір відгуків (від 200 прикладів у TXT, CSV, JSON) 

Для роботи був використаний датасет з відгуками українською з різних сайтів. Посилання на датасет: https://huggingface.co/datasets/vkovenko/cross_domain_uk_reviews.  Ми вибрали 600 відгуків, які користувачі залишили на сайті Rozetka.

2. Очищення текстів: видалення стоп-слів, лематизація, нормалізація 

Було проведено очищення текстів за допомогою функції, яка включала в себе видалення стоп-слів, лематизацію та нормалізацію.

3. Виділення та розмітка ключових сутностей вручну (ціна/знижки, проблеми, назви товарів/послуг) 

Також були виділені ключові сутності (товар, ціна, скарги) та тональність. Для цієї частини завдання ми використовували Mistral API та числові оцінки, залишені авторами відгуків. Деякі сутності були виділені вручну.   


Тиждень 2: Визначення тональності тексту (Sentiment Analysis) 

1. Розробити модель для класифікації відгуків за тональністю (позитивний, нейтральний, негативний). Навчити прості ML-моделі (Naive Bayes, Logistic Regression) і порівняти їх з LSTM 

Спробували класифікувати за допомогою класичних алгоритмів (Logistic Regression, Naive Bayes, Random Forest, SVM, KNN):
![image](https://github.com/user-attachments/assets/ff3502a6-c278-485c-9484-b9d206b2cf33)

Проте для всіх моделей результати виглядали приблизно однаково:
<img width="518" alt="image" src="https://github.com/user-attachments/assets/f8efda82-6306-4d83-b6f8-3c8bf206ec2c" />

Бачимо що клас 'Neutral' модель класифікує зовсім некоректно. Та й загальна точність передбачень не сильно відрізняється від random.


Оскільки результат був незадовільний, ми вирішили збільшити в датасеті клас 'Positive' та 'Neutral':
<img width="1115" alt="image" src="https://github.com/user-attachments/assets/e38312e7-2203-45dc-970d-dc9aaaebc325" />

![image](https://github.com/user-attachments/assets/affee725-417a-43cf-8158-f2d35fc9d031)
<img width="530" alt="image" src="https://github.com/user-attachments/assets/9a90cae6-b129-4b0d-bec1-6862f5fbefe2" />

На жаль результат не сильно покращився, але, принаймні, модель навчилася не класифікувати все як 'Negative'

Далі ми спробували LSTM. 

Попробували додати attention механізм:

```python
class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="attn_weights", shape=(input_shape[-1], 1),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="attn_bias", shape=(1,), initializer="zeros", trainable=True)

    def call(self, inputs):
        scores = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)  # Compute attention scores
        scores = tf.nn.softmax(scores, axis=1)  # Normalize scores across time steps
        context_vector = scores * inputs  # Apply attention weights
        context_vector = tf.reduce_sum(context_vector, axis=1)  # Summarize over sequence
        return context_vector
```

```python
# Hyperparameters
max_sequence_length = 40  
vocab_size = 5_000  
embedding_dim = 32  

# Define model architecture
input_layer = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(input_layer)

lstm_output = LSTM(64, return_sequences=True)(embedding_layer)  
attention_output = AttentionLayer()(lstm_output)  # Apply attention

dense_output = Dense(16, activation="relu")(attention_output)
dropout_layer = Dropout(0.3)(dense_output)

dense_output = Dense(16, activation="relu")(dropout_layer)
dropout_layer = Dropout(0.3)(dense_output)

output_layer = Dense(3, activation="softmax")(dropout_layer)  # Multi-class classification

# Build model
att_model = Model(inputs=input_layer, outputs=output_layer)
att_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```

![image](https://github.com/user-attachments/assets/2ef421e2-6423-467f-9c2d-18640cf64c1e)
![image](https://github.com/user-attachments/assets/eb06236d-1874-45eb-b8d2-eaa0a884fc49)





Третім пунктом була розробка FastAPI де треба врахувати обробку пакетних даних та впровадити обробку помилок (наприклад, якщо вхідний текст порожній). 
Тут приклад роботи api де є один пустий текст і повертається повідомлення про пустий текст:
```json
{
  "results": [
    {
      "text": "Чудовий продукт!",
      "sentiment": "2"
    },
    {
      "text": "Цей продукт має нормальні характеристики.",
      "sentiment": "0"
    },
    {
      "text": "Жах! Не купуйте в цього продавця!",
      "sentiment": "0"
    }
  ],
  "notification": "1 empty text(s) were skipped."
}
```




Тиждень 3: Класифікація відгуків за темами, виділення ключових фраз і питань 

1. Визначити основні теми (наприклад, “Обслуговування”, “Якість”, “Доставка”) 

Для класифікації відгуків за темами ми використали BERTopic. Ця модель добре працює з неструктурованими текстовими даними, дозволяючи автоматично виявляти тематичні кластери у відгуках.

Ідея нашого рішення полягає в тому, що зі всіх знайдених моделлю тем ми будемо групувати їх у категорії, які дійсно корисні для продавця. Наприклад, якщо у відгуку міститься інформація про знижки або високу ціну, ми відносимо його до теми "Ціна", незалежно від того, чи йдеться про побутові товари, техніку чи інші категорії. Додаткова деталізація, така як тип товару, не є критичною, оскільки її можна отримати безпосередньо з каталогу вебсайту, а не з аналізу тексту.


На основі аналізу ми визначили наступні ключові теми, які будемо відслідковувати:

- Ціна – інформація про ціну, знижки, акції.

- Досвід використання – особисті враження користувача після використання товару.

- Рекомендації – чи рекомендує користувач товар, чи ні.

- Доставка/Обслуговування – оцінка процесу доставки та якості обслуговування.

- Якість – інформація про якість товару, дефекти, надійність.

Цей підхід дозволяє сфокусуватися на інформації, яка має найбільшу цінність для аналізу та прийняття рішень продавцем.


2. Виділити ключові скарги та позитивні моменти (наприклад, “Доставка затрималася на 3 дні”, “Чудовий сервіс!”)

Для вирішення цього завдання ми використали apsect-based sentiment analisys model (ABSA). Конкретніше pyABSA (https://github.com/yangheng95/PyABSA) адже вона була навчена на багатьох мовах, включно з українською. 
Ця модель здатна виділяти основні сутності в тексті і щодо них робити семантичний аналіз:

```json
Review: дуже задоволена покупкою. тканина дуже плотна. прошите гарно. наповнювач також всередині плотний. тепле однозначно. і водночас легке. готуємось до зими в умовах війни(((

aspects: ['тканина', 'наповнювач']
sentiments: ['Positive', 'Positive']
```

3. Автоматично знаходити запитання в текстах (наприклад, “Як скасувати замовлення?”)

Для цього ми використали функції бібліотеки spacy.load("uk_core_news_md") і здійснили додатково пошук по ключових словах, таких як: "що", "чому", "де", "коли", "скільки", "звідки" тощо:

```json
Review:
Чому розетка не реагує на негативні відгуки та запитання? Дуже швидко перестав працювати основний брелок. Сенс двосторонньої сигналізації зник 

questions: [Чому розетка не реагує на негативні відгуки та запитання?]
```


Тиждень 4: Розгортання у Docker, тестування та документація

1. Розгортання у Docker
1.1. Dockerfile

- Розгортання проєкту виконується через Dockerfile, який:

- Використовує базовий образ python:3.9-slim.

- Встановлює необхідні системні залежності (gcc, g++, git, libomp5).

- Оптимізує встановлення Python-пакетів (розбивка на кроки для покращення кешування).

- Завантажує мовну модель spaCy для української мови (uk_core_news_md).

- Визначає відкритий порт 8000 та команду запуску FastAPI.

1.2. Docker Compose

Файл docker-compose.yaml описує сервіс api, який:

- Використовує поточний каталог для побудови образу.

- Проброшує порт 8000.

- Змонтовує каталоги ./ та ./models у контейнері.

- Встановлює змінну середовища TZ=UTC.

