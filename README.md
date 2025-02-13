Тиждень 1:

Для роботи був використаний датасет з відгуками українською з різних сайтів. Посилання на датасет: https://huggingface.co/datasets/vkovenko/cross_domain_uk_reviews.  Ми вибрали 600 відгуків, які користувачі залишили на сайті Rozetka.

Було проведено очищення текстів за допомогою функції, яка включала в себе видалення стоп-слів, лематизацію та нормалізацію.

Також були виділені ключові сутності (товар, ціна, скарги) та тональність. Для цієї частини завдання ми використовували Mistral API та числові оцінки, залишені авторами відгуків. Деякі сутності були виділені вручну.   


Тиждень 2:
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

```
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

![image](https://github.com/user-attachments/assets/2ef421e2-6423-467f-9c2d-18640cf64c1e)
![image](https://github.com/user-attachments/assets/eb06236d-1874-45eb-b8d2-eaa0a884fc49)



Також ми спробували використати попередньо навчені векторні представлення моделі FastText які підтримує українську мову. Ваги моделі завантажили у шар Embedding. 


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




