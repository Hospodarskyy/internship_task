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



