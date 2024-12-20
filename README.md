# MACHINE LEARNING PROJECT - SENTIMENT ANALYSIS

## TURKISH:

Bu proje, metinleri **Pozitif**, **Negatif** ve **Nötr** olmak üzere üç farklı duygu kategorisine sınıflandıran bir **makine öğrenmesi** ve **yapay zeka** tabanlı **duygu analizi** sistemidir. Kullanıcıların girdiği metinler, doğal dil işleme (NLP) teknikleri ile işlenir ve ardından bir makine öğrenmesi modeli aracılığıyla duygu etiketleri ile sınıflandırılır.

Proje, **veritabanı** kullanarak verilerin depolanmasını sağlar. Bu veritabanı, metinlerin ve bunlara ait etiketlerin saklandığı bir **SQLite** veritabanıdır. Kullanıcılar, sisteme metinler ekleyebilir, bu metinleri eğitim verisi olarak kullanabilir ve duygu analizi yapılacak metinleri sisteme yükleyebilirler. Veritabanı, modelin eğitim sürecinde kullanılan verileri depolar ve aynı zamanda sınıflandırma işlemleri için kullanılan metinleri yönetir.

Projenin aşamaları şu şekilde işler:
- **Veritabanı Entegrasyonu:** Veriler, kullanıcıdan alınarak SQLite veritabanına kaydedilir. Veritabanı, metinlerin ve etiketlerin düzenli bir şekilde saklanmasını sağlar.
- **Metin Temizleme:** Veritabanından alınan metinler, gereksiz semboller, sayılar ve özel karakterlerden arındırılır. Ayrıca, tüm metin küçük harfe dönüştürülür.
- **Kökleme (Stemming):** Türkçe dilinde kullanılan kelimeler, köklerine indirgenerek anlamlı bir analiz yapılır. Bu aşama, farklı çekimlerdeki kelimelerin aynı kök üzerinden değerlendirilmesine olanak tanır.
- **Stopwords Kaldırma:** Türkçe stopword'ler (anlamsız, önemsiz kelimeler) metinlerden çıkarılır. Bu sayede model, duygu analizi için daha anlamlı kelimelere odaklanabilir.
- **Vektörleştirme:** Temizlenmiş metinler, sayısal verilere dönüştürülerek makine öğrenmesi modeline uygun hale getirilir. Bu aşamada, `CountVectorizer` kullanılarak kelime frekansları sayısal verilere dönüştürülür.
- **Model Eğitimi:** **Naive Bayes** sınıflandırıcı modeli, eğitim verisi üzerinde eğitim yaparak **Pozitif**, **Negatif** ve **Nötr** olmak üzere üç farklı sınıfı tanır. Bu model, metinlerin duygusal tonlarını doğru bir şekilde sınıflandırabilmek için makine öğrenmesi ve yapay zeka tekniklerini kullanır.

Halihazırda 50 bin cümlelik bir veri setiyle eğitilmiştir. Yüzde 88 doğruluk oranıyla çalışır. Siz de veri ekleyerek modeli yeniden eğit tuşu ile yapay zekayı eğitebilirsiniz.

Model, **Pozitif** etiketli metinleri olumlu duyguları, **Negatif** etiketli metinleri olumsuz duyguları ve **Nötr** etiketli metinleri tarafsız veya duygusuz içerikleri temsil edecek şekilde sınıflandırır. Bu sınıflandırma işlemi, **makine öğrenmesi** algoritmalarına dayalı bir yapay zeka sisteminin işlevselliğini yansıtır. 

Proje, özellikle sosyal medya, yorumlar veya müşteri geri bildirimlerini analiz etmek isteyen uygulamalar için faydalıdır ve güçlü bir duygu analizi altyapısı sunar.

<br><br>

## Kullanılan Teknolojiler

- **Python**
- **SQLite** metin verisi saklamak için
- **Scikit-learn** makine öğrenmesi için
- **NLTK** metin ön işleme ve stopwords kaldırma için
- **Tkinter** grafiksel kullanıcı arayüzü oluşturmak için

<br><br>

## Uygulama Arayüzü

<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/fda895b6-1f31-40c6-b661-13ed6d547c07" alt="App Screenshot" width="300">
    </td>
    <td>
      Bu, uygulamanın ana ekranını gösteren bir ekran görüntüsüdür. 
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/18b77164-1ad7-4be6-8001-16730d2f04f5" alt="Example 1" width="300">
    </td>
    <td>
      Bu görüntü, metni sınıflandır tuşuna basınca çıkar. Sizden bir metin girmenizi ister.
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/c7ceedd6-3f48-460c-9fb9-bb651fcf79b7" alt="Result 1" width="300">
    </td>
    <td>
      Bu analiz sonucu, metnin negatif bir duygu taşıdığını belirtir. 
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/12f7b399-45e5-44cd-b24b-c2deff9b968a" alt="Example 2" width="300">
    </td>
    <td>
      Bu görüntü, veri yeni veri ekle tuşuna basınca görüntülenir. Sizden bir metin girmenizi ister.
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/f894c508-62fb-414d-bc1a-ad6467170ea0" alt="Result 2" width="300">
    </td>
    <td>
      Burada ise yazdığınız metnin duygusunu ister. Gönderdikten sonra veriyi veritabanına kaydeder.
    </td>
  </tr>
</table>


<br><br>

## Kurulum

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/hamza-gunes/ML_Text_Sentiment.git
   ```

2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Çalıştırın:
   ```bash
   python main.py
   ```
   
4. Hata alırsanız kütüphaneleri bu komutla da yükleyebilirsiniz:
   ```bash
   pip install numpy pandas scikit-learn joblib nltk TurkishStemmer tkinter
   ```
<br><br>

  ## Authors

  Hamza Güneş, Ferhat Köknar.
   

<br><br><br><br>

# ENGLISH:

This project is a **machine learning** and **artificial intelligence** based **sentiment analysis** system that classifies texts into three different sentiment categories: **Positive**, **Negative**, and **Neutral**. The input texts provided by the users are processed using **natural language processing (NLP)** techniques and then classified with sentiment labels through a machine learning model.

The project uses a **database** for data storage. The database is an **SQLite** database where texts and their corresponding labels are stored. Users can add texts to the system, use them as training data, and upload texts for sentiment analysis. The database stores the data used in the model's training process and also manages the texts used in classification tasks.

The stages of the project work as follows:
- **Database Integration:** Data is collected from the user and saved into the SQLite database. The database ensures that texts and their labels are stored in an organized way.
- **Text Cleaning:** Texts obtained from the database are stripped of unnecessary symbols, numbers, and special characters. Also, all texts are converted to lowercase.
- **Stemming:** Words used in the Turkish language are reduced to their roots for meaningful analysis. This step allows words in different forms to be evaluated based on their root.
- **Stopwords Removal:** Turkish stopwords (meaningless, irrelevant words) are removed from the texts. This helps the model focus on more meaningful words for sentiment analysis.
- **Vectorization:** Cleaned texts are converted into numerical data suitable for the machine learning model. In this step, the `CountVectorizer` is used to transform word frequencies into numerical data.
- **Model Training:** The **Naive Bayes** classifier model is trained on the training data to recognize the three sentiment categories: **Positive**, **Negative**, and **Neutral**. This model uses machine learning and artificial intelligence techniques to accurately classify the emotional tone of texts.

It has been trained with an existing dataset containing 50,000 sentences and operates with an accuracy of 88%. You can also contribute by adding new data and retraining the AI using the "Retrain Model" button.

The model classifies **Positive** labeled texts as positive sentiments, **Negative** labeled texts as negative sentiments, and **Neutral** labeled texts as neutral or emotionless content. This classification process reflects the functionality of an **artificial intelligence** system based on **machine learning** algorithms.

The project is especially useful for applications that want to analyze social media, comments, or customer feedback, offering a powerful sentiment analysis infrastructure.

<br><br>

## Technologies Used

- **Python**
- **SQLite** for storing text data
- **Scikit-learn** for machine learning
- **NLTK** for text preprocessing and stopwords removal
- **Tkinter** for creating the graphical user interface

<br><br>

## Application Interface

<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/fda895b6-1f31-40c6-b661-13ed6d547c07" alt="App Screenshot" width="300">
    </td>
    <td>
      <p>This is a screenshot showing the main screen of the application.</p>
      <p>"Metin analizine hoşgeldiniz." means "Welcome to text analysis."</p>
      <p>"Lütfen bir işlem seçin" means "Please select an operation"</p>
      <p>"Metni sınıflandır" means "Classify the text"</p>
      <p>"Yeni veri ekle" means "Add new data"</p>
      <p>"Modeli yeniden eğit" means "Retrain the model"</p>
      <p>"Çıkış" means "Exit"</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/18b77164-1ad7-4be6-8001-16730d2f04f5" alt="Example 1" width="300">
    </td>
    <td>
      This screen appears when you click the "Classify Text" button. It prompts you to enter a text.
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/c7ceedd6-3f48-460c-9fb9-bb651fcf79b7" alt="Result 1" width="300">
    </td>
    <td>
      This result indicates that the text has a negative sentiment.
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/12f7b399-45e5-44cd-b24b-c2deff9b968a" alt="Example 2" width="300">
    </td>
    <td>
      This screen appears when you click the "Add New Data" button. It prompts you to enter a new text.
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/f894c508-62fb-414d-bc1a-ad6467170ea0" alt="Result 2" width="300">
    </td>
    <td>
      Here, the system asks for the sentiment of the entered text. Once submitted, the data is saved to the database.
    </td>
  </tr>
</table>


<br><br>

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hamza-gunes/ML_Text_Sentiment.git
   ```

2. Install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Run the application:
   ```bash
   python main.py
   ```
   
4. If you encounter errors, you can install the libraries manually using this command:
   ```bash
   pip install numpy pandas scikit-learn joblib nltk TurkishStemmer tkinter
   ```

<br><br>

  ## Authors

  Hamza Güneş, Ferhat Köknar.
