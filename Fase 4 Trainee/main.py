
import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

# Download dos recursos do NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Função para pré-processamento dos dados
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Exemplo de textos pré-processados (substitua por seus próprios dados)
corpus = [
    "Not much to write about here, but it does exactly what it's supposed to. filters out the pop sounds. now my recordings are much more crisp. it is one of the lowest prices pop filters on amazon so might as well buy it, they honestly work the same despite their pricing,",
    "The product does exactly as it should and is quite affordable. I did not realize it was double screened until it arrived, so it was even better than I had expected. As an added bonus, one of the screens carries a small hint of the smell of an old grape candy I used to buy, so for reminiscent's sake, I cannot stop putting the pop filter next to my nose and smelling it after recording. :D If you needed a pop filter, this will work just as well as the expensive ones, and it may even come with a pleasing aroma like mine did! Buy this product! :]",
    "The primary job of this device is to block the breath that would otherwise produce a popping sound, while allowing your voice to pass through with no noticeable reduction of volume or high frequencies. The double cloth filter blocks the pops and lets the voice through with no coloration. The metal clamp mount attaches to the mike stand secure enough to keep it attached. The goose neck needs a little coaxing to stay where you put it.",
    "Nice windscreen protects my MXL mic and prevents pops. Only thing is that the gooseneck is only marginally able to hold the screen in position and requires careful positioning of the clamp to avoid sagging.",
    "This pop filter is great. It looks and performs like a studio filter. If you're recording vocals this will eliminate the pops that get recorded when you sing.",
    "So good that I bought another one. Love the heavy cord and gold connectors. Bass sounds great. I just learned last night how to coil them up. I guess I should read instructions more carefully. But no harm done, still works great!",
    "I have used monster cables for years, and with good reason. The lifetime warranty is worth the price alone. Simple fact: cables break, but getting to replace them at no cost is where it's at.",
    "I now use this cable to run from the output of my pedal chain to the input of my Fender Amp. After I bought Monster Cable to hook up my pedal board I thought I would try another one and update my guitar...",
    "Monster makes the best cables and a lifetime warranty doesn't hurt either. This isn't their top of the line series but it works great with my bass guitar rig and has for some time. You can't go wrong with Monster Cables.",
    "Monster makes a wide array of cables, including some that are very high end. I initially purchased a pair of Monster Rock Instrument Cable - 21 Feet - Angled to Straight 1/4-Inch plug to use with my keyboards...",
    "I got it to have it if I needed it. I have found that I don't really need it that often and rarely use it. If I was really good I can see the need. But this is a keyboard not an organ.",
    "If you are not used to using a large sustaining pedal while playing the piano, it may appear a little awkward.",
    "I love it, I used this for my Yamaha ypt-230 and it works great, I would recommend it to anyone",
    "I bought this to use in my home studio to control my midi keyboard. It does just what I wanted it to do.",
    "I bought this to use with my keyboard. I wasn't really aware that there were other options for keyboard pedals...",
    "The Hosa XLR cables are affordable and very heavily made. I have a large mixer and rack and cables everywhere...",
    "I bought these to go from my board to the amp. We use them for a mobile church so they take a beating. They are still going strong.",
    "Sturdy cord and plugs, inexpensive, good value. I don't require professional-level equipment, so this cord serves my purposes well. Satisfied with purchase.",
    "Use it every week at gigs. Solid, no problems with the solder joints. A good quality cable at a very good price.",
    "Hosa products are a good bang for the buck. I haven't looked up the specifications, but I'm guessing the wire is 22 to 24 AWG, but since it's only 10' long, it's good enough.",
    "This was exactly what I was after. I have a voice touch and needed a small cord to connect the mic to the voice touch and this was perfect. Before I used a 20 foot cord to go about 12 inches...",
    "I bought these because I really had too long of mike cords for my solo live show. And these are really nice cords if you have a home portastudio recording studio like myself...",
    "This cable seems like it will last me for a while. As it is only being used to connect a DI box it will not get abused as much as the vocal mics always do.",
    "These are not the greatest but they're cheap and they get to you fast when you need them. I've only had one fail and I've bought many of them to use in our broadcast studio...",
    "This is a fine cable at a decent price point, nothing exceptional mind, but it gets the job done well enough.",
    "I've used a lot of cables and I always come back to HOSA, they are indeed some of the best audio cables in their price range on the market.",
    "I bought this cord after returning a cheap one that I should've known better than to buy. My son, who has some experience as a musician recommended the Hosa brand when I was seeking a proper replacement.",
    "Nice solid cables, with excellent support at the ends. Should last a lifetime of usage no problem and just what I needed to connect my tube preamp.",
    "Good quality cable and sounds very good",
    "Zero issues with this cable so far. It feels fairly cheap and light weight but it has survived for months of plugging in, unplugging, and packing between practice spaces.",
    "Relatively inexpensive patch cable for electric guitar. I have had it for a few months and so far it has held up pretty well.",
    "I bought this because I wanted a cheap replacement cable for one that had a short. I'm pleasantly surprised with this cable. It's decent sound and decent build quality, for a good price.",
    "This is a very nice cable for the price. I already bent one end of it though, fortunately it still works fine. Inside the phono connector the wires are covered by white shrink plastic. Haven't noticed any hum or crackles. For sure a good buy.",
    "Cheap and good texture rubber that does not get stiff. Only time will tell how well the soldering is. Sounds fine to me."
    "Seems sturdy enough, and no noise issues, so I'm pretty much satisfied with it"
]

#Aplicar pré-processamento aos textos
preprocessed_corpus = [preprocess_text(text) for text in corpus]

#Inicializar o vetorizador BoW e TF-IDF
bow_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

#Ajustar o vetorizador aos dados e transformar os textos em vetores
bow_matrix = bow_vectorizer.fit_transform(preprocessed_corpus)
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_corpus)

labels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]

#Teste e treino
X_train, X_test, y_train, y_test = train_test_split(bow_matrix, labels, test_size=0.2, random_state=42)



#Treinar e avaliar um classificador Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}")

#Treinar e avaliar um classificador SVM
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy:.2f}")

#Preparar diretórios para o formato do TensorFlow
train_data_dir = 'C:/Users/desktop/PycharmProjects/4fase/animals'
test_data_dir = 'C:/Users/desktop/PycharmProjects/4fase/animals'
image_size = (224, 224)
batch_size = 32

#Rede Convolucional do Zero
model_zero = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])
model_zero.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Transfer Learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)
model_transfer = Model(inputs=base_model.input, outputs=output)
model_transfer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

#Renderização de modelos
num_epochs = 2
model_zero.fit(train_generator, epochs=num_epochs)
