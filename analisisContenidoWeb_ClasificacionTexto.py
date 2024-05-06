# Importar las bibliotecas necesarias
import urllib.request  # Para realizar solicitudes HTTP y obtener contenido de una URL
from bs4 import BeautifulSoup  # Para analizar documentos HTML y extraer información de ellos
import nltk  # Para tareas de análisis de texto
from nltk.tokenize import word_tokenize  # Para dividir el texto en palabras individuales (tokens)
from nltk.corpus import stopwords  # Para obtener la lista de palabras vacías (stopwords)
import matplotlib.pyplot as plt  # Para visualización de datos
import seaborn as sns  # Para mejorar el estilo visual de los gráficos
from nltk.corpus import wordnet # Importar la clase wordnet del módulo nltk.corpus para acceder a WordNet
from nltk.stem import SnowballStemmer # Importar la clase SnowballStemmer del módulo nltk.stem para realizar la derivación regresiva en español
import spacy

# Cargar el modelo de español en spaCy
nlp = spacy.load("es_core_news_sm")

# Descargar recursos adicionales de NLTK si no están descargados previamente
nltk.download('punkt')  # Tokenizador Punkt
nltk.download('stopwords')  # Stopwords en español

# URL de la página web a la que se desea acceder
url = 'https://librefinanciero.com'

# Configurar el User-Agent en los encabezados para simular una solicitud de navegador
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# Crear una solicitud de acceso a la URL con los encabezados configurados
request = urllib.request.Request(url, headers=headers)

try:
    # Intentar abrir la URL y obtener la respuesta
    response = urllib.request.urlopen(request)
    
    # Leer el contenido HTML de la respuesta
    html = response.read()
    
    # Crear un objeto BeautifulSoup para analizar el HTML
    soup = BeautifulSoup(html, "html.parser")
    
    # Obtener el texto del HTML eliminando etiquetas y espacios innecesarios
    text = soup.get_text(strip=True)
    
    # Imprimir el texto obtenido
    print(text)
    
except urllib.error.HTTPError as e:
    # Manejar errores de HTTP, en caso de que ocurran
    print("Error al acceder al URL:", e)

# Tokenizar el texto en palabras utilizando el tokenizador para el idioma español
tokens = word_tokenize(text, "spanish")

# Filtrar las palabras para eliminar signos de puntuación y convertir todas las palabras a minúsculas
tokens = [word.lower() for word in tokens if word.isalpha()]

# Eliminar las stopwords del texto tokenizado
clean_tokens = [word for word in tokens if word not in stopwords.words('spanish')]

# Calcular la frecuencia de cada palabra en los tokens limpios
freq_clean = nltk.FreqDist(clean_tokens)

# Imprimir las palabras junto con su frecuencia
for key, val in freq_clean.items():
    print(str(key) + ':' + str(val))

# Visualizar las 30 palabras más comunes
sns.set()  # Establecer el estilo predeterminado de Seaborn
plt.figure(figsize=(10, 6))
freq_clean.plot(30, cumulative=False)
plt.show()

# Inicialización de una lista vacía para almacenar los sinónimos
synonyms = []


# Obtener sinónimos de la palabra "investment" usando WordNet
for syn in wordnet.synsets('investment'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())

# Mantener una copia de los tokens limpios originales
clean_tokens_sin = clean_tokens.copy()

# Reemplazar tokens sinónimos
for ind, sin in enumerate(synonyms):
    clean_tokens_sin = [word.replace(synonyms[ind], 'investment') for word in clean_tokens_sin]

# Reemplazar tokens sinónimos manualmente
sinonimos = ['libertad', 'independencia']
for ind, sin in enumerate(sinonimos):
    clean_tokens_sin = [word.replace(sinonimos[ind], 'libertad') for word in clean_tokens_sin]

# Calcular la frecuencia de palabras con sinónimos agregados
freq_clean_sin = nltk.FreqDist(clean_tokens_sin)

# Imprimir las palabras junto con su frecuencia después de agregar sinónimos
for key, val in freq_clean_sin.items():
    print(str(key) + ':' + str(val))

# Visualizar Tokens con sinónimos
plt.figure(figsize=(10, 6))
freq_clean_sin.plot(30, cumulative=False)
plt.show()
 
# Inicializar una lista vacía para almacenar los antónimos
antonyms = []
 
# Iterar sobre todos los synsets (conjuntos de sinónimos) de la palabra "good" en WordNet
for syn in wordnet.synsets("good"):
    # Iterar sobre todas las palabras lematizadas en el synset actual
    for l in syn.lemmas():
        # Verificar si hay antónimos para la palabra actual
        if l.antonyms():
            # Si hay antónimos, añadir el primer antónimo encontrado a la lista
            antonyms.append(l.antonyms()[0].name())
 
# Imprimir la lista de antónimos encontrados
print(antonyms)
 
# Crear una instancia de SnowballStemmer para el español
spanish_stemmer = SnowballStemmer('spanish')
 
# Aplicar la derivación regresiva a las palabras "trabajando" y "trabajo" y mostrar los resultados
print(spanish_stemmer.stem("trabajando"))
print(spanish_stemmer.stem("trabajo"))

# Derivación regresiva en español para tokens
clean_tokens_sin_stems = [spanish_stemmer.stem(token) for token in clean_tokens_sin]

clean_tokens_sin_stems

# Recalcular frecuencia de palabras con sinónimos agregados y derivación regresiva
freq_clean_sin_stems = nltk.FreqDist(clean_tokens_sin_stems)

# Imprimir las palabras junto con su frecuencia después de agregar sinónimos y derivación regresiva
for key, val in freq_clean_sin_stems.items():
    print(str(key) + ':' + str(val))

# Visualizar Tokens derivados con sinónimos y derivación regresiva
freq_clean_sin_stems.plot(30, cumulative=False)

# Inicializar una lista vacía para almacenar los lemas de las palabras
clean_tokens_sin_lem = []

# Definir una cadena de palabras
palabras = 'comer comida comería comido comen'

# Iterar sobre cada token lematizado generado por spaCy
for token in nlp(palabras): 
    # Imprimir el texto y la parte del discurso de cada token
    print(token.text, token.pos_)
    
    # Agregar el lema de cada token a la lista clean_tokens_sin_lem
    clean_tokens_sin_lem.append(token.lemma_)
    
# Imprimir la lista de lemas resultantes
print(clean_tokens_sin_lem)