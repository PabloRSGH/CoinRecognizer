import cv2
import numpy as np
from keras.models import load_model
from tkinter import filedialog
from tkinter import Tk

#tf 2.9.1
#keras 2.6.0

# Abre un cuadro de diálogo para seleccionar la imagen
root = Tk()
root.withdraw() # Oculta la ventana de Tkinter
ruta_imagen = filedialog.askopenfilename() # Abre el cuadro de diálogo

# Carga la imagen
img = cv2.imread(ruta_imagen)

# Carga el modelo previamente entrenado
model = load_model('Keras_model.h5',compile=False)

# Prepara un array vacío para los datos de la imagen
data = np.ndarray(shape=(1,224,224,3),dtype=np.float32)

# Define las clases de monedas que el modelo puede reconocer
classes = ["5 pesos","10 pesos"]

# Función para preprocesar la imagen
def preProcess(img):
    # Aplica un desenfoque gaussiano
    imgPre = cv2.GaussianBlur(img,(5,5),3)
    # Detecta los bordes de la imagen
    imgPre = cv2.Canny(imgPre,90,140)
    # Prepara un kernel para las operaciones de dilatación y erosión
    kernel = np.ones((4,4),np.uint8)
    # Dilata la imagen (hace los bordes más gruesos)
    imgPre = cv2.dilate(imgPre,kernel,iterations=2)
    # Erode la imagen (hace los bordes más delgados)
    imgPre = cv2.erode(imgPre,kernel,iterations=1)
    return imgPre

# Función para detectar la moneda en la imagen
def DetectarMoneda(img):
    # Redimensiona la imagen a 224x224 píxeles
    imgMoneda = cv2.resize(img,(224,224))
    # Convierte la imagen a un array
    imgMoneda = np.asarray(imgMoneda)
    # Normaliza los valores de los píxeles de la imagen
    imgMonedaNormalize = (imgMoneda.astype(np.float32)/127.0)-1
    # Almacena la imagen normalizada en el array de datos
    data[0] = imgMonedaNormalize
    # Usa el modelo para predecir la clase de la moneda en la imagen
    prediction = model.predict(data)
    # Obtiene el índice de la clase con la mayor probabilidad
    index = np.argmax(prediction)
    # Obtiene la probabilidad de la clase con la mayor probabilidad
    percent = prediction[0][index]
    # Obtiene el nombre de la clase con la mayor probabilidad
    classe = classes[index]
    return classe,percent

# Redimensiona la imagen a 640x480 píxeles
img = cv2.resize(img,(640,480))
# Preprocesa la imagen
imgPre = preProcess(img)
# Encuentra los contornos en la imagen preprocesada
countors,hi = cv2.findContours(imgPre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# Inicializa la cantidad total de dinero detectado
qtd = 0
# Itera sobre cada contorno encontrado
for cnt in countors:
    # Calcula el área del contorno
    area = cv2.contourArea(cnt)
    # Si el área del contorno es mayor que 2000
    if area > 2000:
        # Obtiene el rectángulo delimitador del contorno
        x,y,w,h = cv2.boundingRect(cnt)
        # Dibuja el rectángulo delimitador en la imagen
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # Recorta la imagen en el rectángulo delimitador
        recorte = img[y:y +h,x:x+ w]
        # Detecta la moneda en el recorte de la imagen
        classe, conf = DetectarMoneda(recorte)
        # Si la confianza de la detección es mayor que 0.85
        if conf >0.85:
            # Escribe el nombre de la clase en la imagen
            cv2.putText(img,str(classe),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            # Suma el valor de la moneda detectada a la cantidad total
            if classe == '5 pesos': qtd+=5
            if classe == '10 pesos': qtd += 10

# Dibuja un rectángulo en la imagen para mostrar la cantidad total de dinero
cv2.rectangle(img,(430,30),(600,80),(0,0,255),-1)
# Escribe la cantidad total de dinero en la imagen
cv2.putText(img,f'R$ {qtd}',(440,67),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)

# Muestra la imagen original y la imagen preprocesada
cv2.imshow('IMG',img)
cv2.imshow('IMG PRE', imgPre)
# Espera a que el usuario cierre la ventana
cv2.waitKey(0)
# Cierra todas las ventanas abiertas
cv2.destroyAllWindows()
