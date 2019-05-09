from tkinter import *
from PIL import ImageDraw, Image
import numpy as np
from nnetwork.feed_forward import feed_forward
from nnetwork.utils import confidence_and_prediction

predictions = ['Circulo', 'Cuadrado', 'Triangulo', 'Huevo', 'Arbol', 'Casa', 'Cara Feliz',
               'Cara Triste', 'Signo de Interrogacion', 'Mickey Mouse']

weights = np.load('data/weights.npy', allow_pickle=True)


def draw(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
    imgD.line([x1, y1, x2, y2], fill="black", width=5)


def predict():
    small_img = img.resize((28, 28))
    arr = np.array(small_img).reshape(1, 784)
    A = feed_forward(arr, weights)
    conf_and_pred = confidence_and_prediction(A[-1])
    result.set("Con un {} de confianza, creo que el dibujo es un/una {}".format(
        round(conf_and_pred[0][0], 2),
        predictions[conf_and_pred[0][1]]))
    window.update_idletasks()
    print(predictions[conf_and_pred[0][1]])


def clean():
    canvas.delete('all')


window = Tk()

# window.geometry('512x512')
window.resizable(False, False)
window.title('CroquisRecon')

canvas = Canvas(window, bg='white', cursor='dot', height=512, width=512)
canvas.grid(column=0, row=0)

frame = Frame(window)
frame.grid(column=0, row=1)

predictBtn = Button(frame, text='Predecir', command=predict)
predictBtn.grid(column=1, row=0)

cleanBtn = Button(frame, text='Limpiar', command=clean)
cleanBtn.grid(column=0, row=0)

img = Image.new('L', (512, 512), 255)
imgD = ImageDraw.Draw(img)

frame2 = Frame(window)
frame2.grid(column=0, row=2)

result = StringVar()
label = Label(frame2, textvariable=result, relief=RAISED)
label.pack()

canvas.bind("<B1-Motion>", draw)

window.mainloop()
