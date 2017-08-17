from tkinter import *
from net import *
from PIL import Image

def draw(event, size=20):
    prediction['text'] = ''
    certainty['text'] = ''
    canvas.create_oval(event.x-size, event.y-size, event.x+size, event.y+size, fill ='#000')
    for i in range(int(1.5*size)):
        for j in range(int(1.5*size)):
            try:
                root.image.putpixel(((event.x-int(size/2)+i), (event.y-int(size/2)+j)), (0, 0, 0))
            except IndexError:
                pass

def run():
    img = root.image.resize((16, 16), Image.ANTIALIAS).resize((8, 8), Image.ANTIALIAS)
    data = []
    for i in range(64):
        data.append(sum(img.getpixel((i%8, i//8))) / (3 * 255))
    output = net.run(data)
    index = list(output).index(max(output))
    prediction['text']  = index
    certainty['text'] = '{0:.9f}'.format(output[index])

    """num = 9
    path = 'images/8px_2/test/%r/' % num
    files = os.listdir(path)
    if not files:
        next = 0
    else:
        next = max([int(a.split('.')[0]) for a in files]) + 1
    img.save(path + '%r.png' % next)

    prediction['text'] = ''
    certainty['text'] = ''
    canvas.create_rectangle(-10, -10, 310, 310, fill='#fff')
    root.image = Image.new('RGB', (300, 300), 'white')"""


def clear():
    prediction['text'] = ''
    certainty['text'] = ''
    canvas.create_rectangle(-10, -10, 310, 310, fill='#fff')
    root.image = Image.new('RGB', (300, 300), 'white')

root = Tk()
root.resizable(width=False, height=False)
root.minsize(width=500, height=300)
root.maxsize(width=500, height=300)
root.title('Number classification')
canvas = Canvas(root, width=300, height=300)
canvas.pack()
canvas.place(x=0, y=0)

root.image = Image.new('RGB', (300, 300), 'white')
net = Net.load('trained_weights_16px_2.npy')

canvas.bind('<B1-Motion>', draw)
canvas.create_rectangle(0, 0, 310, 310, fill='#fff')

prediction_label = Label(root, text='Prediction')
prediction_label.pack()
prediction_label.place(x=320, y=20)

prediction = Label(root, text='')
prediction.pack()
prediction.place(x=400, y=20)

certainty_label = Label(root, text='Certainty')
certainty_label.pack()
certainty_label.place(x=320, y=40)

certainty = Label(root, text='')
certainty.pack()
certainty.place(x=400, y=40)

clear = Button(root, text="Clear", command=clear)
clear.pack()
clear.place(x=320, y=80)

clear = Button(root, text="Run", command=run)
clear.pack()
clear.place(x=360, y=80)

mainloop()
