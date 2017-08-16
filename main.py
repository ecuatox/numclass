import os
from net import *
from image import *


def main(dataset, img_size, categories, hidden_layers):
    layers = [int(img_size[0]*img_size[1])]
    layers.extend(hidden_layers)
    layers.append(len(categories))
    net = Net(layers)

    images = [['images/%s/learn/%s/' % (dataset, categories[i]) + a for a in os.listdir('images/%s/learn/%s/' % (dataset, categories[i]))]
        for i in range(len(categories))]
    size = min(map(len, images))
    values = [[read_image(img, img_size) for img in images[i][:size]]
              for i in range(len(categories))]

    input_data, output_data = [], []

    for i in range(len(categories)):
        input_data.extend(values[i])

    for i in range(len(categories)):
        output = np.zeros(len(categories))
        output[i] = 1
        output_data.extend([output for _ in range(size)])

    output_data = np.array(output_data)

    np.random.seed(0)
    np.random.shuffle(input_data)
    np.random.seed(0)
    np.random.shuffle(output_data)

    error = 1.0
    i = -1
    while error > 0.01:
        i += 1
        net.train(input_data, output_data)
        error = net.mean_error(output_data)
        if i % 10000 == 0:
            print(i, error)
    print(i, error)

    np.save('weights.npy', net.weights)

    images = [['images/%s/test/%s/' % (dataset, categories[i]) + a for a in
               os.listdir('images/%s/test/%s/' % (dataset, categories[i]))]
              for i in range(len(categories))]
    values = [[read_image(img, img_size) for img in images[i][:size]]
              for i in range(len(categories))]
    print()
    print(''.join(map(lambda a: a.rjust(8, ' '), categories)))
    for i in range(len(categories)):
        for b in  net.run(values[i]):
            print(categories[i].ljust(2, ' ') + ''.join(['{0:.4f}'.format(a).rjust(8) for a in b]))
    print()
    print(''.join(map(lambda a: a.rjust(8, ' '), categories)))
    for i in range(len(categories)):
        for b in net.run(values[i]):
            print(categories[i].ljust(2, ' ') + ''.join([('x  ' if a >= 0.5 else '').rjust(8) for a in b]))


main('numbers', (3, 3), list(map(str, range(4))), [16])
