from net import *
from data import *
from datetime import datetime

def fprint(f, txt):
    f.write(txt + '\n')
    print(txt)

def main(dataset, img_size, categories, hidden_layers):
    layers = [int(img_size[0]*img_size[1])]
    layers.extend(hidden_layers)
    layers.append(len(categories))
    net = Net(layers)

    # Train neural network with train-data
    input_data, output_data = load_training_data(dataset, categories, img_size)
    print('Iteration'.ljust(9), 'Error')
    error = 1.0
    i = -1
    #while datetime.now().hour <= 18:
    while error > 0.0005:
        i += 1
        net.train(input_data, output_data)
        error = net.mean_error(output_data)
        if i % 10000 == 0:
            print(str(i).rjust(9), '{0:.9f}'.format(error))
    print(str(i).rjust(9), '{0:.9f}'.format(error))

    # Save neural network
    net.save('weights.npy')

    # Test neural network with test-data
    values = load_testing_data(dataset, categories, img_size)
    print()
    with open('result.txt', 'w') as f:
        fprint(f, ''.join(map(lambda a: a.rjust(8, ' '), categories)))
        for i in range(len(categories)):
            for b in  net.run(values[i]):
                fprint(f, categories[i].ljust(2, ' ') + ''.join(['{0:.4f}'.format(a).rjust(8) for a in b]))
        fprint(f, '')
        fprint(f, ''.join(map(lambda a: a.rjust(8, ' '), categories)))
        for i in range(len(categories)):
            for b in net.run(values[i]):
                fprint(f, categories[i].ljust(2, ' ') + ''.join([('x  ' if a >= 0.5 else '').rjust(8) for a in b]))


main('16px', (8, 8), list(map(str, range(10))), [32, 16])
