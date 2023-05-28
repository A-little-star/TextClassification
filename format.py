import numpy as np
def format(path='./data/baike.txt'):
    np.random.seed(2021)
    raw_data = open(path, 'r', encoding='utf-8').readlines()

    num_samples = len(raw_data)
    idx = np.random.permutation(num_samples)
    num_train = int(0.8 * num_samples)
    num_test = num_samples - num_train
    train_idx, test_idx = idx[:num_train], idx[-num_test:]
    f_train = open('./data/data_train.txt', 'w', encoding='utf-8')
    f_test = open('./data/data_test.txt', 'w', encoding='utf-8')

    for i in train_idx:
        r = raw_data[i].strip('\n').split('_!_')
        label, text = r[1], r[0]
        f_train.write(text + '_!_' + label + '\n')
    f_train.close()

    for i in test_idx:
        r = raw_data[i].strip('\n').split('_!_')
        label, text = r[1], r[0]
        f_test.write(text + '_!_' + label + '\n')
    f_test.close()

if __name__ == '__main__':
    format()