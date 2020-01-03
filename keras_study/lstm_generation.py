import os
import numpy as np

data_dir_path = "F:\\ML Dataset"
data_path = os.path.join(data_dir_path, 'nietzsche.txt')

text = open(data_path, encoding='UTF-8').read().lower()
print("Corpus length:", len(text))

maxlen = 60
step = 3
sentences = []
next_data = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])
    next_data.append(text[i + maxlen])

print('Number of sequences:', len(sentences))

chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
char_indices = dict((char, chars.index(char)) for char in chars)

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_data[i]]] = 1

from keras import layers, models, optimizers

model = models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))
print(model.summary())
model.compile(optimizer=optimizers.RMSprop(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])


# model.fit(x, y, batch_size=128, epochs=60, validation_split=0.3)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # 第一个参量是取得个数，第二个参量是概率分布，第三个参量是生成多少个
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


import random, sys

import random
import sys

for epoch in range(1, 60):
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1, verbose=0)
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)
        # 从种子文本开始生成400个字符
        for i in range(400):
            # 对生成的字符进行one-hot 编码
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)
