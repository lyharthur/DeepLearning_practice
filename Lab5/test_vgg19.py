from pre_train_model_vgg19 import VGG_19
import cv2, numpy as np
import json
from keras.optimizers import SGD
import sys


fpath = 'imagenet_class_index.json'

img = sys.argv[1]

im = cv2.resize(cv2.imread(img), (224, 224)).astype(np.float32)
im[:,:,0] -= 123.68
im[:,:,1] -= 116.779
im[:,:,2] -= 103.939
#im = im.transpose((2,0,1))

#print(im)

im = np.expand_dims(im, axis=0)
#print(im)

    # Test pretrained model
model = VGG_19((224,224,3))
model.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

#print(model.summary())
sgd = SGD(lr=0.0, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
preds = model.predict(im)

CLASS_INDEX = json.load(open(fpath))

results = []
for pred in preds:
    top_indices = pred.argsort()[-5:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
    results = [item for sublist in results for item in sublist]


#print('top?,  class,  proba')
print('Top1: ',results[0][1],',  ',results[0][2])
print('Top2: ',results[1][1],',  ',results[1][2])
print('Top3: ',results[2][1],',  ',results[2][2])
print('Top4: ',results[3][1],',  ',results[3][2])
print('Top5: ',results[4][1],',  ',results[4][2])

print('done')

#print(np.argmax(out))
#print(out.argsort()[-10:])


