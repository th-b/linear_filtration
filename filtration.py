# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# See the LICENSE file for more details.

import tensorflow as tf
import numpy as np

def filter_indices(a, classes):
    if not classes:
        return np.repeat(True, a.shape[0])

    return np.all([a != cl for cl in classes], axis=0)

def filtration_matrix(predictions, labels, num_classes, filter_classes, mode='normalization'):

    if mode == 'naive':
        return np.eye(num_classes,dtype='float32')[filter_indices(np.array(range(num_classes)), filter_classes)]

    A = np.stack([predictions[labels==i].mean(axis=0) for i in range(num_classes)],axis=-1)
    B = A[filter_indices(np.array(range(num_classes)), filter_classes)]
      
    if mode == 'normalization':
        target = B[:,filter_indices(np.array(range(num_classes)), filter_classes)]
        for filter_class in filter_classes:
            x = B[:,filter_class]
            x = x + target.mean() - x.mean()
            B[:,filter_class] = x 
    elif mode == 'randomization':
        for filter_class in filter_classes:
            B[:,filter_class] = np.random.randn(num_classes-len(filter_classes)) 
    elif mode == 'zeroing':
        for filter_class in filter_classes:
            B[:,filter_class] = np.zeros(num_classes-len(filter_classes))
    else:
        print('Unknown filtration mode \'{}\''.format(mode))
        assert(False)
        
    return np.matmul(B, np.linalg.inv(A))

def get_weights(layer):
    W = layer.weights[0]
    b = layer.weights[1]

    W = W.numpy()
    b = b.numpy()
    return [W, b]

def set_weights(layer, weights):
    W, b = weights
    
    W = tf.convert_to_tensor(W)
    b = tf.convert_to_tensor(b)

    layer.set_weights([W,b])

def logit_sparse_categorical_crossentropy(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def make_filtered_model(model, data, labels, num_classes, filter_classes, mode='normalization', lin_index=-2, apply_softmax=True):
    input =  model.layers[0].input
    features = model.layers[lin_index-1].output

    lin_layer = model.layers[lin_index]

    m_presm = tf.keras.Model(inputs = input, outputs = lin_layer.output)
    m_presm.compile(loss=logit_sparse_categorical_crossentropy)
    predictions = m_presm.predict(data)
    
    filtered_lin_layer = tf.keras.layers.Dense(num_classes - len(filter_classes))
    filtered_lin_layer.build(features.shape)
    
    F = filtration_matrix(predictions, labels, num_classes, filter_classes, mode=mode)
    W, b = get_weights(lin_layer)
    W = np.matmul(W, np.transpose(F))
    b = np.matmul(b, np.transpose(F))
    set_weights(filtered_lin_layer, [W,b])

    x = filtered_lin_layer(features)
    loss = tf.keras.losses.categorical_crossentropy
    if apply_softmax:
        x = tf.keras.layers.Activation('softmax')(x)
        loss = logit_sparse_categorical_crossentropy

    m_filtered = tf.keras.Model(inputs = input, outputs = x)
    m_filtered.compile(loss=loss)

    return m_filtered