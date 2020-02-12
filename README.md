# Linear filtration

The file `filtration.py` contains an implementation of linear filtration as described in [Machine Unlearning: Linear Filtration for Logit-based Classifiers](https://arxiv.org/abs/2002.02730). Please cite this paper if you are going to use the code in this repository.

The function `filtration_matrix()` calculates the filtration matrix. Its arguments are:
- `predictions` sample used to estimate mean predictions.
- `labels` i-th entry is the true class label of the i-th prediction.
- `num_classes` number of classes (before filtration).
- `filter_classes` array of class labels to unlearn, e.g. [3,5] to unlearn classes 3 and 5.
- `mode` choose from 'naive', 'normalization', 'randomization', or 'zeroing'.

The function `make_filtered_model()` applys filtration to a model. Its arguments are:
- `model` a Keras model to apply filtration to.
- `data`, `labels`, `num_classes`, `filter_classes`, `mode` see description of filtration_matrix().
- `lin_index` the index of the last dense layer in your network. Note that if you use tf.keras.layers.Activation('softmax') Keras counts it as its own layer, so `lin_index` should be -2 in this case. If you use tf.keras.layers.Dense(n, activation='softmax') `lin_index` should be -1.
- `apply_softmax` whether to apply softmax activations to the filtered model.

# Example: model inversion for AT&T Faces
The file `att_faces_example.ipynb` contains an example where we apply linear filtration a model trained on
the [AT&T Laboratories Cambridge Database of Faces](http://cam-orl.co.uk/facedatabase.html), to investigate its effect on reconstructions of face images
by gradient ascent ("model inversion").