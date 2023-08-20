# Preocess

The first cell of code imports various modules and libraries that are needed for the task. Some of them are:

- `tensorflow` and `tensorflow_quantum`: These are the main modules for quantum computing with TensorFlow. They provide functions and classes to create quantum circuits, tensors, models, layers, and optimizers.
- `cirq`: This is the module for creating and manipulating quantum circuits using Python objects. It also provides simulators and noise models to test the circuits.
- `numpy` and `matplotlib`: These are common modules for scientific computing and data visualization. They provide functions and classes to manipulate arrays, matrices, vectors, and plots.
- `cv2`, `PIL`, and `skimage`: These are modules for image processing. They provide functions and classes to read, write, resize, crop, filter, transform, and display images.
- `os` and `glob`: These are modules for file handling. They provide functions and classes to access, create, delete, rename, and search files and directories.
- `tqdm`: This is a module for progress bar. It provides a function to display a progress bar for loops and iterations.

The second cell does the following steps:

- It defines the paths to the dataset folders, where the original images and the groundtruths for each segment are stored.
- It defines the subfolders for each segment, which are named according to the type of lesion.
- It defines the image size and number of channels, which are 256 x 256 pixels and 3 channels (RGB) respectively.
- It defines a function to load and preprocess an image, which involves reading the image as a numpy array, resizing it to the desired size, converting it from BGR to RGB color space, and normalizing it to have values between 0 and 1.
- It defines a function to load and preprocess a groundtruth, which involves reading the groundtruth as a grayscale numpy array, resizing it to the desired size, thresholding it to have binary values (0 or 1), and normalizing it to have values between 0 and 1.
- It defines empty lists to store the images and groundtruths.
- It loops over each segment subfolder, and gets the list of image files and groundtruth files in each subfolder.
- It loops over each image file and its corresponding groundtruth file, and loads and preprocesses them using the defined functions. It then appends them to the lists.
- It converts the lists to numpy arrays, which can be used for further processing or modeling.
- It prints the shapes of the arrays, which should be (number of images, 256, 256, 3) for images and (number of images, 256, 256) for groundtruths.

The 3rd cell code does the following steps:

- It defines the number of qubits to use for encoding, which is 5 in this case. This means that each image will be encoded into a 5-dimensional quantum state, and each groundtruth will be encoded into a one-dimensional quantum state.
- It defines a function to encode an image into a quantum state using amplitude encoding. The function takes an image as input, which is a tensor of shape (`img_size`, `img_size`, `img_channels`), where `img_size` is the image size and `img_channels` is the number of color channels. The function does the following:
- It reshapes the image into a 1D array of length `img_size` * `img_size` * `img_channels`.
- It normalizes the image to have unit norm, which is required for amplitude encoding.
- It creates a circuit using Cirq, a Python library for creating and simulating quantum circuits.
- It adds a state preparation gate for each qubit, which is a rotation gate around the y-axis with an angle equal to the inverse cosine of the corresponding image element. This gate transforms the qubit from the |0⟩ state to the desired amplitude state.
- It defines a function to encode a batch of images into quantum data using amplitude encoding. The function takes a batch of images as input, which is a tensor of shape (`batch_size`, `img_size`, `img_size`, `img_channels`), where `batch_size` is the number of images in the batch. The function does the following:
- It creates an empty list to store the circuits.
- It loops over each image in the batch and encodes it into a circuit using the previous function.
- It appends the circuit to the list.
- It converts the list of circuits to a tensor using TensorFlow Quantum, a library that integrates TensorFlow and Cirq for quantum computing.
- It returns the tensor of circuits, which is a quantum data type that can be used for further processing or modeling.
- It encodes the images and groundtruths into quantum data using amplitude encoding by calling the previous function on them. The images and groundtruths are tensors of shape (`number_of_images`, `img_size`, `img_size`, `img_channels`) and (`number_of_images`, `img_size`, `img_size`) respectively, where `number_of_images` is the total number of images and groundtruths in the dataset.
- It prints the shapes of the quantum data tensors, which should be (`number_of_images`,) for both images and groundtruths.


The fourth cell does the following steps:

- It defines the number of filters to use, which is 4 in this case. This means that there will be 4 different quantum circuits that will act as filters for the quantum data.
- It defines a function to create a parameterized quantum circuit that acts as a convolution filter. The function does the following:
- It creates a circuit using Cirq, a Python library for creating and simulating quantum circuits.
- It adds a layer of Hadamard gates to create superposition on each qubit. This means that each qubit will be in a state of 0 and 1 at the same time, with equal probability.
- It adds a layer of parameterized rotation gates to create entanglement and interference on each qubit. This means that each qubit will be in a state of 0 and 1 with different probabilities, depending on the value of the rotation angle. The rotation angle is a trainable variable that can be optimized using TensorFlow.
- It defines a function to create a measurement operator that defines the output of the filter. The function does the following:
- It creates an empty list to store the measurement operators.
- It loops over each filter and creates a random subset of qubits to measure. This means that only some of the qubits will contribute to the output of the filter, while the others will be ignored.
- It creates a Pauli sum of Z operators on the selected qubits. This means that each qubit will be measured along the Z axis, which can give either +1 or -1 as the result. The sum of these results will be the output of the filter.
- It creates a list of filter circuits using the `create_filter_circuit` function. This means that there will be 4 different quantum circuits with different rotation angles as parameters.
- It creates a list of output operators using the `create_output_operator` function. This means that there will be 4 different measurement operators with different subsets of qubits to measure.
- It prints the filter circuits and output operators for each filter. This shows how the quantum circuits and measurements look like.
- 

The fifth cell does the following steps:

- It defines the batch size and number of epochs, which are hyperparameters that control how the model is trained. The batch size is the number of samples that are processed in each iteration, and the epochs is the number of times that the model goes through the entire dataset.
- It defines a function to create a quantum convolutional layer, which is a layer that applies a quantum convolution filter to the quantum data. The function does the following:
- It creates a sequential layer using TensorFlow Keras, a high-level API for building and training neural networks.
- It adds a quantum convolution filter using the PQC layer from TensorFlow Quantum, a library that integrates TensorFlow and Cirq for quantum computing. The PQC layer takes a parameterized quantum circuit and a measurement operator as inputs, and outputs a scalar value for each sample. The quantum circuit acts as the filter that transforms the quantum data, and the measurement operator defines how the output is obtained from the quantum state. The function randomly selects one of the filter circuits and output operators from the lists that were created in the previous code.
- It adds a batch normalization layer using the BatchNormalization layer from TensorFlow Keras. This layer normalizes the output of the filter to have zero mean and unit variance, which helps to stabilize the training process and reduce overfitting.
- It adds a ReLU activation layer using the ReLU layer from TensorFlow Keras. This layer applies a nonlinear function to the output of the filter, which introduces nonlinearity and sparsity to the model.
- It defines a function to create a classical dense layer, which is a layer that performs a linear transformation followed by a nonlinear activation on the classical data. The function does the following:
- It creates a sequential layer using TensorFlow Keras.
- It adds a dense layer using the Dense layer from TensorFlow Keras. This layer takes a number of units as input, which is the dimensionality of the output space. The dense layer performs a matrix multiplication between the input and a weight matrix, and adds a bias vector to produce the output.
- It adds a batch normalization layer using the BatchNormalization layer from TensorFlow Keras. This layer normalizes the output of the dense layer to have zero mean and unit variance.
- It adds a ReLU activation layer using the ReLU layer from TensorFlow Keras. This layer applies a nonlinear function to the output of the dense layer.
- It builds the quantum convolutional neural network model using the Sequential class from TensorFlow Keras. The model consists of several layers that are stacked together in order. The model does the following:
- It adds an amplitude encoding circuit to encode the images into quantum data using the AddCircuit layer from TensorFlow Quantum. This layer takes a function that returns a tensor of circuits as input, and applies it to each sample in the batch. The function uses amplitude encoding to encode each image into a 16-dimensional quantum state, as explained in the previous code.
- It adds two quantum convolutional layers using the `create_qcnn_layer` function. These layers apply two different quantum convolution filters to extract features from the quantum data.
- It adds a flatten layer to convert the quantum data into classical data using the Flatten layer from TensorFlow Keras. This layer flattens each sample into a 1D vector of length `n_filters`, which is 4 in this case.
- It adds two classical dense layers for classification using the `create_dense_layer` function. These layers perform linear transformations and nonlinear activations on the classical data to reduce its dimensionality and learn higher-level features.
- It adds a final dense layer with sigmoid activation for binary output using the Dense layer from TensorFlow Keras. This layer takes 1 as the number of units, which means that it outputs a scalar value between 0 and 1 for each sample. The sigmoid activation function squeezes the output into the range of 0 and 1, which can be interpreted as the probability of belonging to the positive class (segmented lesion) or the negative class (background).
- It compiles the model with binary crossentropy loss and Adam optimizer using the compile method from TensorFlow Keras. The binary crossentropy loss is a function that measures how well the model predicts the groundtruth labels, which are either 0 or 1. The Adam optimizer is an algorithm that updates the model parameters (weights and biases) based on the gradient of the loss function with respect to them. The metrics argument specifies what metrics to monitor during training and evaluation, such as accuracy, which is the fraction of correctly predicted samples.
- It prints the model summary using the summary method from TensorFlow Keras. This shows the structure and parameters of the model, such as the number of layers, the input and output shapes, and the number of trainable and non-trainable parameters.
- It trains the model on the images and groundtruths using the fit method from TensorFlow Keras. This method takes the quantum data tensors that were created in the previous code as inputs, and performs the following steps for each epoch:
- It shuffles the data and splits it into batches of size `batch_size`.
- It loops over each batch and feeds it to the model, which computes the output and the loss.
- It calculates the gradient of the loss with respect to the model parameters and updates them using the optimizer.
- It records the metrics such as accuracy for the batch and averages them over the epoch.
- It repeats the same steps for a validation set, which is a subset of the data that is not used for training but for evaluating the model performance. The validation set is obtained by splitting 20% of the data using the `validation_split` argument.


The on sixth cell code does the following steps:

- It evaluates the model on the test set using the predict method from TensorFlow Keras. This method takes the quantum data tensors that were created in the previous code as inputs, and outputs the predictions for each sample. The predictions are tensors of shape (`number_of_samples`, 1), where `number_of_samples` is the number of images and groundtruths in the test set, which is 200 in this case.
- It reshapes the predictions and groundtruths to have the same shape as the images, which is (`img_size`, `img_size`), where `img_size` is the image size, which is 256 in this case. This makes it easier to compare and visualize them.
- It plots some examples of images and their groundtruths and predictions using the `plot_image_and_groundtruth` and `plot_image_and_prediction` functions. These functions take an image and a groundtruth or a prediction as inputs, and create a figure with two subplots. The first subplot shows the image, and the second subplot shows the groundtruth or the prediction. The groundtruth and the prediction are binary images that indicate whether each pixel belongs to the segmented lesion (white) or the background (black).
- It plots the ROC curve and AUC score for the test set using the `plot_roc_curve` function. This function takes the groundtruths and predictions as inputs, and calculates and plots the ROC curve and AUC score. The ROC curve is a plot that shows how well the model can distinguish between the positive class (segmented lesion) and the negative class (background) at different thresholds. The AUC score is a scalar value that measures the area under the ROC curve, which ranges from 0 to 1. A higher AUC score means that the model has a better performance.
