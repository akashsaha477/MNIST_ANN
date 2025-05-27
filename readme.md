# MNIST_ANN: Custom Artificial Neural Network for Handwritten Digit Classification

## Project Overview

This repository presents a detailed implementation of a two-layer Artificial Neural Network (ANN) built entirely from scratch using NumPy. The model is designed to classify handwritten digits from the widely recognized MNIST dataset, which consists of grayscale images of size 28x28 pixels representing digits 0 through 9.

Unlike conventional implementations that rely on high-level deep learning libraries, this project emphasizes the foundational mechanics of neural networks — including parameter initialization, forward propagation, backpropagation, activation functions, and parameter updates via gradient descent — to provide an educational and transparent insight into how ANNs operate under the hood.

## Dataset

- The MNIST dataset contains 70,000 labeled images of handwritten digits.
- This project uses CSV files (`mnist_train.csv` and `mnist_test.csv`) containing pixel values flattened into 784-dimensional vectors.
- Data preprocessing includes normalization of pixel intensities to the [0,1] range to facilitate efficient training convergence.

## Neural Network Architecture

- **Input Layer:** 784 neurons, each corresponding to a pixel value of the input image.
- **Hidden Layer:** 10 neurons with ReLU (Rectified Linear Unit) activation, enabling non-linear transformations and better learning capacity.
- **Output Layer:** 10 neurons with Softmax activation to generate probability distributions over digit classes (0-9).

## Key Features and Techniques

- **He Initialization:** Used for weight initialization to maintain variance and improve gradient flow during training.
- **Activation Functions:** ReLU for hidden layers and Softmax for output classification probabilities.
- **Loss Function:** Cross-entropy loss for multi-class classification accuracy.
- **Backpropagation:** Manual implementation of gradient calculations for both layers, ensuring understanding of weight updates.
- **Gradient Descent Optimization:** Parameter updates executed with a configurable learning rate.
- **One-Hot Encoding:** Labels are converted to one-hot vectors to compute losses correctly.

## Project Structure

MNIST_ANN/
│
├── ann_mnist.ipynb # Jupyter notebook with full model code, training loop, and evaluation
├── mnist_train.csv # Training dataset CSV file with labeled digit images
├── mnist_test.csv # Test dataset CSV file for model validation
├── .gitattributes # Git LFS config for managing large dataset files
├── README.md # This descriptive project documentation
└── saved_models/ # Directory for saving trained model parameters (optional)


MNIST_
## Installation and Requirements

Ensure you have Python 3.x installed along with the following packages:
pip install numpy pandas matplotlib


Jupyter Notebook or Jupyter Lab for interactive experimentation:
pip install notebook

1.Clone the Repository:
git clone https://github.com/akashsaha477/MNIST_ANN.git
cd MNIST_ANN

2.Open the Jupyter Notebook:
jupyter notebook ann_mnist.ipynb


3.Execute Notebook Cells

Load and preprocess data
Initialize model parameters
Run training loop with forward and backward propagation
Monitor loss and accuracy metrics
Test predictions with visualization of handwritten digits

4.Modify Parameters

Adjust hyperparameters such as learning rate, batch size, or number of iterations directly in the notebook to experiment with model performance.


Model Performance
Achieves approximately 89% accuracy on the MNIST test dataset.

Demonstrates foundational neural network principles without abstraction.

Provides a basis for extending to deeper networks or convolutional neural networks (CNNs).

Contributing
Contributions are highly encouraged! Feel free to fork this repository, improve the codebase, add documentation, or extend functionality.

License
This project is licensed under the MIT License. See the LICENSE file for more information.

Acknowledgments
Thanks to Samson Zhang for making such a nice guide. https://www.youtube.com/watch?v=w8yWXqWQYmU&t=679s

Thanks to Yann LeCun and collaborators for creating the MNIST dataset.

Inspired by educational resources on neural networks and deep learning fundamentals.

Utilizes NumPy and Matplotlib for numerical computing and visualization.






