# Handwritten-digit-recognizer-GUI
This program is able to detect the digit(0-9) that anyone draws in the canvas.

## Training and Testing Data
* total rows : 42,000
* total columns : 784(features) + 1(label)

* total training data: 13,860(33% percent of the total rows)->Could'nt take more due to my system limitations
* test data: 28,140(67% of total rows)
## Model
The model I used here to train with data is a SVM(support vector machine) with 'rbf' kernel, provided in the sklearn library.
## GUI
The Gui is made using tkinter module of python 
## Image processing
Done using Pillow and OpenCV

## Accuacy (varies)
* Accuracy on training data-97.88%
* Accuracy on testing data-96.17%

* Accuracy on real-time usage ~78%

## Screenshots:

![pic1](https://user-images.githubusercontent.com/53531220/104904664-6b53da00-59a7-11eb-8489-1e8b5baaf6b0.JPG)

![pic2](https://user-images.githubusercontent.com/53531220/104904673-6ee76100-59a7-11eb-907c-961669a8ab5f.JPG)

![pic3](https://user-images.githubusercontent.com/53531220/104904681-71e25180-59a7-11eb-9731-7e89145512c0.JPG)
