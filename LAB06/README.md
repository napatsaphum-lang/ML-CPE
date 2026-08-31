# LAB06 - Neural Network and Its Application

## Overview

This laboratory focuses on applying a **Neural Network (NN)** to a classification problem.

For this experiment, I selected the **Iris Dataset** because it is a simple and well-known dataset that is suitable for learning the basic workflow of classification using Neural Networks.

ในการทดลองนี้ จุดประสงค์ไม่ได้มีเพียงการสร้าง Neural Network เพียง Model เดียว แต่ต้องการทดลองเปรียบเทียบหลาย Configuration โดยเปลี่ยนจำนวน Epochs, Hidden Layers และ Neurons เพื่อดูว่าปัจจัยเหล่านี้ส่งผลต่อประสิทธิภาพของโมเดลอย่างไร

The performance of each model is evaluated using testing accuracy, confusion matrix, training and validation accuracy/loss, and prediction results.

---

## Objectives

The objectives of this laboratory are:

1. To understand the basic working principle of a Neural Network.
2. To apply a Neural Network to a classification problem.
3. To prepare and preprocess a dataset before training.
4. To split the dataset into training, validation, and testing sets.
5. To standardize input features before model training.
6. To compare different Neural Network configurations.
7. To study the effect of different numbers of epochs.
8. To evaluate each model using accuracy.
9. To visualize training and validation accuracy/loss.
10. To use the best trained model for prediction.

วัตถุประสงค์หลักของการทดลองนี้คือเพื่อให้เข้าใจขั้นตอนการทำงานของ Neural Network ตั้งแต่การเตรียมข้อมูล การ Train Model ไปจนถึงการวิเคราะห์ผลลัพธ์ที่ได้จากแต่ละ Configuration

---

## Dataset

The dataset used in this laboratory is the **Iris Species Dataset**.

The dataset contains information about Iris flowers and is commonly used for classification problems.

The input features are:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The target column is:

```text
Species

The dataset contains three classes:

Iris-setosa
Iris-versicolor
Iris-virginica

ไฟล์ Dataset ที่ใช้ในการทดลองคือ

Iris.csv
Dataset Source

The dataset was obtained from Kaggle:

Iris Species Dataset
https://www.kaggle.com/datasets/uciml/iris

Workflow

The overall workflow of this laboratory is shown below.

Iris Dataset
      |
      v
Load Dataset
      |
      v
Data Preprocessing
      |
      v
Label Encoding
      |
      v
Split Dataset
      |
      v
Training / Validation / Testing
      |
      v
Feature Standardization
      |
      v
Build Neural Network
      |
      v
Train Different Models
      |
      v
Evaluate Accuracy
      |
      v
Compare Model Performance
      |
      v
Select Best Model
      |
      v
Prediction and Visualization

ผมแยกขั้นตอนการทำงานออกเป็นหลายไฟล์ เพื่อให้แต่ละส่วนมีหน้าที่ชัดเจน เช่น การโหลดข้อมูล การ Preprocessing การสร้าง Model และการ Evaluate ผลลัพธ์ ทำให้โค้ดอ่านง่ายและแก้ไขได้สะดวกมากขึ้น

Data Preprocessing

Before training the Neural Network, the dataset needs to be prepared.

The preprocessing steps include:

Load the dataset from Iris.csv.
Remove the Id column because it is only an identifier and is not useful for classification.
Select the four input features.
Use Species as the target variable.
Convert the species names into numerical labels using LabelEncoder.

The selected features are:

SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm

The target is:

Species

ขั้นตอนนี้มีความสำคัญเพราะข้อมูลที่ใช้กับ Neural Network ควรอยู่ในรูปแบบตัวเลขและมีเฉพาะ Features ที่เกี่ยวข้องกับการจำแนกเท่านั้น

Dataset Splitting

The dataset is divided into three parts:

Dataset	Ratio
Training Set	70%
Validation Set	10%
Testing Set	20%

The training set is used to train the model.

The validation set is used to observe model performance during training.

The testing set is used to evaluate the final model after training.

The dataset splitting process also uses:

random_state = 42

and

stratify

to keep the class distribution balanced between each dataset.

การแบ่งข้อมูลแบบนี้ช่วยให้สามารถประเมินประสิทธิภาพของโมเดลกับข้อมูลที่ไม่ได้ใช้ในการ Train ได้จริง และช่วยลดความเสี่ยงในการสรุปผลจาก Training Data เพียงอย่างเดียว

Feature Standardization

Before training the Neural Network, the input features are standardized using:

StandardScaler()

The scaler is fitted only on the training data.

After that, the same scaler is used to transform the training, validation, and testing datasets.

This helps keep the input features on a similar scale and makes the training process more stable.

ในส่วนนี้จะ Fit StandardScaler เฉพาะ Training Data ก่อน แล้วจึงนำค่าเดียวกันไป Transform กับ Validation และ Testing Data เพื่อป้องกันไม่ให้ข้อมูลจากชุดทดสอบถูกนำมาใช้ก่อนการประเมินผล

Neural Network Model

The Neural Network is created using TensorFlow / Keras with a Sequential model.

The hidden layers use:

ReLU

as the activation function.

The output layer uses:

Softmax

because this problem contains three output classes.

The model is compiled using:

Optimizer : Adam
Loss      : Sparse Categorical Crossentropy
Metric    : Accuracy

โครงสร้างนี้เหมาะกับโจทย์ Multi-class Classification เนื่องจากต้องจำแนกข้อมูลออกเป็น Iris จำนวน 3 Classes

Neural Network Configurations

In this laboratory, four different Neural Network configurations are tested.

Model	Hidden Layers	Epochs
Model 1	[8]	20
Model 2	[8]	50
Model 3	[16, 8]	50
Model 4	[32, 16, 8]	50

The purpose of using different configurations is to compare how the number of epochs, hidden layers, and neurons affect model performance.

Model 1 vs Model 2

Model 1 and Model 2 use the same architecture with one hidden layer containing 8 neurons.

The difference is the number of epochs:

Model 1 = 20 epochs
Model 2 = 50 epochs

This comparison is used to observe the effect of increasing the number of training epochs.

Model 2 vs Model 3

Both models use 50 epochs, but Model 3 has more hidden layers and neurons.

Model 2 = [8]
Model 3 = [16, 8]

This helps show whether a more complex network can improve classification performance.

Model 3 vs Model 4

Model 4 increases the network size again.

Model 3 = [16, 8]
Model 4 = [32, 16, 8]

จากการออกแบบการทดลองนี้ จะสามารถเปรียบเทียบได้ทั้งผลของจำนวน Epochs และความซับซ้อนของโครงสร้าง Neural Network ซึ่งช่วยให้เห็นว่า Model ที่มีจำนวน Layer หรือ Neurons มากกว่า ไม่จำเป็นต้องให้ผลดีที่สุดเสมอไป

Model Evaluation

After training each Neural Network configuration, the model is evaluated using the testing dataset.

The main evaluation metric is:

Accuracy

The result format is similar to:

Model 1 | Layers: [8]         | Epochs: 20 | Accuracy: ...
Model 2 | Layers: [8]         | Epochs: 50 | Accuracy: ...
Model 3 | Layers: [16, 8]     | Epochs: 50 | Accuracy: ...
Model 4 | Layers: [32, 16, 8] | Epochs: 50 | Accuracy: ...

The model with the highest testing accuracy is selected as the Best Model.

ค่าของ Accuracy จะมาจากการรันโปรแกรมจริง ดังนั้นค่าอาจแตกต่างกันเล็กน้อยในแต่ละครั้งจากกระบวนการ Train ของ Neural Network

Results and Graphs
Model Accuracy Comparison

The following graph compares the testing accuracy of all Neural Network configurations.

กราฟนี้ใช้ดูว่า Configuration ใดให้ Accuracy สูงที่สุด และช่วยให้เปรียบเทียบได้ง่ายว่าการเพิ่มจำนวน Epochs หรือเพิ่มขนาดของ Network ส่งผลต่อผลลัพธ์มากน้อยเพียงใด

Confusion Matrix

The confusion matrix of the best-performing model is shown below.

The confusion matrix provides more information than accuracy alone because it shows how many samples in each class are correctly or incorrectly classified.

กราฟนี้ช่วยให้เห็นว่า Model มีปัญหาในการแยก Class ใดออกจากกันหรือไม่ ไม่ได้ดูเฉพาะค่า Accuracy รวมเพียงอย่างเดียว

Training and Validation Accuracy

The graph below shows the training accuracy and validation accuracy of the best model across each epoch.

This graph is useful for observing how the Neural Network learns during training.

If both training and validation accuracy improve together, it usually means the model is learning useful patterns.

If training accuracy continues to increase while validation accuracy stops improving, it may indicate that the model is beginning to overfit.

จึงสามารถใช้กราฟนี้ดูแนวโน้มของโมเดลระหว่างการ Train และดูความแตกต่างระหว่าง Training Data กับ Validation Data ได้

Training and Validation Loss

The following graph shows the training loss and validation loss across each epoch.

Normally, training loss should decrease during training.

Validation loss can also be used together with training loss to check whether the model is still generalizing well to unseen data.

ถ้า Training Loss ลดลงต่อเนื่อง แต่ Validation Loss เริ่มสูงขึ้น อาจเป็นสัญญาณว่า Model เริ่มเกิด Overfitting ได้

Prediction

After selecting the best model, the model is saved as:

best_model.keras

The saved model can then be loaded using test_nn.py to perform predictions on the testing dataset.

The output format is similar to:

Sample 1
Actual     : Iris-setosa
Predicted  : Iris-setosa
Confidence : xx.xx %

Sample 2
Actual     : Iris-versicolor
Predicted  : Iris-versicolor
Confidence : xx.xx %

For each sample, the program displays:

Actual class
Predicted class
Prediction confidence

ส่วนนี้ทำให้สามารถดูผลการ Prediction เป็นรายตัว และเปรียบเทียบค่าที่โมเดลทำนายกับ Class จริงของข้อมูลได้โดยตรง

Project Structure
LAB06/
│
├── Iris.csv
├── data_loader.py
├── preprocessing.py
├── split_data.py
├── nn_model.py
├── evaluate.py
├── main.py
├── test_nn.py
├── README.md
│
└── outputs/
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── training_accuracy.png
    ├── training_loss.png
    └── best_model.keras

The purpose of each file is shown below.

File	Description
Iris.csv	Dataset used in this laboratory
data_loader.py	Loads the dataset
preprocessing.py	Prepares features, labels, and standardization
split_data.py	Splits the dataset into train, validation, and test sets
nn_model.py	Builds, trains, and predicts using the Neural Network
evaluate.py	Evaluates accuracy and creates result graphs
main.py	Main program for running the experiment
test_nn.py	Loads the best model and performs prediction
outputs/	Stores generated graphs and the trained model
How to Run

Install the required Python libraries:

pip install tensorflow pandas numpy scikit-learn matplotlib

Run the main experiment:

python main.py

After training is completed, the generated results will be stored in:

outputs/

To test the saved best model:

python test_nn.py
Discussion

From this experiment, I can compare the effect of different Neural Network configurations instead of using only one model.

The first comparison focuses on the number of epochs by using the same network architecture with 20 and 50 epochs.

The next comparisons increase the number of hidden layers and neurons while keeping the number of epochs at 50.

จากการทดลองในลักษณะนี้ ทำให้สามารถดูผลของทั้งจำนวน Epochs และโครงสร้างของ Model ได้ในเวลาเดียวกัน และทำให้เห็นว่าการเพิ่มความซับซ้อนของ Neural Network ไม่ได้หมายความว่า Accuracy จะสูงขึ้นเสมอ

The results should be considered together with the training and validation graphs because accuracy alone does not fully describe how the model behaves during training.

ดังนั้นในการวิเคราะห์ผลจึงควรดู Accuracy, Confusion Matrix, Training/Validation Accuracy และ Training/Validation Loss ร่วมกัน เพื่อให้เห็นภาพรวมของ Model ได้ชัดเจนมากขึ้น

Conclusion

In this laboratory, I applied a Neural Network to classify the Iris Dataset.

The experiment covered the complete basic workflow, including data loading, preprocessing, dataset splitting, feature standardization, Neural Network construction, model training, evaluation, visualization, and prediction.

Four different Neural Network configurations were compared by changing the number of epochs, hidden layers, and neurons.

The results were evaluated using testing accuracy, confusion matrix, training and validation accuracy, and training and validation loss.

Finally, the model with the highest testing accuracy was selected as the best model and saved for further prediction.

จาก LAB นี้ทำให้เข้าใจกระบวนการทำ Neural Network มากขึ้น ตั้งแต่การเตรียม Dataset ไปจนถึงการนำ Model ที่ Train เสร็จแล้วมาใช้งานจริง และยังทำให้เห็นว่าการเลือก Architecture ที่เหมาะสมควรพิจารณาจากผลลัพธ์หลายด้าน ไม่ใช่เลือกจาก Model ที่มีโครงสร้างใหญ่ที่สุดเพียงอย่างเดียว

References
UCI Machine Learning. Iris Species Dataset. Kaggle.
https://www.kaggle.com/datasets/uciml/iris
TensorFlow. Keras Sequential Model Documentation.
https://www.tensorflow.org/
Scikit-learn. Machine Learning Tools and Preprocessing Documentation.
https://scikit-learn.org/
