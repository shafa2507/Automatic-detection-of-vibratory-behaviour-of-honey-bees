# Automatic Detection of Vibratory Behavious of Honeybees.

### Installation
Please download our toolkit ```vhbdetector``` from <a href = "https://github.com/shafa2507/Automatic-detection-of-vibratory-behaviour-of-honey-bees/releases/tag/Latest_Version">here</a>.

Following packages are required to be installed on system before using this toolkit(or package).
<ul>
    <li>
        TensorFlow
    </li>
    <li>
        Pickle
    </li>
    <li>
        CV2
    </li>
    <li>
        Sci-Kit Learn (sklearn)
    </li>
</ul>

Use ```pip install package_name``` to install the above packages.

### Getting Started
Please run the ```setup.py``` before to use any file.

Please  see the ```model_training_and_evaluation.py``` file to get help understand the modules and functionality of this library or tool.

### Data Loading
Please use the following snippets for data loading

Use this to load the class and initialize the object of the class

```python
from vhbdetector.data_loader import DataLoader

dl = DataLoader()
```

##### For data loading from zipped folders of images(scratch):

```python
data, labels = dl.load_data_from_scratch(set_frame_len, image_height, image_width, data_file, labels_file)
```

Here, `data_file` refers to the address of `wdd_ground_truth` folder while `labels_file` refers to the address of `ground_truth_wdd_angles.pickle` file. Whereas, `set_frame_len` means how many frames to keep while loading the data, similarly, height and width.

##### For data loading from Pickle files if it is already extracted and managed in NumPy Arrays:

```python
data, labels = dl.load_data(data_file, labels_file)
```

Here, ```data_file``` refers to the file that contains all the data of videos stored in Pickle files and managed in NumPy Arrays similarly for labels file.

##### Save the data (videos) extracted from zipped images to Pickle files:

```python
dl.save_data(videos_data = np.array([]) , videos_label = np.array([]), data_file = "X.pickle", target_class_file = "Y.pickle")
```

`videos_data` means the array whose data will be stored and likewise `videos_label` that contains labels data in NumPy file to store in `target_class_file`.

### Data Pre-processing
Load the necessary <b>modules</b>.

```python
from vhbdetector.pre_processing import Preprocessing

pp = Preprocessing()
```

##### change the dimensions of the data or edit the data
The number of frames, imamge height and image width can be modified using this utility. It returns the updated dataset after data wrangling. The following code will help perform the task:

```python
data = pp.change_data_dimensions(data, new_frame_len, new_img_height, new_img_width)
```

### Data Normalization
Load the necessary <b>modules</b>.

```python
from vhbdetector.pre_processing import Normalization
norm = Normalization()
```

##### Pixels Normalization
It normalizes the data of every frame (image) with pixel normalization method (i.e. data / 255).

```python
data = norm.pixels_normalization(data)
```

### Data Augmentation
Load the necessary <b>modules</b>.

```python
from vhbdetector.pre_processing import Augmentation

aug = Augmentation()
```

##### Balance the dataset using data augmentation by generating roated (transformed) videos of lower classes whose distribution is small to balance the dataset. 

```python
new_data, new_labels = aug.balance_data(data, labels)
```

### How to train a model using this <b>tool/package</b>?
Load the specific <b>model</b>.

```python
from models.cnn.convolutions import CNN_TD                                    # to use Convolution 3D model
from vhbdetector.models.cnn_rnn.conv_lstm import CONVLSTM                     # to use standard CNN-RNN model with diverse hyperparameters
from models.cnn_rnn.conv_lstm_basic import BASIC_CONVLSTM                     # to use Basic CNN-RNN model that runs faster with mininal range of hyperparameters
from models.cnn_rnn.convlstm_regularization_norms import CONVLSTM_REGR        # to use CNN-RNN deep network with regularization effect features and additional hyperparameters
```
<b>Note:</b> There is a greater flexibility ensured in choosing hyperparameters from the following allowed parameters. Please choose the hyperparameters from the corresponding models accordingly.

#### For Convolutions 3D model, following is the way to initialize the model
```python
model = CNN_TD(filters = 8, convolutions_activation = "tanh", drop_rate = 0.1)
```
#### For CNN-RNN model, following is the way to initialize the model
```python
model = CONVLSTM(convolutions_activation = "tanh", recurrent_activation = "hard_sigmoid", time_distributed_drop_rate = 0.2, convolutions_drop_rate = 0.1, recurrent_drop_rate = 0.1)
```
#### For CNN-RNN Basic model, following is the way to initialize the model
```python
model = BASIC_CONVLSTM(convolutions_activation = "tanh", recurrent_activation = "hard_sigmoid", time_distributed_drop_rate = 0.2)
```
#### For CNN-RNN regularized model, following is the way to initialize the model
```python
model = CONVLSTM_REGR(convolutions_activation = "tanh", recurrent_activation = "hard_sigmoid", time_distributed_drop_rate = 0.2, convolutions_drop_rate = 0.1, recurrent_drop_rate = 0.1)
```

<b>Alternatively</b>, Hyperparameters can be chosen using ```.set_hyperparams()``` function.

### Create Model
The following code provides the usage guide to create model.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.3, stratify = labels, shuffle = True, random_state = 5)

model.create_model(X_train, Y_train, X_test, Y_test)
```

### Training a model
```python
history, trained_model = model.train_model()
```

### Measuring Accuracy

```python
accuracy = model.get_accuracy_score(X_test, Y_test)
```

### Making predictions

```python
video_file = r"C:\Users\Muhammad Kaleemullah\.spyder-py3\Software Project\Automatic Detection of Vibratory Honeybees\vhbdetector\datasets\sample_video.mp4"
video_pred = model.predict_video(video_file)
```

<b>Note:</b> Please let us know if you found any issue in this tool.
Thanks for using it!
