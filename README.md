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

Use ``` pip install package_name ``` to install the above packages.

### Getting Started
Please run the ```setup.py``` before to use any file.

Please  see the ```model_training_and_evaluation.py``` file to get help understand the modules and functionality of this library or tool.

### Data Loading
Please use the following snippets for data loading

Use this to load the class and initialize the object of the class

```from vhbdetector.data_loader import DataLoader```

```dl = DataLoader()```

##### For data loading from zipped folders of images(scratch):

```data, labels = dl.load_data_from_scratch(set_frame_len, image_height, image_width, data_file, labels_file)```

Here, `data_file` refers to the address of `wdd_ground_truth` folder while `labels_file` refers to the address of `ground_truth_wdd_angles.pickle` file. Whereas, `set_frame_len` means how many frames to keep while loading the data, similarly, height and width.

##### For data loading from Pickle files if it is already extracted and managed in NumPy Arrays:

```data, labels = dl.load_data(data_file, labels_file)```

Here, ```data_file``` refers to the file that contains all the data of videos stored in Pickle files and managed in NumPy Arrays similarly for labels file.

##### Save the data (videos) extracted from zipped images to Pickle files:

```dl.save_data(videos_data = np.array([]) , videos_label = np.array([]), data_file = "X.pickle", target_class_file = "Y.pickle")```

`videos_data` means the array whose data will be stored and likewise `videos_label` that contains labels data in NumPy file to store in `target_class_file`.

### Data Pre-processing
Load the necessary <b>modules</b>.

```from vhbdetector.pre_processing import Preprocessing```

```pp = Preprocessing()```

##### change the dimensions of the data or edit the data
The number of frames, imamge height and image width can be modified using this utility. It returns the updated dataset after data wrangling. The following code will help perform the task:

```data = pp.change_data_dimensions(data, new_frame_len, new_img_height, new_img_width)```

### Data Normalization
Load the necessary <b>modules</b>.

```from vhbdetector.pre_processing import Normalization```

```norm = Normalization()```

##### Pixels Normalization
It normalizes the data of every frame (image) with pixel normalization method (i.e. data / 255).

```data = norm.pixels_normalization(data)```

### Data Augmentation
Load the necessary <b>modules</b>.

```from vhbdetector.pre_processing import Augmentation```

```aug = Augmentation()```

### Balance the dataset using data augmentation by generating roated (transformed) videos of lower classes whose distribution is small to balance the dataset. 

```new_data, new_labels = aug.balance_data(data, labels)```
