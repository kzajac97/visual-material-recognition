# visual-material-recognition

# Data

Data can be accessed in OneDrive of Wrocław University of Science and technology, under [this link](https://politechnikawroclawska-my.sharepoint.com/:f:/g/personal/235698_student_pwr_edu_pl/EpetyluRy5pOmW9tRP5bdJsBWwztJOqKkFFIH5W0Vk6scw?e=1gMK12).
Data directory has the following structure:
* `images`
  * `processed`
  * `raw`
* `plots`
* `dataset.csv`
* `measurements.csv`

### images

Directory containing images of the microstructures, the process of obtaining them is given in the paper. It consists
of two sub-directories, one with raw images as taken by the microscope and second one with images split into 4 parts
and prepared for using with statistical models. The process of obtaining the dataset is described in the paper and
can be replicated using notebook `eda.ipynb`.

### measurements

CSV file containing measured values of hardness for each sample, it consists of five columns:
* `Sample` - label given to each measurement (4 measurements per image)
* `Hardness` - hardness measured in HV1 units
* `File` - corresponding image file in directory with raw images
* `Location` - location of the measurement, can be one of "top-left", "top-right", "bottom-right", "bottom-left"
* `DELETED` - True/False indicator if given sample should be used in experiments

### dataset

CSV file with dataset prepared for training, contains following columns:
* `Sample` - label given to each measurement (4 measurements per image)
* `Hardness` - hardness measured in HV1 units
* `File` - corresponding image file in processed directory, each row corresponds to unique image (which is a slice of the raw image)
* `Location` - location of the measurement, can be one of "top-left", "top-right", "bottom-right", "bottom-left"
* `Code` - raw image code, each row corresponds to the material sample (4 measurement per sample, with some rejected)
* `Steel Name` - name of the steel, used for plotting and statistics

# Experiments

Experiments are provided in jupyter notebooks, directory notebooks contains following files:
* `eda.ipynb` - notebook constructing dataset and statistics used in the paper
* `experiments.ipynb` - experiments with otsu-based index, fractal-dimension index and 2-point correlation
* `vit.ipynb` - experiements using the vision transformer
* `ablation.ipynb` - small ablation study of the fractal-dimension index

*Note*: The notebooks provided contain only results published in the paper, for more details please contact the authors.
        Other conducted experiments include tuning of all methods (for example by using Sobel filter instead of Canny detector or changing the number of PCA components in 2-point correlation method)
        , using combined otsu and fractal dimension index, using different pre-trained models than ViT and some more.
        

# Code

Utility code used for modelling and computation of metrics is provided in `src` directory. Experiments notebooks import
functions from this package. All requirements are given in `requirements.txt` file, then can be easily installed using:

```bash
pip install -r requirements.txt
```

Experiment related to ViT model containing different set-up, it is not advised to run this on machine without GPU, since
even inference can take significant amount of time. Requirements for this experiment are given in `requirements-vit.txt` file,
which are added on top of original requirements. 
