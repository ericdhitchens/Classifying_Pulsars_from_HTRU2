# Classifying Pulsars from the High Time Resolution Survey, South (HTRU2)

## Overview
The purpose of this project is to analyze telescope data to correctly separate actual pulsar data from radio frequency noise. The dataset poses a binary classification problem, which is analyzed with the following models:

* Multiple Logistic Regression
* Decision Tree Classification
* Random Forest Classification
* Support Vector Machine (SVM) Classification
* Deep Neural Network (DNN) Classification


## Dataset Citation
The dataset was retrieved from the UC Irvine Machine Learning Repository at the following link: https://archive.ics.uci.edu/ml/datasets/HTRU2#

The data comes from the High Time Resolution Universe Survey (South), known as HTRU2. The dataset was donated to the UCI Repository by Dr. Robert Lyon of The University of Manchester, United Kingdom. The two papers requested for citation in the description are listed below:

* R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
* R. J. Lyon, HTRU2, DOI: 10.6084/m9.figshare.3080389.v1.


## Repository File Structure
### 01_Data
Contains all data files in .xls or .xlsx and .csv format.
1. **Original** - The original data retrieved from the UCI dataset.
2. **Transformed** - The files with my own reformatting.
3. **Loaded** - The final data used in the Python Jupyter Notebooks.


### 02_Python_Code
Contains the code that was written in Jupyter Notebooks. Python files are also included for reference. PDF reports generated via LaTeX from the Jupyter notebooks are presented in the 03_PDF_Reports folder.
1. **Exploratory_Data_Analysis**
	* READ THIS JUPYTER NOTEBOOK FIRST.
	* Provides dataset background, features, statistics, and exploratory visualizations.
2. **Modeling**
	* Contains all of the models tested in this project:
		* Multiple Logistic Regression
		* Decision Tree Classification
		* Random Forest Classification
		* Support Vector Machine (SVM) Classification
		* Deep Neural Network (DNN) Classification
	* Each model may be read indepenedently, but they should be read after the exporatory data analysis notebook.
	* The performance of each model is compared in the 03_Analysis folder
3. **Analysis**
	* READ THIS JUPYTER NOTEBOOK LAST.
	* Compares the performance of all models and describes the winning model.


### 03_PDF_Reports
Contains PDFs of the Jupyter notebooks from the 02_Python_Code folder. The PDFs were generated via LaTeX from the Jupyter Notebook platform. They are intended to be read in order (01 through 06). However, PDFs 02-05 describing each model may be understood separately.