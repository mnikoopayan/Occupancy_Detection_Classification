
# Occupancy Detection Classification Project

## Overview

This project implements and compares three classification algorithms—Random Forest, Support Vector Machine (SVM), and Long Short-Term Memory (LSTM)—to predict room occupancy based on environmental sensor data such as temperature, humidity, light, CO₂, and humidity ratio. The performance of these models is evaluated using 10-fold cross-validation, and various performance metrics are calculated manually.

**Author:** Mohammad Saleh Nikoopayan Tak

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [Project Structure](#project-structure)
3. [Installation and Setup](#installation-and-setup)
4. [Usage Instructions](#usage-instructions)
5. [Results](#results)
6. [Discussion](#discussion)
7. [Conclusion](#conclusion)
8. [References](#references)
9. [Acknowledgments](#acknowledgments)
10. [Contact Information](#contact-information)
11. [Additional Notes](#additional-notes)

## Dataset Description

**Dataset Name:** Occupancy Detection Data Set  
**Source:** UCI Machine Learning Repository  
**URL:** Occupancy Detection Data Set  
**Description:** The dataset contains experimental data used for binary classification (room occupancy) based on temperature, humidity, light, CO₂, and humidity ratio measurements. Ground-truth occupancy was obtained from time-stamped pictures taken every minute.

## Project Structure

- **FinalDataset.txt:** The dataset used for the project.
- **NikoopayanTak_MohammadSaleh_finaltermproj.ipynb:** The Jupyter Notebook containing the complete code for data preprocessing, model implementation, and evaluation.
- **NikoopayanTak_MohammadSaleh_finaltermproj.pdf:** The project report in PDF format.
- **NikoopayanTak_MohammadSaleh_finaltermproj.html:** The project report in HTML format.
- **README.md:** Project documentation and instructions (this file).

## Installation and Setup

### Prerequisites

- Python 3.x  
- Jupyter Notebook  

### Required Packages

- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Keras  
- TensorFlow  

### Installation Instructions

#### Clone the Repository

```bash
git clone https://github.com/mnikoopayan/Occupancy_Detection_Classification.git
```

#### Navigate to the Project Directory

```bash
cd Occupancy_Detection_Classification
```

#### Install the Required Packages

```bash
pip install numpy pandas matplotlib seaborn scikit-learn keras tensorflow
```


#### Ensure the Dataset is in Place

The dataset file `FinalDataset.txt` should be located in the project directory.

## Usage Instructions

1. **Open the Jupyter Notebook**  
   ```bash
   jupyter notebook NikoopayanTak_MohammadSaleh_finaltermproj.ipynb
   ```

2. **Run the Notebook**  
   Execute all cells sequentially by selecting `Cell > Run All` in the Jupyter interface.  
   The notebook includes code for:
   - Importing libraries  
   - Loading and preprocessing the data  
   - Exploratory Data Analysis (EDA)  
   - Implementing 10-fold cross-validation  
   - Defining performance metrics  
   - Training and evaluating Random Forest, SVM, and LSTM models  
   - Visualizing results and ROC curves  
   - Discussing findings and conclusions  

3. **View the Results**  
   The outputs, including performance metrics and plots, will be displayed within the notebook.  
   Detailed explanations are provided alongside code cells for better understanding.

## Results

### Average Performance Metrics Across 10 Folds

| Metric               | Random Forest | SVM     | LSTM    |
|-----------------------|---------------|---------|---------|
| True Positive (TP)    | 469.8000      | 473.1000| 472.5000|
| True Negative (TN)    | 1572.3000     | 1560.2000|1560.2000|
| False Positive (FP)   | 8.7000        | 20.8000 | 20.8000 |
| False Negative (FN)   | 5.2000        | 1.9000  | 2.5000  |
| True Positive Rate (TPR)| 0.9891      | 0.9960  | 0.9947  |
| True Negative Rate (TNR)| 0.9945      | 0.9868  | 0.9868  |
| Precision             | 0.9818        | 0.9579  | 0.9578  |
| F1 Score              | 0.9854        | 0.9765  | 0.9759  |
| Accuracy              | 0.9932        | 0.9890  | 0.9887  |
| AUC                   | 0.9989        | 0.9944  | 0.9967  |
| Brier Score           | 0.0052        | 0.0107  | 0.0102  |

### Key Findings

- Random Forest achieved the highest accuracy (99.32%) and AUC (99.89%).
- SVM and LSTM also performed exceptionally well but slightly underperformed compared to Random Forest.
- The models effectively utilized environmental sensor data for occupancy prediction.


## Discussion

- **Random Forest** is recommended for deployment due to its superior performance and computational efficiency.  
- **Effectiveness of Features:** Environmental sensor data, especially light and CO₂ levels, are strong indicators of room occupancy.  
- **Class Imbalance:** The dataset has a class imbalance (76.90% unoccupied). While the models handled it well, future work could explore techniques like SMOTE or adjusting class weights.  
- **Model Complexity vs. Performance:** Random Forest offers a good balance between complexity and performance, making it suitable for real-time applications.

## Conclusion

The project demonstrates the feasibility of implementing predictive models for room occupancy detection using environmental sensor data. Random Forest provided the best balance of accuracy and efficiency, making it suitable for practical applications in smart buildings and energy management systems.

## References

1. **Dataset Source:**  
   - UCI Machine Learning Repository - Occupancy Detection Data Set  

2. **Documentation:**  
   - [Scikit-learn](https://scikit-learn.org/stable/)  
   - [Keras](https://keras.io/)  
   - [TensorFlow](https://www.tensorflow.org/)

## Acknowledgments

- **Dataset Contributors:** Luis M. Candanedo and Véronique Feldheim at the University of Mons, Belgium.  
- **Academic Guidance:** (Include any professors or mentors if applicable).  



## Contact Information

**Author:** Mohammad Saleh Nikoopayan Tak  
**GitHub:** mnikoopayan  
**Email:** mn552@njit.com  

Feel free to reach out if you have any questions or suggestions regarding this project.

## Additional Notes

- **Reproducibility:** Ensure all the required packages are installed and the dataset is correctly placed in the project directory for seamless execution.  
- **Updates:** Future updates may include hyperparameter tuning, addressing class imbalance, and incorporating temporal features for enhanced model performance.  
- **Contributions:** Contributions are welcome. Please fork the repository and submit a pull request for any enhancements or bug fixes.  

Thank you for exploring the Occupancy Detection Classification Project!
