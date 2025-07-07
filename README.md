# Comparative Study of Text Classification Models in Science Education

---

## Description

This repository presents a **comparative study** of text classification models applied to theses and dissertations in the field of science education. The goal is to **automate the labeling** of these scientific publications, which is currently a manual, time-consuming, and error-prone process.

The study investigates and contrasts the performance of:

- Classical Machine Learning models:  
  - Support Vector Machine (SVM)  
  - Multilayer Perceptron (MLP)  

- Advanced language model:  
  - Large Language Model (LLM) LLaMA 3  
  employing the **Few-Shot Learning** approach.

---

## Methodology

- Extensive **data preprocessing** to clean and normalize textual data  
- **Text vectorization** using TF-IDF to capture term importance  
- Training and rigorous evaluation of models with established metrics:  
  - Precision  
  - Recall  
  - F1-Score  
- In-depth analysis of **confusion matrices** to identify class-specific performance and common misclassifications

---

## Results

The experimental results revealed several key insights:

- Traditional models (SVM and MLP) were effective in classifying most categories, achieving solid overall metrics. However, they struggled notably with the **"General Sciences"** category due to significant vocabulary overlap with other classes, which led to higher misclassification rates.

- The LLaMA 3 model, even when using a **quantized and resource-efficient version**, exhibited superior generalization capabilities. It outperformed classical models across nearly all metrics, demonstrating robustness in handling nuanced and overlapping scientific content.

- These findings strongly indicate that **Large Language Models hold considerable promise for advancing automated text classification tasks in Portuguese**, especially in specialized domains like science education. Moreover, the ability to deploy quantized LLMs effectively broadens accessibility for institutions with limited computational resources.

---

## Conclusion and Future Perspectives

This work serves as a compelling **proof of concept** that automatic classification of theses and dissertations is feasible and can significantly reduce manual effort, increasing both speed and consistency.

The demonstrated advantages of LLMs suggest that further investment in fine-tuning, prompt engineering, and integration of additional metadata could substantially enhance classification accuracy and reliability.

Future research directions include:

- Developing more sophisticated and domain-specific **prompt strategies** for LLMs to further improve precision and recall  
- Incorporating **metadata features** (such as author keywords, abstracts, or publication venues) to enrich input data and context  
- Exploring hybrid models combining classical ML and LLM techniques to leverage the strengths of both approaches

Ultimately, this study highlights a promising pathway towards scalable, efficient, and accurate management of scientific document classification in Portuguese-language educational research.

---
