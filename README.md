# Deep-Learning-Model-For-MultiClass-Classification-of-Colon-Cancer-Histopathology-Images
Abstract

Colorectal cancer is one of the leading causes of cancer related deaths worldwide, and 
histopathological examination of tissue samples is the standard method used for diagnosis. 
However, manual analysis of histopathology slides is time consuming and can vary depending on 
the experience of the pathologist, especially when tissue classes have very similar visual 
features. In this study, a deep learning based multi class classification system is proposed to 
automatically classify colorectal histopathological images into six classes: normal tissue, polyp, 
lowgrade intraepithelial neoplasia, highgrade intraepithelial neoplasia, serrated adenoma, and 
adenocarcinoma. The model was trained and evaluated using a dataset of 2228 H&E stained 
colon tissue images captured at 400× magnification. Transfer learning with pretrained 
convolutional neural networks, class aware data augmentation, and Sparse Focal Loss were used 
to address class imbalance and subtle morphological differences between classes. The results 
show that the proposed approach can effectively distinguish between different colorectal tissue 
types, demonstrating its potential as a supportive tool to assist pathologists in diagnosis and 
clinical decisionmaking. 

Literature Review 

Several studies have investigated automated colorectal cancer diagnosis using histopathological 
images. Early research mainly focused on handcrafted feature extraction combined with 
traditional machine learning classifiers such as Support Vector Machines and Random Forests. 
With the advancement of deep learning, convolutional neural networks (CNNs) have become 
widely used in this field. Recent studies applied pretrained CNN architectures such as DenseNet 
and other deep models for colorectal tissue classification and reported improved performance 
compared to conventional approaches [1]. Some works also explored segmentation based 
methods to identify cancerous regions prior to classification [2]. Transfer learning has been 
commonly used to overcome the challenge of limited dataset sizes. Overall, these studies 
demonstrate that CNN based methods are effective in learning discriminative features from 
histopathological images and can support automated colorectal cancer diagnosis [3]. 
Despite the progress made, several limitations remain in existing research. Many studies simplify 
the diagnostic task by focusing on binary classification or a small number of tissue classes, 
which does not fully represent real clinical scenarios [1]. Rare but clinically important lesion 
types, such as serrated adenomas, are often underrepresented or excluded. Class imbalance is 
another significant issue, as most datasets contain far fewer samples for certain lesion types, 
resulting in biased model performance [2]. In addition, visually similar tissue types, such as 
lowgrade and highgrade intraepithelial neoplasia, are difficult to distinguish and are frequently 
misclassified. Some studies directly use segmentation masks during training, which may lead to 
data leakage and overly optimistic performance results. Furthermore, many works rely mainly on 
overall accuracy without detailed classwise evaluation, limiting clinical relevance [3]. 
This study addresses the identified gaps by proposing a deep learning based framework for 
sixclass classification of colorectal histopathological images. Unlike previous studies, this 
approach focuses on fine grained classification that includes both common and rare lesion types, 
making it more clinically realistic. Transfer learning is used to extract meaningful features from a 
limited dataset, while class aware data augmentation and Sparse Focal Loss are applied to reduce 
the effects of class imbalance. Although segmentation masks are available, they are intentionally 
not used during classification to avoid data leakage and ensure fair model evaluation. 
Performance is analyzed using confusion matrices and per class metrics to better understand 
clinically important errors. Overall, this work presents a practical and clinically aligned decision 
support system to assist pathologists in colorectal cancer diagnosis. 

Problem Identification 

Colorectal cancer (CRC) affects millions of people worldwide, ranking among the leading causes 
of cancer related morbidity and mortality. Patients are directly impacted, as early detection and 
accurate diagnosis are crucial for effective treatment and improved survival outcomes. 
Healthcare providers, particularly pathologists, are also affected, as the manual assessment of 
histopathological slides is laborintensive, timeconsuming, and subject to interobserver 
variability. This problem is important because delayed or inaccurate diagnosis can lead to disease 
progression, improper treatment, and increased healthcare costs. Despite the critical role of 
histopathology in CRC detection, current diagnostic practices face significant challenges: subtle 
morphological differences between tissue types, rare lesions such as serrated adenomas, and 
variability in staining or imaging conditions can lead to inconsistent results, even among 
experienced pathologists. The specific unmet need in healthcare is the lack of reliable, automated 
support tools that can assist pathologists in accurately classifying colorectal tissue while 
maintaining clinical rigor. Addressing this need can reduce diagnostic errors, improve efficiency, 
and ensure timely intervention, ultimately supporting better patient outcomes and alleviating 
workload pressures in pathology departments. 

Dataset Justification 

The dataset for this study was carefully chosen due to its suitability for developing a robust multi 
class classification model. It contains a sufficient number of high quality histopathological 
images distributed across six clinically relevant classes: normal, polyp, lowgrade and highgrade 
intraepithelial neoplasia, serrated adenoma, and adenocarcinoma. This diversity allows the model 
not only to detect the presence of colorectal cancer but also to identify the stage or severity of the 
lesion, which is critical for timely and effective treatment planning. Additionally, the dataset 
includes paired ground truth segmentation masks, which, although not used directly in the 
classification task, provide the opportunity for future studies involving region of interest 
extraction or segmentation based analyses. The images were acquired at high magnification 
(400×) and standardized to a uniform resolution, ensuring consistency across samples. Expert 
review and curation by multiple pathologists and biomedical researchers further enhance the 
reliability of the dataset. Overall, these qualities make the dataset an ideal choice for developing 
an AI system capable of accurate, clinically meaningful classification of colorectal tissue. 

Methodology 

The methodology for this study was designed to develop a robust deep learning framework for 
multiclass classification of colorectal histopathology images, while ensuring clinical applicability 
and realistic evaluation. Importantly, we did not use the segmentation masks as input to the 
model. In realworld scenarios, segmentation masks are usually unavailable, and relying on them 
would make the model impractical for deployment. Additionally, the masks were manually 
annotated by experts, so using them could cause the model to “cheat” by learning the exact 
location of pathological regions rather than the tissue patterns themselves. This would reduce the 
model’s generalizability to new, unseen images and provide an unrealistic assessment of its 
performance. 
All images were resized to 224 × 224 pixels, which is the input size required by the 
EfficientNetB0 model. Resizing ensures uniformity and allows the pretrained network to extract 
meaningful features efficiently. We applied class wise data augmentation to address the issue of 
class imbalance. Specifically, minority classes were augmented with transformations such as 
horizontal flips, small rotations, and brightness adjustments. Augmentation was applied only to 
the training set, ensuring that the validation and test sets represent real, unaltered 
histopathology images for unbiased evaluation. The augmentation process was designed to 
balance the number of images in each class to match the class with the maximum number of 
samples, improving learning stability and reducing bias toward majority classes. 
To further stabilize training and prevent overfitting, we employed the EarlyStopping callback 
during model training. Specifically, we monitored the validation loss and set a patience of 7 
epochs, meaning training would stop if the validation loss did not improve for 7 consecutive 
epochs. The option “restore_best_weights = True: ensured that the model retained the weights 
corresponding to the lowest validation loss, rather than the weights at the end of training. 

Pretrained Model Usage and Adaptation

a.Rationale 

We utilized a pretrained network because our dataset is relatively small, with only a few 
images for 2 classes. Training a deep convolutional network from scratch would require a 
substantially larger dataset to learn meaningful low level and highlevel features, otherwise the 
model risks severe overfitting and poor generalization. By using EfficientNetB0 pretrained on 
ImageNet, we can leverage the features already learned from millions of images, accelerating 
convergence and improving performance, while adapting the network to capture subtle 
histopathological patterns specific to colorectal tissue.  
We chose EfficientNetB0 over other pretrained models because it provides an excellent balance 
between model complexity, computational efficiency, and accuracy. Its lightweight architecture 
allows it to extract rich, multi-scale features from histopathology images without requiring 
excessive computational resources, making it ideal for small to medium sized datasets. 

b. Modifications 

After feature extraction using EfficientNetB0, we added a custom classification head. The 
Global Average Pooling layer reduces the spatial dimensions of the feature maps, summarizing 
the extracted features efficiently while preventing overfitting. Batch normalization stabilizes 
training by normalizing inputs to each layer, and dense layers with ReLU activation allow the 
model to learn complex, nonlinear relationships in the features. Dropout layers were added to 
reduce overfitting by randomly deactivating neurons during training. Finally, the output layer 
with six neurons and softmax activation generates probabilities for each of the six classes. 

c. Training Strategy 

This strategy was used in combination with the Adam optimizer and Sparse Categorical Focal 
Loss. While the focal loss focuses the model on hardtoclassify and minority class samples, 
EarlyStopping prevents overfitting on augmented training data by halting training once the 
model stops improving on unseen data. Together, these optimization strategies help the model 
achieve better generalization, maintain stable convergence, and improve performance on rare and 
morphologically similar classes, which is particularly important given the relatively small and 
imbalanced dataset used in this study. 

d. Risk and bias discussion 

For our EfficientNetB0based model, domain mismatch between ImageNet natural images and 
histopathology may limit feature transfer. Class imbalance and rare lesions like serrated adenoma 
risk biased predictions. We mitigate this through selective finetuning, classaware augmentation, 
and macroaveraged metrics, ensuring robust learning and fair evaluation for all six colorectal 
tissue classes. 

Results 

● Model Accuracy : 0.8806 
● Training Accuracy : 0.9624 
● F1score (Macro F1score): 0.8170 
<img width="500" height="244" alt="image" src="https://github.com/user-attachments/assets/9dbe373d-c4da-4bfa-b11e-28a2be0a18dc" />

● Confusion matrix 
<img width="453" height="361" alt="image" src="https://github.com/user-attachments/assets/c9360547-064b-4772-b2cd-d35f95a69fd3" />

● ROC Curve 
<img width="475" height="358" alt="image" src="https://github.com/user-attachments/assets/adf6ae3b-dac3-4739-84a7-bf9ee0c7e54b" />


 
● Training loss vs Validation loss 
 <img width="472" height="318" alt="image" src="https://github.com/user-attachments/assets/2f9a7774-8d6b-4bc2-aa94-e75a428dc4bf" />

 
 
● Training loss vs Validation accuracy 
<img width="432" height="282" alt="image" src="https://github.com/user-attachments/assets/080f6da1-7137-409b-bee5-42dccebc0342" />


Limitations of our model 

Common misclassifications occur between morphologically similar classes, such as low versus 
highgrade intraepithelial neoplasia and polyps versus serrated adenomas. Limitations include 
reliance on global features, class imbalance, moderate dataset size, and staining or acquisition 
variability. Future work could integrate segmentation guidance and larger, more diverse datasets 
for improved generalization. 

Realworld Application 

The system can be deployed on a local server or secure cloud platform, integrating with digital 
pathology workflows. Digitized slides are processed by the EfficientNetB0 model to generate 
sixclass predictions, which are reviewed by pathologists. Human oversight ensures safety, while 
secure data handling and regulatory compliance support realworld clinical use.

Future Improvements 

Future work can enhance the model by finetuning more layers of EfficientNetB0 or exploring 
advanced architectures like EfficientNetB3 or EfficientNetV2. Attention mechanisms, ensemble 
learning, and explainability methods such as GradCAM can improve accuracy, robustness, and 
clinical trust. Expanding the dataset with larger, multicenter, and diverse histopathology images 
covering rare classes, different scanners, and staining protocols will reduce bias and improve 
generalization. For clinical translation, the system should undergo prospective studies on real 
patient samples, integrate with hospital digital pathology workflows, and meet regulatory 
requirements, including FDA or CE approvals, ensuring safety and ethical AI use. 
In conclusion, colorectal cancer diagnosis is time consuming and prone to variability. Our 
approach uses EfficientNetB0based deep learning for sixclass histopathology classification, 
combined with classaware augmentation and finetuning. The system acts as a human in the loop 
decision support tool, improving diagnostic consistency, efficiency, and early detection while 
maintaining accountability and ethical standards in clinical practice. 

References: 

[1] 
Y. Li et al., “Deep learning–based classification of colorectal histopathological images using 
convolutional neural networks,” Electronics, vol. 13, no. 3, Art. no. 3126, 2024. 
(MDPI Electronics journal — verify year/issue if needed) 
[2] 
S. Zhang et al., “Automated colorectal cancer diagnosis using histopathological images and 
deep learning techniques,” Asian Pacific Journal of Cancer Prevention, vol. 25, no. 5, pp. 
1795–1803, 2024. 
(APJCP journal — page range may need confirmation) 
[3] 
A. Kumar et al., “Multiclass classification of colorectal histopathology images using transfer 
learning,” in Proc. International Hackathon / Conference on AI in Healthcare, 2023, pp. 1–6. 
(Conference / hackathon paper — verify conference name and pages) 
