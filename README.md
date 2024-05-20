# KL-Grade-classification
- This project is conducted as a part of Healthhub AI Research Center.

## Dual Input Model 
- Medial and lateral images are convoluted separately, and concatenated afterwards. Dense layer makes predictions afterwards
- Histogram Equalization & Normalization. 
- Multitasking. We trained the network to make a second prediction: rather the X ray captures the left or right knee. The model converged very quickly. Loss weight 0.75:0.35 for KL grade and side, respectively, yielded the highest accuracy. Mean accuracy 70% (5-fold validation) 
- Result Summary: Mean accuracy 80% (5-fold cross validation). Near perfect accuracy for Class 2. 
![alt text](https://user-images.githubusercontent.com/21049855/102966384-21f0a600-4533-11eb-8491-8ef4c599afe3.png)

## Autoencoder 

## Adversarial Autoencoder 
- Adversarial autoencoder that is trained to follow Normal distribution (0,1). 
- Autoencoder model reconstructs original image. Kl grader predicts KL grade of the input image using latent vector.
- 50 % accuracy 
![alt text](https://user-images.githubusercontent.com/21049855/103620420-cb458c00-4f76-11eb-9d3c-018ce6209d7b.png)


