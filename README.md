Optimizing CIFAR-10 Classification with a Modified ResNet-34 Deep Learning Model
Author: Yinuo Cao, Yufei Wang
1. Introduction: 
In this project, we developed a deep learning model based on the ResNet architecture for the CIFAR-10 image classification task. Initially, the CIFAR-10 dataset was loaded using a custom unpickle function, and the images underwent a series of preprocessing steps including random rotation, cropping, horizontal flipping, and conversion into tensors followed by normalization. The CustomCIFAR10Dataset class was used to package the preprocessed data into a dataset suitable for batch processing. The model structure utilizes basic residual blocks from ResNet, which consist of sequential convolutional layers, batch normalization, and ReLU activation functions, with appropriate downsampling to ensure consistency in feature dimensions. Our model begins with an initial convolutional layer, progresses through multiple residual blocks, integrates features via a global average pooling layer, and outputs the final classification results through a fully connected layer. During the training process, the Adam optimizer and cross-entropy loss function were employed to optimize the model's parameters over multiple iterations, while logging the loss for each epoch to monitor training progress.

2. Method:  
We employed the ResNet-34 architecture, a deep convolutional neural network composed of multiple residual blocks, each containing two layers with the same number of filters. These blocks are designed to address the training challenges of deeper networks by incorporating skip connections that allow gradients to flow directly through several layers. The entire network includes an initial convolutional layer followed by four groups of residual blocks, with a varying number of convolutional layers within each group and an increase in the number of feature channels as the network progresses. To cater to the CIFAR-10 dataset and reduce overfitting, we applied various image transformation techniques during the data preprocessing stage. These include random rotations, cropping, and horizontal flipping, which not only enrich the diversity of the training data but also help the model learn to recognize objects from different angles and scales, thereby enhancing the model's generalization ability.

3. Experimental Setup

Training Process
In order to optimize the model performance, we conducted several experiments involving ResNet architectures of different complexity: ResNet18, ResNet34, and ResNet101. after 50 iterations, ResNet101 with a learning rate (lr) of 0.001 performed very well, reaching 98% accuracy. However, the model contains more than 5 million parameters, so efforts need to be made to reduce the complexity of the model.

A comparative analysis of the ResNet18 and ResNet34 models in a four-layer structure reveals that ResNet34 consistently outperforms ResNet18, and in order to further reduce the number of parameters while maintaining the efficacy, we modified the structure of ResNet34 by reducing the number of layers from four to three and integrating a bottleneck function. This modification improved performance compared to the four-layer ResNet18, and we ultimately chose the three-layer ResNet34 with the bottleneck function.

The bottleneck design resulted in several improvements: reduced dimensionality and overfitting, increased depth, and support for regular applications. The feature dimensionality is compressed at the outset using a 1x1 convolution, which acts as a regularizer and thus improves generalization. In addition, adaptive pooling replaces the last convolutional block of the original ResNet structure, thus adapting to different input sizes and enhancing the flexibility of the model. This technique ensures that the spatial dimension is reduced to 1x1 before reaching the final fully connected layer, thus enhancing the robustness of the model. The final classification is done by a linear layer and the preceding complexity reduction improves the accuracy of the CIFAR-10 dataset.


Learning Rate Exploration
Adjustments were made to optimize the learning rate, with experiments conducted in the range of [0.1, 0.01, 0.001, 0.0001, 0.00001]. The resulting loss graph indicated that the model achieved minimal loss at a learning rate of 0.001, with higher losses observed at both higher and lower rates.

At a conceptual level, the iterative optimization via gradient descent involves computing a direction of descent and updating model parameters accordingly. The learning rate influences the size of these updates:

High Learning Rate (e.g., 0.1): Initially expedites convergence but may prevent fine-tuning of weights, leading to suboptimal solutions or divergence.
Optimal Learning Rate (e.g., 0.001): Facilitates efficient learning across a sufficiently broad range of step sizes, enabling rapid progress while accommodating complex loss landscapes.
Low Learning Rate (e.g., 1e-05): Ensures stable convergence through smaller updates but may slow the training process excessively, potentially causing the model to become stuck in local minima or overfit.

Future Improvements
To enhance training efficiency, an adaptive learning rate approach is recommended. Starting with a higher learning rate and gradually reducing it could merge the benefits of rapid convergence with the precision of smaller updates. This strategy may help in achieving a balance between speed and accuracy in model training.

 4. Results and discussion

Model architecture and parameters
The final realized model is a modified version of the ResNet34 architecture, designed to accommodate the limitations of the number of parameters while maintaining high accuracy. The architecture uses BasicBlock as a regular layer, with a slight adjustment to the number of layers to reduce the total number of parameters. The model starts with a convolutional layer containing 32 filters, followed by three main layers with decreasing numbers of layer blocks to optimize performance versus complexity.

- Initial layer: 32 filters with 3x3 kernel size.
- Layer 1: 3 blocks with step size 1, using 32 filters.
- Layer 2: 4 blocks in steps of 2, using 64 filters.
- Layer 3: 5 blocks (initially 6, reduced by 1) with a step of 2, gradually increasing to 128 filters.
- Last layer: global average pooling followed by a linear layer with 10 levels.

After these adjustments, the total number of parameters was drastically reduced to 1,753,258, ensuring a lightweight and effective model.

Training and testing performance
The model was optimized using the Adam optimizer, and the model was trained at a learning rate of 0.001 for 50 epochs. The training process used standard data enhancement techniques such as random cropping and level flipping to ensure that even small changes in the input data remain stable.

The final test accuracy for the CIFAR-10 dataset was 98.36%. Such a high accuracy rate indicates that the model has a strong ability to generalize from the training data to the unseen test data despite the reduced complexity and parameters.

Discussion.
The choice of learning rate plays a crucial role in achieving a high accuracy rate. a learning rate of 0.001 proved to be the best choice for balancing convergence speed and accuracy, as higher or lower learning rates led to poorer results. 
In addition, the adaptive structure of the model, including the reduction of the number of layers and the strategic use of bottleneck features, is crucial for maximizing the depth and effectiveness of the network while keeping the number of parameters low. The bottleneck structure in particular helps to reduce dimensionality and overfitting, which is critical for improving performance on a relatively small dataset such as CIFAR-10.



