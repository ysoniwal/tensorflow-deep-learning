# tensorflow-deep-learning

There are 10 notebooks + 1 fundamentals notebook in this Repo. These are creating as part of Tensorflow Deep Learning course on [Udemy] (https://www.udemy.com/course/tensorflow-developer-certificate-machine-learning-zero-to-mastery/)

Description

1. 00_tensorflow_fundamentals.ipynb -> Tensorflow constants, variables, math operations, transformations, numpy conversion, shapes, len, ndim
2. 01_neural_network_regression_with_tensorflow.ipynb -> Regression using tensorlow for https://www.kaggle.com/datasets/mirichoi0218/insurance dataset. Topics covered: Model building using Sequential API, Model compiling (adding loss, optimizer and metrics) and Model Training
3. 02-neural_network_classification_with_tensorflow.ipynb -> Classification using tensorflow. Topics covered: LearningRateScheduler callback, using history to plot loss curves, Multiclass classification on MNIST data for 10 classes, Adding `validation_data` during training, plotting confusion matrix for multiclass classification
4. 03-introduction_to_computer_vision_with_tensrflow.ipynb -> Computer vision using Tensorflow. Topics covered: Normalizing and Reshaping images, Data Augmentation, Dropout, MaxPool, Using ImageDataGenerator and flow_from_directory preprocessing methods to create data from directory, Create TinyVGG architecture
5. 04-transfer_learning_with_tensorflow_part1_features_extraction.ipynb -> Transfer learning using feature extraction. Topics covered: Using tf.keras.utils.image_dataset_from_directory() to read images from directory, EfficientNet_B0 and Resnet_v2_50 using tensorflow_hub, Uploading records to tensorboard
6. 05-transfer_learning_with_tensorflow_part2_fine_tuning.ipynb -> Transfer learning using fine tuning. Topics covered: Using prebuilt model from https://www.tensorflow.org/api_docs/python/tf/keras/applications/, feature extraction followed by fine tuning with reduced learning rate, unfreezing layers and adding initial_epochs in training 
7. 06-transfer_learning_with_tensorflow_part3_scaling_up.ipynb -> Data Augmentation through layers, save and load model
8. 07-milestone_project_1_food_101.ipynb -> Beat DeepFood paper using fine tuning EfficientNetB0. Topics covered: Mixed precision Training, TFDS, Data Pipelines, Batching the data, Prefetching.
9. 08-introduction_to_nlp_with_tensorflow.ipynb -> NLP for disaster tweet classification data. Topics covered: Tokenization, Naive Bayes, Dense Model, Word Embeddings, Visualizing word embedding using Tensoflow Projector, LSTM, GRU, Bidirectional LSTM, Conv1D, Feature Extractor model using tensorflow_hub Universal Sentence Encoder, Most wrong predictions, Speed/score tradeoff
10. 09_skimlit_milestone_project2.ipynb -> Replicating paper: https://arxiv.org/abs/1710.06071 (data) and https://arxiv.org/abs/1612.05251 (architecture) -> Multiclass classification of PubMed paper abstracts. Topics covered: Character level embeddings, Multimodal Model, Create tf.data.Dataset for multiple inputs
11. 10_time_series_fundamentals_with_tensorflow.ipynb -> Forecast Bitcoin price using 7 years of historical data. Topics covered: Naive forecasting, windows and horizons, creating windows, N-BEATS algorithm, Creating custom layers in Tensorflow, Using Lambda layers, ensemble models, confidence intervals, turkey problem in forecasting
