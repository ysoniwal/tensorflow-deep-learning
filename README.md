# tensorflow-deep-learning

There are 10 notebooks + 1 fundamentals notebook in this repo. These are created as part of Tensorflow Deep Learning course on [Udemy](https://www.udemy.com/course/tensorflow-developer-certificate-machine-learning-zero-to-mastery/)

Description:

1. 00_tensorflow_fundamentals.ipynb
  * `tf.constant`, `tf.Variable`, `.assign`
  * `random` class
  * `numpy` to tensor
  * `shape`, `ndim`
  * Other ways to create tensors
  * indexing
  * Math operations : `+,-,/,*`
  * Matrix multiplication
  * Reshape
  * Transpose
  * Aggregations
  * GPU configuration
2. 01_neural_network_regression_with_tensorflow.ipynb -> 
  * Regression using tensorlow for https://www.kaggle.com/datasets/mirichoi0218/insurance dataset
  * Model building using `Sequential` API
  * `model.compile` (adding `loss`, `optimizer` and `metrics`)
  * `model.fit`
  * `model.summary`
  * `model.evaluate`
  * `model.predict`
  * Download files from google colab
  * one-hot encoding in `pandas`
  * `tf.keras.utils.plot_model`
  * plotting model `history`
3. 02-neural_network_classification_with_tensorflow.ipynb -> Classification using tensorflow. Topics covered: `LearningRateScheduler` callback, using history to plot loss curves, multiclass classification on MNIST data for 10 classes, Adding `validation_data` during training, plotting confusion matrix for multiclass classification
4. 03-introduction_to_computer_vision_with_tensrflow.ipynb -> Computer vision using Tensorflow. Topics covered: normalizing and reshaping images, data augmentation, dropout, maxPool, using `ImageDataGenerator` and `flow_from_directory` preprocessing methods to create data from directory, create TinyVGG architecture
5. 04-transfer_learning_with_tensorflow_part1_features_extraction.ipynb -> Transfer learning using feature extraction. Topics covered: using `tf.keras.utils.image_dataset_from_directory` to read images from directory, `EfficientNet_B0` and `Resnet_v2_50` using `tensorflow_hub`, uploading records to tensorboard
6. 05-transfer_learning_with_tensorflow_part2_fine_tuning.ipynb -> Transfer learning using fine tuning. Topics covered: Using prebuilt model from https://www.tensorflow.org/api_docs/python/tf/keras/applications/, feature extraction followed by fine tuning with reduced learning rate, unfreezing layers and adding initial_epochs in training 
7. 06-transfer_learning_with_tensorflow_part3_scaling_up.ipynb -> Data augmentation through layers, save and load model
8. 07-milestone_project_1_food_101.ipynb -> Beat DeepFood paper using fine tuning EfficientNetB0. Topics covered: Mixed precision Training, TFDS, data pipelines, Batching the data, prefetching
9. 08-introduction_to_nlp_with_tensorflow.ipynb -> NLP for disaster tweet classification data. Topics covered: tokenization, naive bayes, dense model, word embeddings, visualizing word embedding using `Tensoflow Projector`, `LSTM`, `GRU`, `Bidirectional LSTM`, `Conv1D`, feature Extractor model using tensorflow_hub `Universal Sentence Encoder`, Most wrong predictions, Speed/score tradeoff
10. 09_skimlit_milestone_project2.ipynb -> Replicating paper: https://arxiv.org/abs/1710.06071 (data) and https://arxiv.org/abs/1612.05251 (architecture) -> Multiclass classification of PubMed paper abstracts. Topics covered: character level embeddings, multimodal model, Create `tf.data.Dataset` for multiple inputs
11. 10_time_series_fundamentals_with_tensorflow.ipynb -> Forecast Bitcoin price using 7 years of historical data. Topics covered: naive forecasting, windows and horizons, creating windows, `causal` padding in Convolution, N-BEATS algorithm, creating custom layers in tensorflow using `subclassing`, Using Lambda layers, ensemble models, confidence intervals, turkey problem in forecasting
