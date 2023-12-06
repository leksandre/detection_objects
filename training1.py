from imageai.Prediction.Custom import ModelTraining
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir('C:\python_work08052021') 
model_trainer = ModelTraining()
# model_trainer.setModelTypeAsResNet()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("idenauto")
# model_trainer.trainModel(num_objects=4, num_experiments=200, enhance_data=True, batch_size=64, show_network_summary=True, continue_from_model = 'C:\python_work08052021\idenauto\models\model_ex-001_acc-1.000000.h5' )

model_trainer.trainModel(num_objects=4, num_experiments=5, enhance_data=True, batch_size=64, show_network_summary=True, save_full_model = True )#

#
"""
'trainModel()' function starts the model actual training. It accepts the following values:
- num_objects , which is the number of classes present in the dataset that is to be used for training
- num_experiments , also known as epochs, it is the number of times the network will train on all the training dataset
- enhance_data (optional) , this is used to modify the dataset and create more instance of the training set to enhance the training result
- batch_size (optional) , due to memory constraints, the network trains on a batch at once, until all the training set is exhausted. The value is set to 32 by default, but can be increased or decreased depending on the meormory of the compute used for training. The batch_size is conventionally set to 16, 32, 64, 128.
- initial_learning_rate(optional) , this value is used to adjust the weights generated in the network. You rae advised to keep this value as it is if you don't have deep understanding of this concept.
- show_network_summary(optional) , this value is used to show the structure of the network should you desire to see it. It is set to False by default
- training_image_size(optional) , this value is used to define the image size on which the model will be trained. The value is 224 by default and is kept at a minimum of 100.
- continue_from_model (optional) , this is used to set the path to a model file trained on the same dataset. It is primarily for continuos training from a previously saved model.
- transfer_from_model (optional) , this is used to set the path to a model file trained on another dataset. It is primarily used to perform tramsfer learning.
- transfer_with_full_training (optional) , this is used to set the pre-trained model to be re-trained across all the layers or only at the top layers.
- initial_num_objects (required if 'transfer_from_model' is set ), this is used to set the number of objects the model used for transfer learning is trained on. If 'transfer_from_model' is set, this must be set as well.
- save_full_model ( optional ), this is used to save the trained models with their network types. Any model saved by this specification can be loaded without specifying the network type.



- num_objects - количество классов, присутствующих в наборе данных, которые будут использоваться для обучения.
- num_experiments, также известные как эпохи, это количество раз, когда сеть будет обучаться на всем наборе обучающих данных.
--hance_data (необязательно), это используется для изменения набора данных и создания дополнительных экземпляров обучающего набора для улучшения результата обучения
- batch_size (необязательно), из-за ограничений памяти сеть обучается на пакете сразу, пока не будет исчерпан весь обучающий набор. По умолчанию установлено значение 32, но его можно увеличить или уменьшить в зависимости от объема вычислений, используемых для обучения. Batch_size обычно устанавливается равным 16, 32, 64, 128.
- initial_learning_rate (необязательно), это значение используется для настройки весов, генерируемых в сети. Вы посоветовали оставить это значение как есть, если у вас нет глубокого понимания этой концепции.
- show_network_summary (необязательно), это значение используется для отображения структуры сети, если вы хотите ее увидеть. По умолчанию установлено значение False.
- training_image_size (необязательно), это значение используется для определения размера изображения, на котором будет обучаться модель. По умолчанию значение равно 224 и поддерживается как минимум 100.
- continue_from_model (необязательно), используется для установки пути к файлу модели, обученному на том же наборе данных. Это в первую очередь для непрерывного обучения по ранее сохраненной модели.
- transfer_from_model (необязательно), используется для установки пути к файлу модели, обученному на другом наборе данных. Он в основном используется для обучения перемещению.
- transfer_with_full_training (необязательно), это используется для установки предварительно обученной модели для повторного обучения на всех уровнях или только на верхних уровнях.
- initial_num_objects (требуется, если установлено 'transfer_from_model'), это используется для установки количества объектов, на которых обучается модель, используемая для трансферного обучения. Если установлено 'transfer_from_model', это также должно быть установлено.
- save_full_model (необязательно), это используется для сохранения обученных моделей с их типами сетей. Любая модель, сохраненная в этой спецификации, может быть загружена без указания типа сети.


"""