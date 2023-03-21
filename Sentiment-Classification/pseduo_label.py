import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from datasets import DatasetDict
from tensorflow.keras.utils import to_categorical
from tensorflow.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class Config:
    """
    The configuration class for training.
    """
    def __init__(self,
                 max_seq_len = 64,
                 batch_size = 32,
                 do_lower_case = True,
                 seed = 12,
                 test_size = 0.2,
                 val_size = 0.2,
                 learning_rate = 5e-5,
                 epsilon = 1e-8,
                 weight_decay = 0.01,
                 num_labels = 3,
                 epochs = 10 ):
        """
        Parameters
        ----------
        max_seq_length: int
            The maximum length of the sequence to be used. Default value is 64.
        batch_size: int
            The batch size for training. Default value is 32.
        do_lower_case: bool
            Determines whether to make all training and evaluation examples lower case. Default is True.
        seed: int
            Random seed. Defaults to 12.
        test_size: float
            Proportion of test sample of whole dataset
        val_size: float
            Proportion of validation sample of the remaining dataset
        learning_rate: float
            The learning rate. Default value is 5e-5.
        epsilon: float
            The epsilon rate. Default value is 1e-8
        weight_decay: float
            The weight decay rate. Default value is 0.01
        num_labels: int
            The number of classification. Default value is 3
        epochs: int
            Number of epochs to train. Default value is 10.
        """
        self.max_seq_len = int(max_seq_len),
        self.batch_size = batch_size,
        self.do_lower_case = do_lower_case,
        self.seed = seed,
        self.test_size = test_size,
        self.val_size = val_size,
        self.learning_rate =  learning_rate,
        self.epsilon = epsilon,
        self.weight_decay = weight_decay,
        self.num_labels = num_labels,
        self.epochs = epochs

class DataAugmentation:
    """ 
    The main class for data augmenting
    """

    def __init__(self, config, model, benchmark_model, tokenizer):
        self.config = config
        self.models = {"benchmark_model": benchmark_model, "upgraded_model":  model}
        self.models_history = {}
        self.tokenizer = tokenizer
        self.xlabel = None #text
        self.ylabel = None #label
        self.data = {}
        self.tokenized_data = {}
        self.batched_data = {}
        self.accuracy = {}

    def tokenize_data(self, train, validation, test, additional_train, x_label, y_label):
        
        self.data["train"] = train
        self.data["validation"] = validation
        self.data["test"] = test    
        self.data["train_add"] = pd.concat([train, additional_train], axis=0) # mustcheck the 
 

        self.xlabel = x_label
        self.ylabel = y_label

        x_train = self.tokenizer(text = self.data["train"][self.xlabel].to_list(),
                                add_special_tokens = True,
                                max_length = self.config.max_seq_len[0],
                                truncation = True,
                                padding = True,
                                return_tensors = "tf",
                                return_token_type_ids = False,
                                return_attention_mask = True,
                                verbose = True)

        x_test =  self.tokenizer(text = self.data["test"][self.xlabel].to_list(),
                                add_special_tokens = True,
                                max_length = self.config.max_seq_len[0],
                                truncation = True,
                                padding = True,
                                return_tensors = "tf",
                                return_token_type_ids = False,
                                return_attention_mask = True,
                                verbose = True)

        x_val  =  self.tokenizer(text = self.data["validation"][self.xlabel].to_list(),
                                add_special_tokens = True,
                                max_length = self.config.max_seq_len[0],
                                truncation = True,
                                padding = True,
                                return_tensors = "tf",
                                return_token_type_ids = False,
                                return_attention_mask = True,
                                verbose = True)

        x_additional = self.tokenizer(text = self.data["train_add"][self.xlabel].to_list(),
                                add_special_tokens = True,
                                max_length = self.config.max_seq_len[0],
                                truncation = True,
                                padding = True,
                                return_tensors = "tf",
                                return_token_type_ids = False,
                                return_attention_mask = True,
                                verbose = True)
                    
        self.tokenize_data["train"] = dict(x_train)
        self.tokenize_data["test"]= dict(x_test)
        self.tokenize_data["val"] = dict(x_val)
        self.tokenize_data["train_add"] = dict(x_additional)


    def process_data(self, train, validation, test, additional_train, x_label, y_label):
        
        self.tokenize_data(train, validation, test, additional_train, x_label, y_label) 
        
        # Creating labels for each of the data splits
        train_labels = to_categorical(self.data["train"][self.ylabel])
        val_labels = to_categorical(self.data["val"][self.ylabel])
        test_labels = to_categorical(self.data["test"][self.ylabel])
        train_add_labels = to_categorical(self.data["train_add"][self.ylabel])

        # Creating TF Datasets for each of our data splits
        train_dataset = Dataset.from_tensor_slices((self.tokenize_data["train"], train_labels))
        val_dataset = Dataset.from_tensor_slices((self.tokenize_data["val"], val_labels))
        test_dataset = Dataset.from_tensor_slices((self.tokenize_data["test"], test_labels))
        train_add_dataset = Dataset.from_tensor_slices((self.tokenize_data["train_add"], test_labels))

        # Shuffling and batching our data
        train_dataset = train_dataset.shuffle(len(self.tokenize_data["train"]), seed=self.config.seed[0]).batch(self.config.batch_size[0])
        val_dataset = val_dataset.shuffle(len(self.tokenize_data["val"]), seed=self.config.seed[0]).batch(self.config.batch_size[0])
        test_dataset = test_dataset.shuffle(len(self.tokenize_data["test"]), seed=self.config.seed[0]).batch(self.config.batch_size[0])
        train_add_dataset = train_dataset.shuffle(len(self.tokenize_data["train_add"]), seed=self.config.seed[0]).batch(self.config.batch_size[0])

        self.batched_data["train"] = train_dataset
        self.batched_data["val"] = val_dataset  
        self.batched_data["test"] = test_dataset
        self.batched_data["train_add"] = train_add_dataset

    def validate(self,  tokenized_train, y_true_train, tokenized_test, y_true_test):
        
        y_pred_train = model.predict({"input_ids" : tokenized_train["input_ids"] ,"attention_mask" : tokenized_train["attention_mask"]})
        # Converting predicted logits to probabilities
        predictions_df_train = tf.nn.softmax(y_pred_train.logits)
        # Extracting the indices with the highest probabilities
        predictions_train =  tf.argmax(predictions_df_train, axis=1).numpy()
        accuracy_train = accuracy_score(y_true_train, predictions_train)
        
        y_pred_test = model.predict({"input_ids" : tokenized_test["input_ids"] ,"attention_mask" : tokenized_test["attention_mask"]})
        # Converting predicted logits to probabilities
        predictions_df_test = tf.nn.softmax(y_pred_test.logits)
        # Extracting the indices with the highest probabilities
        predictions_test = tf.argmax(predictions_df_test, axis=1).numpy()
        accuracy_test= accuracy_score(y_true_test, predictions_test)
        
        return accuracy_train, accuracy_test

    def update_validation(self):
        self.accuracy["benchmark"] = self.validate(self.tokenize_data["train"], self.data["train"][self.y_label], self.tokenize_data["test"], self.data["test"][self.y_label])
        self.accuracy["upgraded"] = self.validate(self.tokenize_data["train_add"], self.data["train_add"][self.y_label], self.tokenize_data["test"], self.data["test"][self.y_label])

    
    def fit(self): 
    # A function defining our learning rate schedule
        def lr_decay(epoch, lr):
            if epoch <= 10:
                return lr
            else:
                return lr * np.exp(-0.1 * epoch)

        # Instantiating our learning rate scheduler callback
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule=lr_decay, verbose=1)

        for model in self.models:
            self.models[model].compile(optimizer=tf.keras.optimizers.Adam(learning_rate= self.config.learning_rate[0],
                                                                        epsilon = self.config.epsilon[0], 
                                                                        decay  = self.config.weight_decay[0], 
                                                                        clipnorm = 1.0), 
                                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                                        metrics=tf.keras.metrics.CategoricalAccuracy())


        # Setting some hyperparameters and compiling the model
        #self.benchmark_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= self.config.learning_rate,
        #                                                               epsilon = self.config.epsilon , 
        #                                                                decay  = self.config.weight_decay , 
        #                                                                clipnorm = 1.0), 
        #                            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
        #                            metrics=tf.keras.metrics.CategoricalAccuracy())

        #self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= self.config.learning_rate,
        #                                                    epsilon = self.config.epsilon , 
        #                                                    decay  = self.config.weight_decay , 
        #                                                    clipnorm = 1.0), 
        #                            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
        #                            metrics=tf.keras.metrics.CategoricalAccuracy())

        #Training the model
        print("Benchmark model in training")
        self.model_history["benchmark_model"]  = self.models["benchmark_model"].fit(self.batched_data["train"],
                                                                                    validation_data=self.batched_data["val"], 
                                                                                    epochs=self.config.epoch[0], callbacks=[lr_scheduler])

        print("Upgraded model in training")
        self.model_history["upgraded_model"]  = self.models["upgraded_model"].fit(self.batched_data["train_add"],
                                                                                    validation_data=self.batched_data["val"], 
                                                                                    epochs=self.config.epoch[0], callbacks=[lr_scheduler])
        self.update_validation()
        

    def get_evaluate(self):
        benchmark_model_df = pd.DataFrame(self.model_history["benchmark_model"].history , columns = ['loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy'])
        upgraded_model_df =  pd.DataFrame(self.model_history["upgraded_model"].history  , columns = ['loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy'])
    
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1,figsize=(15,10))
        ax1.plot(benchmark_model_df["categorical_accuracy"]) 
        ax1.plot(benchmark_model_df["val_categorical_accuracy"])       
        ax1.set(xlabel="Epochs",ylabel="Accuracy")
        ax1.set_label("Benchmark Model Accuracy")

        ax2.plot(benchmark_model_df["loss"]) 
        ax2.plot(benchmark_model_df["val_loss"])       
        ax2.set(xlabel="Epochs", ylabel="Loss")
        ax2.set_label("Benchmark Model Loss")

        ax3.plot(upgraded_model_df["categorical_accuracy"]) 
        ax3.plot(upgraded_model_df["val_categorical_accuracy"])       
        ax3.set(xlabel="Epochs",ylabel="Accuracy")
        ax3.set_label("Upgraded Model Accuracy")

        ax4.plot(upgraded_model_df["loss"]) 
        ax4.plot(upgraded_model_df["val_loss"])       
        ax4.set(xlabel="Epochs", ylabel="Loss")
        ax4.set_label("Upgraded Model Loss")

        plt.show()


    def get_validation(self):
        validation_dict = {"train_accuracy": [self.accuracy["benchmark"][0], self.accuracy["upgraded"][0]], "test_accuracy": [self.accuracy["benchmark"][1], self.accuracy["upgraded"][1]]}
        validation_df = pd.DataFrame(validation_dict, index=["benchmark", "upgraded"])   
        return validation_df                                        
