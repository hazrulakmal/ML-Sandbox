def create_dataset(self, csv_path, x_label, y_label, max_seq=True): 
    """ 
    split pandas dataframe into train set, validation set and test set
    
    Parameters
    ----------
    csv_path : str
        path where the dataset is stored

    max_seq : boolen 
        Specify config as to wether to use the whole sentence or truncate it based a default maximum sequence. Default value is True
        
    Returns:
    dictionary: keys (train, validation, test), values (the sets)
    """

    df = pd.read_csv(csv_path)

    self.xlabel = x_label
    self.ylabel = y_label

    #transform non-numerical lables or normalise labels
    le = LabelEncoder()
    df[y_label] = le.fit_transform(df[y_label])

    if max_seq:
        self.config.max_seq_len = self.find_max_length(df[x_label])

    training_df, test_df = train_test_split(df,
                                            test_size= self.config.test_size ,
                                            random_state= self.config.seed,
                                            shuffle= True,
                                            )

    train_df, validation_df = train_test_split(training_df,
                                            test_size= self.config.val_size,
                                            random_state= self.config.seed,
                                            shuffle= True,
                                            )
    
    self.data["train"] = train_df
    self.data["validation"] = validation_df
    self.data["test"] = test_df

def find_max_length(self, text):
    self.config.max_seq_len = len(max(text, key=lambda x: len(x.split())).split())