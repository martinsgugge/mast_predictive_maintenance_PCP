
class Metadata():
    train_run = None
    model_name = None
    pre_process_step = None
    training_start = None
    training_stop = None
    data_training_start = None
    data_training_stop = None
    train_size = None
    test_size = None
    validation_size = None
    hyperparameter = None
    accuracy = None
    test_accuracy = None
    number_of_epochs = None
    checkpoint_path = None
    previous_checkpoint_path = None
    features = None
    failed = False
    no_of_features = None
    no_outputs = None
    previous_data_training_start = None
    previous_checkpoint_path_list = None
    feature_type = None
    use_all = None
    raw = None
    aggregation_time = None
    average = None
    standard_deviation = None
    stateful = None

    header = ['model name', 'training run', 'pre-process steps', 'training_start', 'training_stop',
              'data_training_start', 'data_training_stop', 'train_test_split', 'hyperparameter', 'accuracy',
              'number_of_epochs', 'checkpoint path']
    csv_file = None

    def __init__(self, name):
        self.model_name = name
        print(self.model_name)
        # self.csv_file = CSV()
        # data = self.csv_file.csvToArray('./Metadata/' + self.model_name + '.csv', ';')
        #
        # if self.csv_file.generic_error:
        #
        #     self.train_run = str(1)
        #     self.csv_file.create_or_append_csv('./Metadata/' + self.model_name, self.header,
        #                                        [self.model_name, self.train_run])
        #
        # else:
        #     self.unpack_csv(data)
        #     self.train_run = str(int(self.train_run) + 1)

    def unpack_csv(self):
        pass
class MetadataSVM(Metadata):
    method = None
    c = None
    gamma = None
    coef0 = None

    def __init__(self, name):
        super().__init__(name)

class MetadataLSTM(Metadata):
    layers = None
    neurons_layer_one = None
    neurons_layer_two = None
    sequence = None
    sequence_steps = None
    pad = None
    batch_size = None

    def __init__(self, name):
        super().__init__(name)

    def unpack_csv(self, data):
        print(data)
        try:
            self.train_run = data[0]
        except IndexError as e:
            pass
        try:
            self.model_name = data[1]

            print(f'reading model name: {self.model_name}')
        except IndexError as e:
            pass
        try:
            self.pre_process_step = data[2]
        except IndexError as e:
            pass
        try:
            self.training_start = data[3]
        except IndexError as e:
            pass
        try:
            self.training_stop = data[4]
        except IndexError as e:
            pass
        try:
            self.data_training_start = data[5]
        except IndexError as e:
            pass
        try:
            self.data_training_stop = data[6]
        except IndexError as e:
            pass
        try:
            self.train_test_split = data[7]
        except IndexError as e:
            pass
        try:
            self.hyperparameter = data[8]
        except IndexError as e:
            pass
        try:
            self.accuracy = data[9]
        except IndexError as e:
            pass
        try:
            self.number_of_epochs = data[10]
        except IndexError as e:
            pass
        try:
            self.checkpoint_path = data[11]
        except IndexError as e:
            pass
        try:
            self.header = ['model name', 'training run', 'sensors', 'pre-process steps', 'training_start',
                           'training_stop', 'data_training_start', 'data_training_stop', 'train_test_split', 'Layers',
                           'Neurons (layer1|2)', 'accuracy', 'number_of_epochs', 'checkpoint path']
        except IndexError as e:
            pass

    def write_data(self, train_run=None, model_name=None, pre_process_step=None, training_start=None,
                   training_stop=None, data_training_start=None, data_training_stop=None, train_test_split=None,
                   layers=None, neurons='0|0', accuracy=None, number_of_epochs=None, checkpoint_path=None):
        if train_run is not None:
            self.train_run = train_run
        elif self.train_run is not None:
            pass
        else:
            self.train_run = ''

        if model_name is not None:
            print(f'input model name: {model_name}')
            self.model_name = model_name
        elif self.model_name is not None:
            print(f'getting name from self: {model_name}')
            pass
        else:
            self.model_name = ''

        if pre_process_step is not None:
            self.pre_process_step = pre_process_step
        elif self.pre_process_step is not None:
            pass
        else:
            self.pre_process_step = ''
        if training_start is not None:
            self.training_start = training_start
        elif self.training_start is not None:
            pass
        else:
            self.training_start = ''
        if training_stop is not None:
            self.training_stop = training_stop
        elif self.training_stop is not None:
            pass
        else:
            self.training_stop = ''
        if data_training_start is not None:
            self.data_training_start = data_training_start
        elif self.data_training_start is not None:
            pass
        else:
            self.data_training_start = ''
        if data_training_stop is not None:
            self.data_training_stop = data_training_stop
        elif self.data_training_stop is not None:
            pass
        else:
            self.data_training_stop = ''
        if train_test_split is not None:
            self.train_test_split = train_test_split
        elif self.train_test_split is not None:
            pass
        else:
            self.train_test_split = ''
        if layers is not None:
            self.layers = layers
        elif self.layers is not None:
            pass
        else:
            self.layers = ''
        if neurons is not None:
            self.neurons = neurons
        elif self.neurons is not None:
            pass
        else:
            self.neurons = ''
        if accuracy is not None:
            self.accuracy = accuracy
        elif self.accuracy is not None:
            pass
        else:
            self.accuracy = ''
        if number_of_epochs is not None:
            self.number_of_epochs = number_of_epochs
        elif self.number_of_epochs is not None:
            pass
        else:
            self.number_of_epochs = ''

        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        elif self.checkpoint_path is not None:
            pass
        else:
            self.checkpoint_path = ''

        data = [str(self.train_run), str(self.model_name), str(self.pre_process_step), str(self.training_start),
                str(self.training_stop), str(self.data_training_start), str(self.data_training_stop),
                str(self.train_test_split), str(self.layers), str(self.neurons), str(self.accuracy),
                str(self.number_of_epochs), str(checkpoint_path)]
        for x in data:
            print(type(x))
            print(x)
        print(self.model_name)
        print(f'./Metadata/{self.model_name}')
        self.csv_file.create_or_append_csv(f'./Metadata/{self.model_name}', self.header, data)

if __name__ == '__main__':
    a = MetadataLSTM('LSTM_test')
    a.write_data(neurons='00')