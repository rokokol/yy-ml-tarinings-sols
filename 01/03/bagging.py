import numpy as np


class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            indices = np.random.choice(data_length, data_length, replace=True)
            self.indices_list.append(indices)

    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(
            data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            indices = self.indices_list[bag]
            data_bag, target_bag = data[indices], target[indices]
            self.models_list.append(model.fit(data_bag, target_bag))
        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        predictions = np.array([model.predict(data) for model in self.models_list])
        return np.mean(predictions, axis=0)

    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        for bag in range(self.num_bags):
            model = self.models_list[bag]
            mask = np.ones(len(self.data), dtype=bool)
            mask[self.indices_list[bag]] = False
            oob_indices = np.where(mask)[0]
            for idx in oob_indices:
                pred = model.predict(self.data[idx].reshape(1, -1))[0]
                list_of_predictions_lists[idx].append(pred)

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)

    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        if not hasattr(self, 'list_of_predictions_lists'):
            raise AttributeError(
                "list_of_predictions_lists is not generated. Call '_get_oob_predictions_from_every_model()' first.")
        self.oob_predictions = np.array(
            [np.mean(preds) if len(preds) > 0 else None for preds in self.list_of_predictions_lists])

    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        if not hasattr(self, 'oob_predictions'):
            raise AttributeError("oob_predictions is not generated. Call '_get_averaged_oob_predictions()' first.")
        valid_indices = [i for i, pred in enumerate(self.oob_predictions) if pred is not None]
        valid_oob_predictions = self.oob_predictions[valid_indices]
        valid_targets = self.target[valid_indices]
        if len(valid_oob_predictions) > 0:
            mse = np.mean((valid_targets - valid_oob_predictions) ** 2)
            return mse

        else:
            return None
