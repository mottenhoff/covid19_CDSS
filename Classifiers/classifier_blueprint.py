'''
DO NOT OVERWRITE THIS FILE, MAKE A COPY TO EDIT!

This file contains blueprint for the model that is used
in covid19_ICU_admission.
This class should take care of training a classifier/regressor,
scoring each iteration and evaluating post training.

Make sure the names, inputs and outputs of the predefined methods stay
the same!


Model parameters:
Please store all parameters in the defined dicts in __init__().
This way it is easy to change some parameters without changing
the main code and without the need for extra config files.

Most important naming conventions and variables:
Model:      The whole class in this file. This means that "model" takes
            care of training, scoring and evaluating

Clf:        This is the actually classifier/regressor. It can be for example
            the trained instance from the Sklearn package
            (e.g. LogisticRegression())

Datasets:   dictionary containin all training and test sets:
            train_x, train_y, test_x, test_y, test_y_hat

..._args:   Dictionary that holds the parameters that are used as
            input for train, score or evaluate
'''


class YourClass:

    def __init__(self):
        ''' Initialize model.
        Save all model parameters here.
        Please don't change the names of the preset
        parameters, as they might be called outside
        this class.
        '''

        self.goal = None            # Will be overwritten
        self.model_args = {}
        self.score_args = {}
        self.evaluation_args = {}

        # Add any variables you want to keep track
        # off here

    def train(self, datasets):
        ''' Initialize, train and predict a classifier.
        This includes: Feature engineering (i.e. PCA) and
        selection, training clf, (hyper)parameter optimization,
        and a prediction on the test set. Make sure to save
        all variables you want to keep track of in the instance.

        Input:
            datasets:: dict
                Contains train and test x, y

        Output:
            clf:: instance, dict, list, None
                Trained classifier/regressor instance, such as
                sklearn logistic regression. Is not used
                outside this file, so can be left empty
            datasets:: dict
                Dictionary containing the UPDATED train and test
                sets. Any new features should be present in this
                dict
            test_y_hat:: list
                List containing the probabilities of outcomes.
        '''
        clf = None
        test_y_hat = None
        return clf, datasets, test_y_hat

    def score(self, clf, datasets, test_y_hat, rep):
        ''' Scores the individual prediction per outcome.
        NOTE: Be careful with making plots within this
        function, as this function can be called mutliple
        times. You can use rep as control

        Input:
            clf:: instance, dict, list, None
                Trained classifier/regressor from self.train()
            datasets:: dict
                Dictionary containing the datasets used for
                training
            test_y_hat:: list
                List containing probabilities of outcomes.

        Output:
            score:: int, float, list
                Calculated score of test_y_hat prediction.
                Can be a list of scores.
        '''

        score = None
        return score

    def evaluate(self, clf, datasets, scores):
        ''' Evaluate the results of the modelling process,
        such as, feature importances.
        NOTE: No need to show plots here, plt.show is called right
        after this function returns

        Input:
            clf:: instance, dict, list, None
                Trained classifier/regressor from self.train()
            datasets:: dict
                Dictionary containing the datasets used for
                training
            scores:: list
                List of all scores generated per training
                iteration.
        '''

        pass
