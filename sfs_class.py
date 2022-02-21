import numpy as np
from sklearn.model_selection import KFold
import math
import tqdm
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import normalize
import random
from numpy.linalg import norm
import time
import os

def reset_seeds(reset_graph_with_backend=None):
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print("KERAS AND TENSORFLOW GRAPHS RESET")  # optional

    np.random.seed(1)
    random.seed(2)
    tf.compat.v1.set_random_seed(3)

class SFS_hub(object):
    def __init__(self, rhythmic_scores=None, error_threshold=60):
        self.error_threshold = error_threshold
        self.rhythmic_scores = rhythmic_scores
        # colours in this case = phase bins, but they previously represented clusters
        self.colours = np.unique(self.rhythmic_scores['phase'])

        # iteration counters
        self.current_genes = 0
        self.counter = 0

        # gene lists
        self.genes_perm = []
        self.all_past_genes = []
        i_genes = None

        # define phase counts
        self.counts = {}
        for p in range(self.colours.shape[0]):
            self.counts[str(self.colours[p])] = 0

        if not os.path.exists('Results'):
            os.mkdir('Results')
        self.exp_name = time.time()
        self.folds = KFold(n_splits=6, shuffle=True, random_state=0)

        self.results_record = {'idx': [], 'genes': [], 'train_error': [], 'train_preds': [], 'test_error': [], 'test_preds': []}
        self.results_iteration = None
        self.results_remove = None

        # scoring
        self.base_score = 0


        self.early_stop = EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss', mode='min')


    def custom_loss(self, y_true, y_pred):
        return tf.reduce_mean((tf.math.acos(tf.matmul(y_true, tf.transpose(y_pred)) / (
                    (tf.norm(y_true) * tf.norm(y_pred)) + tf.keras.backend.epsilon()))))

    def cyclical_loss(self, y_true, y_pred):
        error = 0
        for i in range(y_pred.shape[0]):
            error += np.arccos((y_true[i, :] @ y_pred[i, :]) / (norm(y_true[i, :]) * norm(y_pred[i, :]) + 1e-8))
        return error

    def larger_model(self):
        # lr = 0.00001
        adam = Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.999, amsgrad=False)

        # create model

        model = Sequential()
        model.add(Dense(32, kernel_initializer='normal', activation='relu'))
        model.add(Dense(128, kernel_initializer='normal', activation='relu'))
        model.add(Dense(512, kernel_initializer='normal', activation='relu'))
        model.add(Dense(2, kernel_initializer='normal'))
        # Compile model
        model.compile(loss=self.custom_loss, optimizer=adam)
        return model

    def phase_selection(self, count_num, used_genes):
        counts = self.counts

        if count_num >= 1:
            print(used_genes, 'used')

            used_genes = self.rhythmic_scores.loc[self.genes_perm]['phase']
            # make sure genes aren't repeated
            used_genes = used_genes.loc[[i for i in used_genes.index if i not in self.genes_perm]]

            for j in used_genes:
                counts[str(j)] += 1

            min_val = min(counts.values())
            min_counts = [k for k, v in counts.items() if v == min_val]
            random.seed()
            colour = random.choice(min_counts)


            genes = self.rhythmic_scores.loc[self.rhythmic_scores['phase'] == int(colour)]
            idx = genes.index

            return idx

        if count_num == 0:
            np.random.seed()
            i = np.random.randint(0, self.colours.shape[0])
            colour = self.colours[i]
            genes = self.rhythmic_scores.loc[self.rhythmic_scores['phase'] == colour]
            idx = genes.index.values

            return idx


    def run_model(self, i_gene, X_data, Y_data, X_test, Y_test, type=None):
        X_d = X_data[i_gene].values
        X_t = X_test[i_gene].values

        error = 0  # Initialise error
        all_preds = np.zeros((Y_data.shape[0], 2))  # Create empty array
        all_test_preds = []

        for n_fold, (train_idx, valid_idx) in enumerate(self.folds.split(X_data, Y_data)):
            X_train, Y_train = X_d[train_idx], Y_data[train_idx]  # Define training data for this iteration
            X_valid, Y_valid = X_d[valid_idx], Y_data[valid_idx]

            reset_seeds()
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model = self.larger_model()
            # batch size = 2
            model.fit(X_train.astype('float32'), Y_train.astype('float32'),
                      validation_data=(X_valid.astype('float32'), Y_valid.astype('float32')),
                      batch_size=8, epochs=200, callbacks=[self.early_stop],
                      verbose=0)  # Fit the model on the training data
            preds = normalize(model(X_valid.astype('float32')))  # Predict on the validation data
            all_preds[valid_idx] = normalize(model(X_valid.astype('float32')))
            all_test_preds.append(normalize(model(X_t.astype('float32'))))
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model = None
            del model

            error += self.cyclical_loss(Y_valid.astype('float32'), preds.astype('float32'))  # Evaluate the predictions
        # forward stage
        if type == None:
            self.results_iteration['train_error'].append(
                    60 * 12 * self.cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')) / (
                            Y_data.shape[0] * np.pi))
            self.results_iteration['test_error'].append(
                    60 * 12 * self.cyclical_loss(Y_test.astype('float64'), np.mean(all_test_preds, axis=0).astype('float64')) / (
                            Y_test.shape[0] * np.pi))
        # reverse stage
        else:
            self.results_remove['train_error'].append(
                60 * 12 * self.cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')) / (
                        Y_data.shape[0] * np.pi))
            self.results_remove['test_error'].append(
                60 * 12 * self.cyclical_loss(Y_test.astype('float64'),
                                             np.mean(all_test_preds, axis=0).astype('float64')) / (
                        Y_test.shape[0] * np.pi))


    def sfs_iterator(self, X_data, Y_data, N_GENES, X_test=None, Y_test=None, main_gene=None):
        os.mkdir('Results/{}'.format(self.exp_name))
        used_genes = None
        # start with seed gene first
        if self.counter == 0:
            self.genes_perm = [[main_gene]]
            self.all_past_genes.append([main_gene])
        remove = False

        #
        while self.current_genes < N_GENES: #if number of genes less than max

            self.results_iteration = {k : [] for k,v in self.results_record.items()}


            self.i_genes = self.phase_selection(self.counter,  used_genes)

            self.current_genes += 1

            # test adding each gene from the current phase bin
            for j in tqdm.tqdm(range(self.i_genes.shape[0])):

                i_gene = self.i_genes[j]

                if self.counter >= 1:
                    i_gene = np.concatenate((np.array(self.genes_perm).reshape(-1), np.array([i_gene])))
                if self.counter == 0:
                    i_gene = [main_gene, i_gene]

                self.results_iteration['idx'].append(i_gene)
                self.results_iteration['genes'].append(i_gene)


                self.run_model(i_gene, X_data, Y_data, X_test, Y_test)



            # print(self.results_iteration['idx'][self.results_iteration['train_error'].index(min(self.results_iteration['train_error']))])
            # this selects the gene set with the lowest error
            self.genes_perm = self.results_iteration['idx'][self.results_iteration['train_error'].index(min(self.results_iteration['train_error']))]

            # this updates the current error score
            self.base_score = self.results_iteration['train_error'][self.results_iteration['train_error'].index(min(self.results_iteration['train_error']))]


            self.counter += 1

            if len(self.genes_perm) != self.current_genes:
                self.current_genes = len(self.genes_perm)
            print(len(self.genes_perm))

            remove = True
            print(self.base_score)
            remove_count = 0

            # Test removing one gene at a time EXCEPT the gene that was just added
            while remove == True:
                if self.current_genes > 5:


                    # len(self.genes_perm) makes sure the previously added gene isn't removed
                    for m in range(0, len(self.genes_perm)-1):
                        self.results_remove = {k: [] for k, v in self.results_record.items()}

                        gene_remove = self.genes_perm.copy()
                        gene_remove = np.delete(gene_remove, m)
                        remove_count += 1

                        self.results_remove['idx'].append(gene_remove)
                        self.results_iteration['genes'].append(gene_remove)
                        # run model each time with a gene removed
                        self.run_model(gene_remove, X_data, Y_data, X_test, Y_test, type='remove')
                        print(self.base_score, self.results_remove['train_error'])

                        # if a new result is better than the baseline error - let the gene be removed
                        if self.results_remove['train_error'] < self.base_score:
                            self.base_score = self.results_remove['train_error']

                            self.genes_perm = gene_remove
                            self.current_genes -= 1
                            # if a gene is removed, the loop restarts
                            break


                        if remove_count >= len(self.genes_perm):
                            remove = False

                else:
                    remove = False
            import pickle
            print(self.genes_perm)
            # if under a minimum value, dump gene list into a pickle folder
            if np.min(self.base_score) < 120:
                with open('Results/{}/{}_{}SFSGenes.p'.format(self.exp_name, len(self.genes_perm), self.base_score), 'wb') as handle:
                    pickle.dump(self.genes_perm, handle)





