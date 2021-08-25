# @Todo: cite https://github.com/jygerardy/causality/blob/master/examples/PCalg.ipynb
from utils import HiddenPrints
import numpy as np
import functools
import networkx as nx
from itertools import combinations, permutations
from collections import defaultdict
import scipy
import scipy.stats
from fcit import fcit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
import os 
from utils import HiddenPrints


def check_dataset(dataset):
    assert isinstance(dataset, np.ndarray), (
        'Dataset must be a 2D numpy array')
    assert dataset.shape[1] >= 2, (
        'Need at leat 2 variables: shape is {}'.format(dataset.shape))


def get_features(dataset, feature_names=None):
    if feature_names:
        len_condition = len(feature_names) == dataset.shape[1]
        assert isinstance(feature_names, list) and len_condition,\
            "number of elements in feature_names\
        and number of features in dataset do not match"
        return {k: v for k, v in enumerate(feature_names)}

    return {k: v for k, v in enumerate(range(dataset.shape[1]))}


def trackcalls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)
    wrapper.has_been_called = False
    return wrapper


class pcalg():
    """
    Peter Spirtes and Clark Glymour ALGorithm

    input :
    dataset =  N*M numpy array where N=sample
            size and M=feature size

    feature_names = dictionary where key = column
                    position, value = column name.
                    if no feature_names provided,
                    key=value=column position

    """
    
    
    def __init__(self, dataset, device, exp_name, feature_names=None, kwargs_m=None, kwargs_c=None):
        self.dataset = check_dataset(dataset)
        self.dataset = dataset
        self.features = get_features(dataset,
                                     feature_names)
        self.G = nx.Graph()
        self.kwargs_m = kwargs_m
        self.kwargs_c = kwargs_c

        self.device = device

        self.exp_name = exp_name
        
    def _instantiate_fully_connected_graph(self):
        self.G.add_nodes_from(self.features.keys())
        for x, y in combinations(self.features.keys(), 2):
            self.G.add_edge(x, y)

            
    def identify_skeleton_original(self, indep_test, alpha=0.05):
        """
        STEP 1 of PC algorighm
        estimate skeleton graph from the data.
        input :
        indep_test = independence function

        alpha = significance level for independence test

        """
        self._instantiate_fully_connected_graph()
        self.d_separators = {} # minimal set
        level = 0
        cont = True
        counter = 1

        while cont:
            #neighbors = {k:list(self.G.neighbors(k)) for k in self.features.keys()}
            print("Level order: {}".format(level))
            cont = False
            # in the stable version,
            # only update neighbors at each level
            for x, y in permutations(self.features.keys(), 2):
                x_neighbors = list(self.G.neighbors(x))
                if y not in x_neighbors:
                    continue
                x_neighbors.remove(y)
                if len(x_neighbors) >= level:
                    cont = True
                    for z in combinations(x_neighbors, level):
                        print("""Starting Independence test between {} and {} conditioned on {}""".format(
                           self.features[x],self.features[y],[self.features[f] for f in z]))
                        with HiddenPrints():
                            pvalue = indep_test(self.dataset[:,x:x+1],
                                                self.dataset[:,y:y+1],
                                                self.dataset[:,z] if len(z) > 1 else None,
                                                transform_marginals=False,
                                                kwargs_m=self.kwargs_m,
                                                kwargs_c=self.kwargs_c,
                                                exp_name='{}_{}_{}'.format(x,y,z),
                                                device=self.device,
                                                disable_tqdm=True)
                        print("""Independence test between {} and {} conditioned on {}: {}""".format(
                           self.features[x],self.features[y],[self.features[f] for f in z], pvalue))
                        print("Test Number: {}".format(counter))
                        counter += 1
                        if pvalue < alpha:
                            print('Removed')
                            self.G.remove_edge(x, y)
                            self.d_separators[(x, y)] = z
                            self.d_separators[(y, x)] = z
                            break                        
            level += 1
        return self    

    
    def orient_graph(self, indep_test, alpha):
        """
        STEP 2 of the PC algorithm: edge orientation
        """
        self.G = self.G.to_directed()

        # STEP 1: IDENTIFYING UNSHIELDED COLLIDERS
        # for each X and Y, only connected through
        # a third variable (e.g. Z in X--Z--Y), test idenpendence
        # between X and Y conditioned upon Z.
        # If conditionally dependent, Z is an unshielded collider.
        # Orient edges to point into Z (X->Z<-Y)
        # Testing for conditional independence is not required when using
        # the stable version of the PCALG(i.e. the SGS variant) because
        # we already have all d-separators of order level 1 between X and Y
        self.colliders = {}
        for x, y in combinations(self.features.keys(), 2):
            x_successors = self.G.successors(x)
            if y in x_successors:
                continue
            y_successors = self.G.successors(y)
            if x in y_successors:
                continue
            intersect = set(x_successors).intersection(set(y_successors))
            for z in intersect:
                if z in self.d_separators[(x, y)]:
                    continue
                if not self.stable:
                    with HiddenPrints():
                        pvalue = indep_test(self.dataset[:,x],
                                            self.dataset[:,y],
                                            self.dataset[:,z] if len(z) > 1 else None,
                                            transform_marginals=False,
                                            kwargs_m=self.kwargs_m,
                                            kwargs_c=self.kwargs_c,
                                            exp_name='pc_{}_{}_{}'.format(x,y,z),
                                            device=self.device,
                                            disable_tqdm=True)
                    if pvalue <= alpha:
                        # x and y are conditionnaly dependent
                        # so z is a collider.
                        self.G.remove_edge(z, x)
                        self.G.remove_edge(z, y)
                        continue
                else:
                    self.G.remove_edge(z, x)
                    self.G.remove_edge(z, y)

        # STEP 2: PREVENT SPURIOUS UNSHIELDED COLLIDERS
        # for each X Y Z such that
        # X->Z--Y
        # and where X and Y are not directly connected,
        # orient the ZY edge to point into Y:
        # X->Z->Y
        # if  X->Z<-Y were true, Z would have been picked up
        # as unshielded collider in STEP 1


        #  STEP 3: PREVENT CYCLES
        # If there is a pair of variables, X and Y connected 
        # both by an undirected edge and by a directed path,
        # starting at X, through one or more other variables to Y,
        # orient the undirected edge as X->Y

    def render_graph(self):
        render = nx.draw_networkx(G=self.G, labels=self.features)
        return render

    def save_class(self):
        return


def resit(X, Y, Z, **kwargs):
    """
    Independently model X and Y as a
    function of Z using models that follow
    the sklearn fit and predict api.

    Predict both X and Y and retrieve residuals.

    Run unconditional independence test between
    the residuals from the X model and the
    residuals from the Y model.

    http://jmlr.org/papers/volume15/peters14a/peters14a.pdf
    """
    X = X.reshape(-1, )
    Y = Y.reshape(-1, )
    sklearn_model = LinearRegression()
    if Z is not None:
        Z = Z.reshape(Z.shape[0], Z.shape[1])
        Z_shape = Z.shape[1]
    else:
        Z_shape = 0
    if Z_shape == 0:
        # unconditional independence test
        return mutual_info_regression(X.reshape(-1, 1), Y)
        
    else:
        poly = PolynomialFeatures(Z_shape)
        Z = poly.fit_transform(Z)
        model_X = sklearn_model
        model_X.fit(Z, X)
        X_hat = model_X.predict(Z)
        
        model_Y = sklearn_model     
        model_Y.fit(Z, Y)
        Y_hat = model_Y.predict(Z)
        return mutual_info_regression((X_hat - X).reshape(-1, 1), Y_hat - Y)

    
def FCIT(X, Y, Z):
    if Z.shape[1] == 0:
        # unconditional independence test
        return fcit.test(X, Y)
    else:
        return fcit.test(X, Y, Z)