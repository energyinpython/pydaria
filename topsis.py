import numpy as np

from normalizations import minmax_normalization
from mcda_method import MCDA_method

class TOPSIS(MCDA_method):
    def __init__(self, normalization_method = minmax_normalization):
        """
        Create the TOPSIS method object and select normalization method `normalization_method` and
        distance metric `distance metric`.

        Parameters
        -----------
            normalization_method : function
                method for decision matrix normalization chosen from `normalizations`

        """
        self.normalization_method = normalization_method


    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix `matrix` with m alternatives in rows and 
        n criteria in columns using criteria `weights` and criteria `types`.

        Parameters
        ----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Vector with criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Vector with criteria types. Profit criteria are represented by 1 and cost by -1.

        Returns
        -------
            ndrarray
                Vector with preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ---------
        >>> topsis = TOPSIS(normalization_method = minmax_normalization)
        >>> pref = topsis(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        TOPSIS._verify_input_data(matrix, weights, types)
        return TOPSIS._topsis(matrix, weights, types, self.normalization_method)


    @staticmethod
    def _topsis(matrix, weights, types, normalization_method):
        # Normalize matrix using chosen normalization (for example linear normalization)
        norm_matrix = normalization_method(matrix, types)

        # Multiply all rows of normalized matrix by weights
        weighted_matrix = norm_matrix * weights

        # Calculate vectors of PIS (ideal solution) and NIS (anti-ideal solution)
        pis = np.max(weighted_matrix, axis=0)
        nis = np.min(weighted_matrix, axis=0)

        # Calculate distance of every alternative from PIS and NIS using Euclidean distance metric
        Dp = np.sqrt(np.sum(np.square(weighted_matrix - pis), axis = 1))
        Dm = np.sqrt(np.sum(np.square(weighted_matrix - nis), axis = 1))

        C = Dm / (Dm + Dp)
        return C