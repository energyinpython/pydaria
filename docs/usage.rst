Usage
=======

Usage examples
----------------------

The DARIA method
_______________________

.. code-block:: python
	
	import numpy as np
	import pandas as pd

	# Initialize the DARIA method object
	daria = DARIA()

	# Calculate the variability of TOPSIS preferences in all years using the Gini coefficient.
	# provide ``matrix`` including TOPSIS preference values for each year in particular rows for alternatives in columns.
	G = daria._gini(matrix)

	# Calculate variability directions
	dir_list, dir = daria._direction(matrix)

	# Update efficiencies using DARIA methodology.
	# Provide vector `S` containing preference values from the most recent evaluated period
	S = matrix[-1, :]
	# final updated preferences
	final_S = daria._update_efficiency(S, G, dir)

	# The TOPSIS ranking is prepared in descending order according to prefs.
	final_rank = rank_preferences(final_S, reverse = True)


The TOPSIS method
_______________________

The TOPSIS method is used to calculate the preference of evaluated alternatives. When creating the object of the TOPSIS method, you have to provide
``normalization_method`` (it is ``minmax_normalization`` by default). The TOPSIS method requires providing 
the decision matrix ``matrix``, vector with criteria weights ``weights``, and vector with criteria types ``types``. The TOPSIS method returns a vector with 
preference values ``pref``. To generate the TOPSIS ranking of alternatives, ``pref`` has to be sorted in descending order. The ranking is generated by ``rank_preferences``, providing
``pref`` as argument and setting parameter ``reverse`` as ``True`` because we need to sort preferences descendingly.

.. code-block:: python

	import numpy as np
	from topsis import TOPSIS
	from normalizations import minmax_normalization
	from additions import rank_preferences

	# provide decision matrix in array numpy.darray
	matrix = np.array([[256, 8, 41, 1.6, 1.77, 7347.16],
	[256, 8, 32, 1.0, 1.8, 6919.99],
	[256, 8, 53, 1.6, 1.9, 8400],
	[256, 8, 41, 1.0, 1.75, 6808.9],
	[512, 8, 35, 1.6, 1.7, 8479.99],
	[256, 4, 35, 1.6, 1.7, 7499.99]])

	# provide criteria weights in array numpy.darray. All weights must sum to 1.
	weights = np.array([0.405, 0.221, 0.134, 0.199, 0.007, 0.034])

	# provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
	types = np.array([1, 1, 1, 1, -1, -1])

	# Create the TOPSIS method object providing normalization method.
	topsis = TOPSIS(normalization_method = norms.minmax_normalization)

	# Calculate the TOPSIS preference values of alternatives
	pref = topsis(matrix, weights, types)

	# Generate ranking of alternatives by sorting alternatives descendingly according to the TOPSIS algorithm (reverse = True means sorting in descending order) according to preference values
	rank = rank_preferences(pref, reverse = True)

	print('Preference values: ', np.round(pref, 4))
	print('Ranking: ', rank)
	
Output

.. code-block:: console

	Preference values:  [0.4242 0.3217 0.4453 0.3353 0.8076 0.2971]
	Ranking:  [3 5 2 4 1 6]

	

Correlation coefficients
__________________________

Spearman correlation coefficient

This method is used to calculate correlation between two different rankings. It requires two vectors ``R`` and ``Q`` with rankings of the same size. It returns value
of correlation.

.. code-block:: python

	import numpy as np
	from correlations import spearman

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using ``spearman`` coefficient
	coeff = spearman(R, Q)
	print('Spearman coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Spearman coeff:  0.9

	
	
Weighted Spearman correlation coefficient

This method is used to calculate correlation between two different rankings. It requires two vectors ``R`` and ``Q`` with rankings of the same size. It returns value
of correlation.

.. code-block:: python

	import numpy as np
	from correlations import weighted_spearman

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using ``weighted_spearman`` coefficient
	coeff = weighted_spearman(R, Q)
	print('Weighted Spearman coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Weighted Spearman coeff:  0.8833
	
	
	
Similarity rank coefficient WS

This method is used to calculate similarity between two different rankings. It requires two vectors ``R`` and ``Q`` with rankings of the same size. It returns value
of similarity.

.. code-block:: python

	import numpy as np
	from correlations import WS_coeff

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the similarity using ``WS_coeff`` coefficient
	coeff = WS_coeff(R, Q)
	print('WS coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	WS coeff:  0.8542

	
	
Pearson correlation coefficient

This method is used to calculate correlation between two different rankings. It requires two vectors ``R`` and ``Q`` with rankings of the same size. It returns value
of correlation.

.. code-block:: python

	import numpy as np
	from correlations import pearson_coeff

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using ``pearson_coeff`` coefficient
	coeff = pearson_coeff(R, Q)
	print('Pearson coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Pearson coeff:  0.9
	
	
	
Methods for criteria weights determination
___________________________________________

Entropy weighting method

This method is used to calculate criteria weights based on alternatives perfromance values provided in decision matrix. This method requires
providing two-dimensional decision matrix ``matrix`` with perfromance values of alternatives in rows considering criteria in columns. It returns
vector with criteria weights. All values in vector ``weights`` must sum to 1.
		
.. code-block:: python

	import numpy as np
	from weighting_methods import entropy_weighting

	matrix = np.array([[30, 30, 38, 29],
	[19, 54, 86, 29],
	[19, 15, 85, 28.9],
	[68, 70, 60, 29]])

	weights = entropy_weighting(matrix)

	print('Entropy weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	Entropy weights:  [0.463  0.3992 0.1378 0.    ]
	

CRITIC weighting method

This method is used to calculate criteria weights based on alternatives perfromance values provided in decision matrix. This method requires
providing two-dimensional decision matrix ``matrix`` with perfromance values of alternatives in rows considering criteria in columns. It returns
vector with criteria weights. All values in vector ``weights`` must sum to 1.
		
.. code-block:: python

	import numpy as np
	from weighting_methods import critic_weighting

	matrix = np.array([[5000, 3, 3, 4, 3, 2],
	[680, 5, 3, 2, 2, 1],
	[2000, 3, 2, 3, 4, 3],
	[600, 4, 3, 1, 2, 2],
	[800, 2, 4, 3, 3, 4]])

	weights = critic_weighting(matrix)

	print('CRITIC weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	CRITIC weights:  [0.157  0.2495 0.1677 0.1211 0.1541 0.1506]


Standard deviation weighting method

This method is used to calculate criteria weights based on alternatives perfromance values provided in decision matrix. This method requires
providing two-dimensional decision matrix ``matrix`` with perfromance values of alternatives in rows considering criteria in columns. It returns
vector with criteria weights. All values in vector ``weights`` must sum to 1.
		
.. code-block:: python

	import numpy as np
	from weighting_methods import std_weighting

	matrix = np.array([[0.619, 0.449, 0.447],
	[0.862, 0.466, 0.006],
	[0.458, 0.698, 0.771],
	[0.777, 0.631, 0.491],
	[0.567, 0.992, 0.968]])

	weights = std_weighting(matrix)

	print('Standard deviation weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	Standard deviation weights:  [0.2173 0.2945 0.4882]
	
	
Normalization methods
______________________

Here is an example of vector normalization usage. Other normalizations provided in ``normalizations``, namely ``minmax_normalization``, ``max_normalization``,
``sum_normalization``, ``linear_normalization`` are used in analogous way.


Vector normalization

This method is used to normalize decision matrix ``matrix``. It requires providing decision matrix ``matrix`` with performance values of alternatives in rows
considering criteria in columns and vector with criteria types ``types``. This method returns normalized matrix.

.. code-block:: python
	
	import numpy as np
	from normalizations import vector_normalization

	matrix = np.array([[8, 7, 2, 1],
	[5, 3, 7, 5],
	[7, 5, 6, 4],
	[9, 9, 7, 3],
	[11, 10, 3, 7],
	[6, 9, 5, 4]])

	types = np.array([1, 1, 1, 1])

	norm_matrix = vector_normalization(matrix, types)
	print('Normalized matrix: ', np.round(norm_matrix, 4))
	
Output

.. code-block:: console
	
	Normalized matrix:  [[0.4126 0.3769 0.1525 0.0928]
	[0.2579 0.1615 0.5337 0.4642]
	[0.361  0.2692 0.4575 0.3714]
	[0.4641 0.4845 0.5337 0.2785]
	[0.5673 0.5384 0.2287 0.6499]
	[0.3094 0.4845 0.3812 0.3714]]

	
