Welcome to pydaria documentation!
========================================

pydaria is The Data Variability Based Multi-Criteria Assessment Method based on MCDA.
This library includes:

- DARIA class:

	- Data variability measures:
	
		- ``gini`` (Gini coefficient)
		- ``entropy`` (Entropy)
		- ``std`` (Standard deviation)
		- ``stat_var`` (Statistical variance)
		- ``coeff_var`` (Coefficient of variation)
		
	- Variability direction determination ``direction``
	
	- Calculation of the final overall alternatives efficiency values ``update_efficiency``

- MCDA method ``TOPSIS``
	
- Correlation coefficients:

	- ``spearman`` (Spearman rank correlation coefficient)
	- ``weighted_spearman`` (Weighted Spearman rank correlation coefficient)
	- ``pearson_coeff`` (Pearson correlation coefficient)
	- ``WS_coeff`` (Similarity rank coefficient - WS coefficient)
	
- Methods for normalization of decision matrix:

	- ``linear_normalization`` (Linear normalization)
	- ``minmax_normalization`` (Minimum-Maximum normalization)
	- ``max_normalization`` (Maximum normalization)
	- ``sum_normalization`` (Sum normalization)
	- ``vector_normalization`` (Vector normalization)
	
- Methods for determination of criteria weights (weighting methods):

	- ``equal_weighting`` (Equal weighting method)
	- ``entropy_weighting`` (Entropy weighting method)
	- ``std_weighting`` (Standard Deviation weighting method)
	- ``critic_weighting`` (CRITIC weighting method)
	
- additions:

	- ``rank_preferences`` (Method for ordering alternatives according to their preference values obtained with MCDA methods)
	
Check out the :doc:`usage` section for further information.

.. note::

   This project is under active development.

Contents
---------

.. toctree::
	:maxdepth: 2
	
	usage
	example
