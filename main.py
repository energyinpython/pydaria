import os
import copy
import numpy as np
import pandas as pd

from topsis import TOPSIS
from weighting_methods import critic_weighting
from normalizations import minmax_normalization
from additions import rank_preferences

from daria import DARIA


def main():

    # Temporal alternatives assessment using the DARIA-TOPSIS method based on the Gini coefficient 
    # variability measure.
    # Load the name of the folder with CSV files, including data.
    path = 'data'
    # Create the list with years to be analyzed that are elements of data files names.
    str_years = [str(y) for y in range(2015, 2020)]
    # Create a list with latex symbols of evaluated alternatives.
    list_alt_names = [r'$A_{' + str(i) + '}$' for i in range(1, 26 + 1)]
    
    # Create dataframes for TOPSIS preferences and rankings for each evaluated year.
    preferences = pd.DataFrame(index = list_alt_names)
    rankings = pd.DataFrame(index = list_alt_names)
    
    # Evaluate alternatives with the TOPSIS method for each year.
    for el, year in enumerate(str_years):
        # Load data from a CSV file for a given year.
        file = 'data_' + str(year) + '.csv'
        pathfile = os.path.join(path, file)
        data = pd.read_csv(pathfile, index_col = 'Country')
        
        # Create a dataframe with a decision matrix.
        df_data = data.iloc[:len(data) - 1, :]
        df_data = df_data.dropna()

        # Create a vector with criteria types.
        types = data.iloc[len(data) - 1, :].to_numpy()
        
        #list_of_cols = list(df_data.columns)
        matrix = df_data.to_numpy()
        # Calculate criteria weights using the CRITIC weighting method.
        weights = critic_weighting(matrix)

        # Initialize the TOPSIS method object.
        topsis = TOPSIS(normalization_method=minmax_normalization)
        # Calculate the TOPSIS preferences.
        pref = topsis(matrix, weights, types)
        # Generate the TOPSIS ranking based on calculated preferences.
        rank = rank_preferences(pref, reverse = True)
        # Save the results in dataframes.
        preferences[year] = pref
        rankings[year] = rank

    preferences = preferences.rename_axis('Ai')
    preferences.to_csv('results/preferences.csv')

    rankings = rankings.rename_axis('Ai')
    rankings.to_csv('results/rankings.csv')
    
    # DARIA method
    # dataframe `preferences` includes preferences of alternatives for evaluated years
    df_varia_fin = pd.DataFrame(index = list_alt_names)
    # Create a matrix with preference values for each year
    # and transpose it to have years in rows and alternatives in columns
    df = preferences.T
    matrix = df.to_numpy()

    # the TOPSIS method orders preferences in descending order
    met = 'topsis'

    # Calculate efficiencies variability using DARIA methodology
    # Initialize the DARIA method object
    daria = DARIA()

    # Calculate the variability of TOPSIS preferences in all years using the Gini coefficient.
    # You can also choose another variability measure such as _entropy, _std, _stat_var, and _coeff_var
    # from daria class
    var = daria._gini(matrix)
    # Calculate variability directions
    dir_list, dir_class = daria._direction(matrix)

    # variability of preference values
    df_varia_fin[met.upper() + ' var'] = list(var)
    # directions of preferences variability
    df_varia_fin[met.upper() + ' dir'] = list(dir_class)

    df_varia_fin = df_varia_fin.rename_axis('Ai')
    df_varia_fin.to_csv('results/variability.csv')

    df_final_results = pd.DataFrame(index = list_alt_names)
    
    # S = preferences['2019'].to_numpy()
    S = matrix[-1, :]
    G = copy.deepcopy(var)
    dir = copy.deepcopy(dir_class)

    # Update efficiencies using DARIA methodology.
    # final updated preferences
    final_S = daria._update_efficiency(S, G, dir)

    # The TOPSIS ranking is prepared in descending order according to prefs.
    rank = rank_preferences(final_S, reverse = True)
    
    # Save aggregated final preference values and rankings.
    df_final_results[met.upper() + ' pref'] = final_S
    df_final_results[met.upper() + ' rank'] = rank
    df_final_results = df_final_results.rename_axis('Ai')
    df_final_results.to_csv('results/final_results.csv')
    

if __name__ == '__main__':
    main()