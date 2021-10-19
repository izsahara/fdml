import warnings
import numpy as np
import h5py as hdf
from os.path import exists, abspath
from typing import Optional, Union
from .src._utilities import KernelPCA as _KPCA
from .src._kernels import SquaredExponential as _SE
from .src._kernels import Matern52 as _M52
from .src._kernels import Matern32 as _M32

from .parameters import FloatParameter as _FP
from .parameters import VectorParameter as _VP
from .base_models import GPR as _GPR
from .deep_models2 import  GPNode as _GPNode
from .deep_models2 import GPLayer as _GPLayer
from .deep_models2 import SIDGP as _SIDGP

# Type Definitions
TSE    = str(_SE)
TM32   = str(_M32)
TM52   = str(_M52)
TGPR   = str(_GPR)
TSIDGP = str(_SIDGP)
TNone  = type(None)


class KernelPCA(_KPCA):
    def __init__(self, n_components : int, kernel : str = "sigmoid"):
        super(KernelPCA, self).__init__(n_components=n_components, kernel=kernel)

    def transform(self, X : np.ndarray):
        return super(KernelPCA, self).transform(X)


# ============================ IDAES ROUTINES ============================ #
"""
Code Reference:
https://github.com/IDAES/idaes-pse/blob/main/idaes/surrogate/pysmo/sampling.py

Copyright Notice:
Institute for the Design of Advanced Energy Systems Process Systems Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2021, 
by the software owners: The Regents of the University of California, through Lawrence Berkeley National Laboratory, 
National Technology & Engineering Solutions of Sandia, LLC,
Carnegie Mellon University,
West Virginia University Research Corporation, et al. All rights reserved.

Disclaimer:
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, 
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Modifications:
- Commented pandas types

"""
import random
from six import string_types

class FeatureScaling:
    """
    A class for scaling and unscaling input and output data. The class contains three main functions
    """

    def __init__(self):
        pass

    @staticmethod
    def data_scaling_minmax(data):
        """
        This function performs column-wise minimax scaling on the input dataset.
            Args:
                data (NumPy Array or Pandas Dataframe): The input data set to be scaled. Must be a numpy array or dataframe.
            Returns:
                scaled_data(NumPy Array): A 2-D numpy array containing the scaled data. All array values will be between [0, 1].
                data_minimum(NumPy Array): A 2-D row vector containing the column-wise minimums of the input data
                data_maximum(NumPy Array): A 2-D row vector containing the column-wise maximums of the input data
            Raises:
                TypeError: Raised when the input data is not a numpy array or dataframe
        """
        # Confirm that data type is an array or DataFrame
        if isinstance(data, np.ndarray):
            input_data = data
        elif isinstance(data, pd.DataFrame):
            input_data = data.values
        else:
            raise TypeError('original_data_input: Pandas dataframe or numpy array required.')

        if input_data.ndim == 1:
            input_data = input_data.reshape(len(input_data), 1)
        data_minimum = np.min(input_data, axis=0)
        data_maximum = np.max(input_data, axis=0)
        scale = data_maximum - data_minimum
        scale[scale == 0.0] = 1.0
        scaled_data = (input_data - data_minimum)/scale
        # scaled_data = (input_data - data_minimum) / (data_maximum - data_minimum)
        data_minimum = data_minimum.reshape(1, data_minimum.shape[0])
        data_maximum = data_maximum.reshape(1, data_maximum.shape[0])
        return scaled_data, data_minimum, data_maximum

    @staticmethod
    def data_unscaling_minmax(x_scaled, x_min, x_max):
        """
        This function performs column-wise un-scaling on the a minmax-scaled input dataset.
            Args:
                x_scaled(NumPy Array): The input data set to be un-scaled. Data values should be between 0 and 1.
                x_min(NumPy Array): 1-D or 2-D (n-by-1) vector containing the actual minimum value for each column. Must contain same number of elements as the number of columns in x_scaled.
                x_max(NumPy Array): 1-D or 2-D (n-by-1) vector containing the actual maximum value for each column. Must contain same number of elements as the number of columns in x_scaled.
            Returns:
                unscaled_data(NumPy Array): A 2-D numpy array containing the scaled data, unscaled_data = x_min + x_scaled * (x_max - x_min)
            Raises:
                IndexError: Function raises index error when the dimensions of the arrays are inconsistent.
        """
        # Check if it can be evaluated. Will return index error if dimensions are wrong
        if x_scaled.ndim == 1:  # Check if 1D, and convert to 2D if required.
            x_scaled = x_scaled.reshape(len(x_scaled), 1)
        if (x_scaled.shape[1] != x_min.size) or (x_scaled.shape[1] != x_max.size):
            raise IndexError('Dimensionality problems with data for un-scaling.')
        unscaled_data = x_min + x_scaled * (x_max - x_min)
        return unscaled_data

class SamplingMethods:

    @staticmethod
    def nearest_neighbour(full_data, a):
        """
        Function determines the closest point to a in data_input (user provided data).
        This is done by determining the input data with the smallest L2 distance from a.
        The function:
        1. Calculates the L2 distance between all the input data points and a,
        2. Sorts the input data based on the calculated L2-distances, and
        3. Selects the sample point in the first row (after sorting) as the closest sample point.
        Args:
            self: contains, among other things, the input data.
            full_data: refers to the input dataset supplied by the user.
            a: a single row vector containing the sample point we want to find the closest sample to.
        Returns:
            closest_point: a row vector containing the closest point to a in self.x_data
        """

        dist = full_data[:, :-1] - a
        l2_norm = np.sqrt(np.sum((dist ** 2), axis=1))
        l2_norm = l2_norm.reshape(l2_norm.shape[0], 1)
        distances = np.append(full_data, l2_norm, 1)
        sorted_distances = distances[distances[:, -1].argsort()]
        closest_point = sorted_distances[0, :-1]
        return closest_point

    @staticmethod
    def prime_number_generator(n):
        """
        Function generates a list of the first n prime numbers
            Args:
                n(int): Number of prime numbers required
            Returns:
                prime_list(list): A list of the first n prime numbers
        Example: Generate first three prime numbers
            >>  prime_number_generator(3)
            >> [2, 3, 5]
        """
        # Alternative way of generating primes using list generators
        # prime_list = []
        # current_no = 2
        # while len(prime_list) < n:
        #     matching_objs = next((o for o in range(2, current_no) if current_no % o == 0), 0)
        #     if matching_objs==0:
        #         prime_list.append(current_no)
        #     current_no += 1

        prime_list = []
        current_no = 2
        while len(prime_list) < n:
            for i in range(2, current_no):
                if (current_no % i) == 0:
                    break
            else:
                prime_list.append(current_no)
            current_no += 1
        return prime_list

    @staticmethod
    def base_conversion(a, b):
        """
        Function converts integer a from base 10 to base b
            Args:
                a(int): Number to be converted, base 10
                b(int): Base required
            Returns:
                string_representation(list): List containing strings of individual digits of "a" in the new base "b"
        Examples: Convert (i) 5 to base 2 and (ii) 57 to base 47
            >>  base_conversion(5, 2)
            >> ['1', '0', '1']
            >>  base_conversion(57, 47)
            >> ['1', '10']
        """

        string_representation = []
        if a < b:
            string_representation.append(str(a))
        else:
            while a > 0:
                a, c = (a // b, a % b)
                string_representation.append(str(c))
            string_representation = (string_representation[::-1])
        return string_representation

    @staticmethod
    def prime_base_to_decimal(num, base):
        """
        ===============================================================================================================
        Function converts a fractional number "num" in base "base" to base 10. Reverses the process in base_conversion
        Note: The first string element is ignored, since this would be zero for a fractional number.
            Args:
                num(list): Number in base b to be converted. The number must be represented as a list containing individual digits of the base, with the first entry as zero.
                b(int): Original base
            Returns:
                decimal_equivalent(float): Fractional number in base 10
        Examples:
        Convert 0.01 (base 2) to base 10
            >>  prime_base_to_decimal(['0', '0', '1'], 2)  # Represents 0.01 in base 2
            >> 0.25
        Convert 0.01 (base 20) to base 10
            >>  prime_base_to_decimal(['0', '0', '1'], 20)  # Represents 0.01 in base 20
            >> 0.0025
        ================================================================================================================
        """
        binary = num
        decimal_equivalent = 0
        # Convert fractional part decimal equivalent
        for i in range(1, len(binary)):
            decimal_equivalent += int(binary[i]) / (base ** i)
        return decimal_equivalent

    def points_selection(self, full_data, generated_sample_points):
        """
        Uses L2-distance evaluation (implemented in nearest_neighbour) to find closest available points in original data to those generated by the sampling technique.
        Calls the nearest_neighbour function for each row in the input data.
        Args:
            full_data: refers to the input dataset supplied by the user.
            generated_sample_points(NumPy Array): The vector of points (number_of_sample rows) for which the closest points in the original data are to be found. Each row represents a sample point.
        Returns:
            equivalent_points: Array containing the points (in rows) most similar to those in generated_sample_points
        """

        equivalent_points = np.zeros((generated_sample_points.shape[0], generated_sample_points.shape[1] + 1))
        for i in range(0, generated_sample_points.shape[0]):
            closest_point = self.nearest_neighbour(full_data, generated_sample_points[i, :])
            equivalent_points[i, :] = closest_point
        return equivalent_points

    def sample_point_selection(self, full_data, sample_points, sampling_type):
        if sampling_type == 'selection':
            sd = FeatureScaling()
            scaled_data, data_min, data_max = sd.data_scaling_minmax(full_data)
            points_closest_scaled = self.points_selection(scaled_data, sample_points)
            points_closest_unscaled = sd.data_unscaling_minmax(points_closest_scaled, data_min, data_max)

            unique_sample_points = np.unique(points_closest_unscaled, axis=0)
            if unique_sample_points.shape[0] < points_closest_unscaled.shape[0]:
                warnings.warn(
                    'The returned number of samples is less than the requested number due to repetitions during nearest neighbour selection.')
            print('\nNumber of unique samples returned by sampling algorithm:', unique_sample_points.shape[0])

        elif sampling_type == 'creation':
            sd = FeatureScaling()
            unique_sample_points = sd.data_unscaling_minmax(sample_points, full_data[0, :], full_data[1, :])

        return unique_sample_points

    def data_sequencing(self, no_samples, prime_base):
        """
        ===============================================================================================================
        Function which generates the first no_samples elements of the Halton or Hammersley sequence based on the prime number prime_base
        The steps for generating the first no_samples of the sequence are as follows:
        1. Create a list of numbers between 0 and no_samples --- nums = [0, 1, 2, ..., no_samples]
        2. Convert each element in nums into its base form based on the prime number prime_base, reverse the base digits of each number in num
        3. Add a decimal point in front of the reversed number
        4. Convert the reversed numbers back to base 10
            Args:
                no_samples(int): Number of Halton/Hammersley sequence elements required
                prime_base(int): Current prime number to be used as base
            Returns:
                sequence_decimal(NumPy Array): 1-D array containing the first no_samples elements of the sequence based on prime_base
        Examples:
        First three elements of the Halton sequence based on base 2
            >>  data_sequencing(self, 3, 2)
            >> [0, 0.5, 0.75]
        ================================================================================================================
        """
        pure_numbers = np.arange(0, no_samples)
        bitwise_rep = []
        reversed_bitwise_rep = []
        sequence_bitwise = []
        sequence_decimal = np.zeros((no_samples, 1))
        for i in range(0, no_samples):
            base_rep = self.base_conversion(pure_numbers[i], prime_base)
            bitwise_rep.append(base_rep)
            reversed_bitwise_rep.append(base_rep[::-1])
            sequence_bitwise.append(['0.'] + reversed_bitwise_rep[i])
            sequence_decimal[i, 0] = self.prime_base_to_decimal(sequence_bitwise[i], prime_base)
        sequence_decimal = sequence_decimal.reshape(sequence_decimal.shape[0], )
        return sequence_decimal

class LatinHypercubeSampling(SamplingMethods):
    """
    A class that performs Latin Hypercube Sampling. The function returns LHS samples which have been selected randomly after sample space stratification.
    It should be noted that no minimax criterion has been used in this implementation, so the LHS samples selected will not have space-filling properties.
    To use: call class with inputs, and then run ``sample_points`` method.
    **Example:**

    .. code-block:: python

        # To select 10 LHS samples from "data"
        # >>> b = rbf.LatinHypercubeSampling(data, 10, sampling_type="selection")
        # >>> samples = b.sample_points()
    """

    def __init__(self, data_input, number_of_samples=None, sampling_type=None):
        """
        Initialization of **LatinHypercubeSampling** class. Two inputs are required.
        Args:
            data_input (NumPy Array, Pandas Dataframe or list) :  The input data set or range to be sampled.

                - When the aim is to select a set of samples from an existing dataset, the dataset must be a NumPy Array or a Pandas Dataframe and **sampling_type** option must be set to "selection". The output variable (y) is assumed to be supplied in the last column.
                - When the aim is to generate a set of samples from a data range, the dataset must be a list containing two lists of equal lengths which contain the variable bounds and **sampling_type** option must be set to "creation". It is assumed that no range contains no output variable information in this case.
            number_of_samples (int): The number of samples to be generated. Should be a positive integer less than or equal to the number of entries (rows) in **data_input**.
            sampling_type (str) : Option which determines whether the algorithm selects samples from an existing dataset ("selection") or attempts to generate sample from a supplied range ("creation"). Default is "creation".
        Returns:
            **self** function containing the input information
        Raises:
            ValueError: The input data (**data_input**) is the wrong type.
            Exception: When **number_of_samples** is invalid (not an integer, too large, zero, or negative)
        """
        if sampling_type is None:
            sampling_type = 'creation'
            self.sampling_type = sampling_type
            print('Creation-type sampling will be used.')
        elif not isinstance(sampling_type, string_types):
            raise Exception('Invalid sampling type entry. Must be of type <str>.')
        elif (sampling_type.lower() == 'creation') or (sampling_type.lower() == 'selection'):
            sampling_type = sampling_type.lower()
            self.sampling_type = sampling_type
        else:
            raise Exception(
                'Invalid sampling type requirement entered. Enter "creation" for sampling from a range or "selection" for selecting samples from a dataset.')
        print('Sampling type: ', self.sampling_type, '\n')

        if self.sampling_type == 'selection':
            # if isinstance(data_input, pd.DataFrame):
            #     data = data_input.values
            #     data_headers = data_input.columns.values.tolist()
            if isinstance(data_input, np.ndarray):
                data = data_input
                data_headers = []
            else:
                raise ValueError('Pandas dataframe or numpy array required for sampling_type "selection."')
            self.data = data
            self.data_headers = data_headers

            # Catch potential errors in number_of_samples
            if number_of_samples is None:
                print("\nNo entry for number of samples to be generated. The default value of 5 will be used.")
                number_of_samples = 5
            elif number_of_samples > data.shape[0]:
                raise Exception('LHS sample size cannot be greater than number of samples in the input data set')
            elif not isinstance(number_of_samples, int):
                raise Exception('number_of_samples must be an integer.')
            elif number_of_samples <= 0:
                raise Exception('number_of_samples must a positive, non-zero integer.')
            self.number_of_samples = number_of_samples
            self.x_data = self.data[:, :-1]

        elif self.sampling_type == 'creation':
            if not isinstance(data_input, list):
                raise ValueError('List entry of two elements expected for sampling_type "creation."')
            elif len(data_input) != 2:
                raise Exception('data_input must contain two lists of equal lengths.')
            elif not isinstance(data_input[0], list) or not isinstance(data_input[1], list):
                raise Exception('data_input must contain two lists of equal lengths.')
            elif len(data_input[0]) != len(data_input[1]):
                raise Exception('data_input must contain two lists of equal lengths.')
            elif data_input[0] == data_input[1]:
                raise Exception('Invalid entry: both lists are equal.')
            else:
                bounds_array = np.zeros((2, len(data_input[0]),))
                bounds_array[0, :] = np.array(data_input[0])
                bounds_array[1, :] = np.array(data_input[1])
                data_headers = []
            self.data = bounds_array
            self.data_headers = data_headers

            # Catch potential errors in number_of_samples
            if number_of_samples is None:
                print("\nNo entry for number of samples to be generated. The default value of 5 will be used.")
                number_of_samples = 5
            elif not isinstance(number_of_samples, int):
                raise Exception('number_of_samples must be an integer.')
            elif number_of_samples <= 0:
                raise Exception('number_of_samples must a positive, non-zero integer.')
            self.number_of_samples = number_of_samples
            self.x_data = bounds_array  # Only x data will be present in this case

    def variable_sample_creation(self, variable_min, variable_max):
        """
        Function that generates the required number of sample points for a given variable within a specified range using stratification.
        The function divides the variable sample space into self.number_of_samples equal strata and generates a single random sample from each strata based on its lower and upper bound.
        Args:
            self
            variable_min(float): The lower bound of the sample space region. Should be a single number.
            variable_max(float): The upper bound of the sample space region. Should be a single number.
        Returns:
            var_samples(NumPy Array): A numpy array of size (self.number_of_samples x 1) containing the randomly generated points from each strata
        """

        strata_size = 1 / self.number_of_samples
        var_samples = np.zeros((self.number_of_samples, 1))
        for i in range(self.number_of_samples):
            strata_lb = i * strata_size
            sample_point = strata_lb + (random.random() * strata_size)
            var_samples[i, 0] = (sample_point * (variable_max - variable_min)) + variable_min
        return var_samples

    def lhs_points_generation(self):
        """
        Generate points within each strata for each variable based on stratification. When invoked, it:
        1. Determines the mimumum and maximum value for each feature (column),
        2. Calls the variable_sample_creation function on each feature, passing in its mimmum and maximum
        3. Returns an array containing the points selected in each strata of each column
        Returns:
            sample_points_vector(NumPy Array): Array containing the columns of the random samples generated in each strata.
        """

        ns, nf = np.shape(self.x_data)
        sample_points_vector = np.zeros(
            (self.number_of_samples, nf))  # Array containing points in each interval for each variable
        for i in range(nf):
            variable_min = 0  # np.min(self.x_data[:, i])
            variable_max = 1  # np.max(self.x_data[:, i])
            var_samples = self.variable_sample_creation(variable_min, variable_max)  # Data generation step
            sample_points_vector[:, i] = var_samples[:, 0]
        return sample_points_vector

    @staticmethod
    def random_shuffling(vector_of_points):
        """
        This function carries out random shuffling of column data to generate samples.
        Data in each of the columns  in the input array is shuffled separately, meaning that the rows of the resultant array will contain random samples from the sample space.
        Args:
            vector_of_points(NumPy Array): Array containing ordered points generated from stratification. Should usually be the output of the lhs_points_generation function. Each column self.number_of_samples elements.
        Returns:
            vector_of_points(NumPy Array): 2-D array containing the shuffled data. Should contain number_of_sample rows, with each row representing a potential random sample from within the sample space.
        """

        _, nf = np.shape(vector_of_points)
        for i in range(0, nf):
            z_col = vector_of_points[:, i]
            np.random.shuffle(z_col)
            vector_of_points[:, i] = z_col
        return vector_of_points

    def sample_points(self):
        """
        ``sample_points`` generates or selects Latin Hypercube samples from an input dataset or data range. When called, it:
            1. generates samples points from stratified regions by calling the ``lhs_points_generation``,
            2. generates potential sample points by random shuffling, and
            3. when a dataset is provided, selects the closest available samples to the theoretical sample points from within the input data.
        Returns:
            NumPy Array or Pandas Dataframe:     A numpy array or Pandas dataframe containing **number_of_samples** points selected or generated by LHS.
        """

        vector_of_points = self.lhs_points_generation()  # Assumes [X, Y] data is supplied.
        generated_sample_points = self.random_shuffling(vector_of_points)
        unique_sample_points = self.sample_point_selection(self.data, generated_sample_points, self.sampling_type)

        # if len(self.data_headers) > 0:
        #     unique_sample_points = pd.DataFrame(unique_sample_points, columns=self.data_headers)
        return unique_sample_points


# ============================ IDAES ROUTINES ============================ #













