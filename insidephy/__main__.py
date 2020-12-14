import unittest
import pkg_resources
import pandas as pd
from insidephy.size_based_models.SBMi import SBMi
from insidephy.size_based_models.SBMc import SBMc


class UnitTestCases(unittest.TestCase):
    def setUp(self):
        data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
        self.allometries = pd.read_hdf(data_path, 'allodtf')
        self.cultures = pd.read_hdf(data_path, 'batchdtf')
        self.sizedtf = pd.read_hdf(data_path, 'sizedtf')

    def test_allometries_dtf_shape(self):
        self.assertEqual(self.allometries.shape, (22, 17))

    def test_cultures_dtf_shape(self):
        self.assertEqual(self.cultures.shape, (263, 10))

    def test_size_dtf_shape(self):
        self.assertEqual(self.sizedtf.shape, (94, 6))

    def test_sbmc_instance(self):
        sbmc = SBMc(0.002, [1e6], ['Aa'], [10], [100],  [10], 0.0, 1.0)
        self.assertIsInstance(sbmc, SBMc)

    def test_sbmi_instance(self):
        sbmi = SBMi(0.002, [1e6], ['Aa'], [10], [100], [100], 50, 500, 0.0, 1.0)
        self.assertIsInstance(sbmi, SBMi)

    def test_sbmc_type_exception(self):
        with self.assertRaises(TypeError) as cm:
            SBMc(0.002, [1e6], ['Aa'], 10, [100], [10], 0.0, 1.0)

    def test_sbmi_type_exception(self):
        with self.assertRaises(TypeError) as cm:
            SBMi(0.002, [1e6], ['Aa'], 10, [100], [10], 50, 500, 0.0, 1.0)

    def test_sbmc_value_exception(self):
        with self.assertRaises(ValueError) as cm:
            SBMc(0.002, [1e6], ['Aa', 'Bb'], [10], [100], [10], 0.0, 1.0)

    def test_sbmi_value_exception(self):
        with self.assertRaises(ValueError) as cm:
            SBMi(0.002, [1e6], ['Aa', 'Bb'], [10], [100], [10], 50, 500, 0.0, 1.0)


if __name__ == '__main__':
    unittest.main()
