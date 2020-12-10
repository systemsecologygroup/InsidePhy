import unittest
import pkg_resources
import pandas as pd


class UnitTestCases(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_data_integrity(self):
        data_path = pkg_resources.resource_filename('insidephy.data', 'maranon_2013EcoLet_data.h5')
        allometries = pd.read_hdf(data_path, 'allodtf')
        cultures = pd.read_hdf(data_path, 'batchdtf')
        # TODO: Complete unit tests


if __name__ == '__main__':
    unittest.main()
