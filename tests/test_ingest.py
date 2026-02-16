import unittest
import pandas as pd
import numpy as np
from src.ingest.mnrega import merge_datasets

class TestIngest(unittest.TestCase):
    def test_merge_datasets(self):
        # Create Dummy Agri Data
        agri_data = pd.DataFrame({
            'State_Name': ['KARNATAKA', 'KARNATAKA'],
            'District_Name': ['MYSORE', 'MANDYA'],
            'Crop': ['Rice', 'Sugarcane']
        })
        
        # Create Dummy MNREGA Data
        mnrega_data = pd.DataFrame({
            'State_Name': ['KARNATAKA', 'TAMIL NADU'],
            'District_Name': ['MYSORE', 'CHENNAI'],
            'MNREGA_JobCards_Issued': [100, 200],
            'MNREGA_Active_Workers': [50, 150]
        })
        
        # Test Merge
        merged = merge_datasets(mnrega_data, agri_data)
        
        # Expectation: 
        # - Mysore should have MNREGA data (100)
        # - Mandya should have NaN/0 (since no match in MNREGA df) -> Script fills 0
        # - Chennai should be dropped (Left Join on Agri)
        
        self.assertEqual(len(merged), 2) # Mysore + Mandya
        self.assertEqual(merged.iloc[0]['MNREGA_JobCards_Issued'], 100)
        self.assertEqual(merged.iloc[1]['MNREGA_JobCards_Issued'], 0.0) # Filled with 0
        
if __name__ == '__main__':
    unittest.main()
