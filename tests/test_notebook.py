"""Regression checks using synthetic fixtures only; no original data required."""
import json
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks' / 'titanic-survival-prediction.ipynb'
CELLS = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']

def source(index):
    return ''.join(CELLS[index]['source'])

def run(namespace, *indices):
    for i in indices:
        exec(compile(source(i), f'{NOTEBOOK.name}:cell-{i}', 'exec'), namespace)

class NotebookTests(unittest.TestCase):
    def test_python_cells_compile(self):
        for i, cell in enumerate(CELLS):
            if cell['cell_type'] != 'code':
                continue
            # Notebook magics are handled by IPython, not the Python parser.
            code = '\n'.join(line for line in source(i).splitlines() if not line.lstrip().startswith(('%', '!')))
            with self.subTest(cell=i):
                compile(code, f'{NOTEBOOK.name}:cell-{i}', 'exec')

    def test_inference_uses_unlabeled_passenger_ids(self):
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import GradientBoostingClassifier
        def passengers(n, start):
            return pd.DataFrame({'PassengerId': range(start, start+n), 'Name': ['x']*n, 'Ticket': ['x']*n, 'Cabin': ['x']*n, 'Embarked': ['S']*n, 'Fare': [10.]*n, 'Sex': ['female', 'male']*(n//2), 'Age': [20., np.nan]*(n//2), 'Pclass': [1, 3]*(n//2), 'SibSp': [0]*n, 'Parch': [0]*n})
        train = passengers(20, 1)
        train['Survived'] = [1, 0]*10
        test = passengers(4, 100)
        ns = dict(pd=pd, train=train, test=test, train_test_split=train_test_split, GradientBoostingClassifier=GradientBoostingClassifier)
        run(ns, 7, 9, 10, 22, 26, 28, 43, 44, 45, 46)
        self.assertNotIn('PassengerId', ns['x'].columns)
        self.assertNotIn('Survived', ns['X_inference'].columns)
        self.assertEqual(ns['new_dataframe']['PassengerId'].tolist(), [100, 101, 102, 103])
        self.assertTrue(ns['new_dataframe']['Survived'].isin([0, 1]).all())

if __name__ == '__main__':
    unittest.main()
