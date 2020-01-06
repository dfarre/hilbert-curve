import unittest
from ipynb_tests import tester


class HilbertTests(tester.NotebookTester, unittest.TestCase):
    notebooks_path = 'notebooks'
