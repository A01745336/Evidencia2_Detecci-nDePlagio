import unittest
from plagiarism_detection import evaluate_performance, generate_report
from sklearn.metrics import roc_curve, auc

class TestEvaluation(unittest.TestCase):
    def test_performance_metrics(self):
        suspicious_filenames = ['doc1.txt', 'doc2.txt', 'doc3.txt']
        similarities = [
                        [0.5, 0.2],  # Similitudes de doc1.txt con cada documento original
                        [0.3, 0.6],  # Similitudes de doc2.txt con cada documento original
                        [0.7, 0.1]   # Similitudes de doc3.txt con cada documento original
                       ]
        ground_truth = {'doc1.txt': True, 'doc2.txt': False, 'doc3.txt': True}
        results = evaluate_performance(similarities, 0.3, ground_truth, suspicious_filenames)
        self.assertEqual(results['TP'], 2)
        self.assertEqual(results['TN'], 0)
        self.assertEqual(results['FP'], 1)
        self.assertEqual(results['FN'], 0)

    def test_auc_calculation(self):
        # AUC should be correctly calculated
        similarities = [0, 0.5, 1]
        ground_truth = [0, 1, 1]
        fpr, tpr, thresholds = roc_curve(ground_truth, similarities)
        auc_value = auc(fpr, tpr)
        self.assertAlmostEqual(auc_value, 1)

    def test_mismatched_lengths(self):
        # Verifica que se maneje correctamente cuando las longitudes no coinciden
        suspicious_filenames = ['doc1.txt', 'doc2.txt']
        similarities = [[0.5, 0.2]]  # Solo un elemento
        ground_truth = { 'doc1.txt': True, 'doc2.txt': False }  # Dos elementos
        with self.assertRaises(IndexError):
            evaluate_performance(similarities, 0.5, ground_truth, suspicious_filenames)

    def test_all_false_positives(self):
        suspicious_filenames = ['doc1.txt', 'doc2.txt']
        similarities = [
                        [0.6],  # False positive
                        [0.8]   # False positive
                       ]
        ground_truth = {'doc1.txt': False, 'doc2.txt': False}
        results = evaluate_performance(similarities, 0.5, ground_truth, suspicious_filenames)
        self.assertEqual(results['TP'], 0)
        self.assertEqual(results['FP'], 2)
        self.assertEqual(results['TN'], 0)
        self.assertEqual(results['FN'], 0)

    def test_all_true_negatives(self):
        suspicious_filenames = ['doc1.txt', 'doc2.txt']
        similarities = [
                        [0.1],  # True negative
                        [0.2]   # True negative
                       ]
        ground_truth = {'doc1.txt': False, 'doc2.txt': False}
        results = evaluate_performance(similarities, 0.3, ground_truth, suspicious_filenames)
        self.assertEqual(results['TP'], 0)
        self.assertEqual(results['FP'], 0)
        self.assertEqual(results['TN'], 2)
        self.assertEqual(results['FN'], 0)

    def test_varying_thresholds(self):
        suspicious_filenames = ['doc1.txt', 'doc2.txt', 'doc3.txt']
        similarities = [
                        [0.5, 0.2],
                        [0.3, 0.6],
                        [0.7, 0.1]
                       ]
        ground_truth = {'doc1.txt': True, 'doc2.txt': False, 'doc3.txt': True}

        low_threshold_results = evaluate_performance(similarities, 0.2, ground_truth, suspicious_filenames)
        high_threshold_results = evaluate_performance(similarities, 0.6, ground_truth, suspicious_filenames)

        self.assertEqual(low_threshold_results['TP'], 2)  # Lower threshold includes more positives
        self.assertEqual(high_threshold_results['TP'], 1)  # Higher threshold excludes lower similarities

    def test_non_existent_documents(self):
        suspicious_filenames = ['doc1.txt', 'doc2.txt', 'doc5.txt']  # doc5.txt does not exist in similarities or ground truth
        similarities = [
                        [0.5, 0.2],
                        [0.3, 0.6]
                       ]
        ground_truth = {'doc1.txt': True, 'doc2.txt': False}
        with self.assertRaises(KeyError):  # Expecting a KeyError due to missing document in ground truth
            evaluate_performance(similarities, 0.3, ground_truth, suspicious_filenames)

if __name__ == '__main__':
    unittest.main()
