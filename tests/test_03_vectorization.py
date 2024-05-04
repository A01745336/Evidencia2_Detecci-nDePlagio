import unittest
from plagiarism_detection import generate_vector_space_models

class TestVectorization(unittest.TestCase):
    def test_vector_output(self):
        texts = ["hello world", "hello"]
        original_vectors, suspicious_vectors, feature_names = generate_vector_space_models(texts, texts)
        # Ahora esperamos 3 características: 'hello', 'world', y 'hello world'
        self.assertEqual(len(feature_names), 3)
        self.assertEqual(original_vectors.shape[0], 2)
        self.assertEqual(suspicious_vectors.shape[0], 2)
        self.assertEqual(original_vectors.shape[1], 3)
        self.assertEqual(suspicious_vectors.shape[1], 3)

    def test_empty_input(self):
        # Verifica que se maneje correctamente la entrada vacía
        with self.assertRaises(ValueError):
            generate_vector_space_models([], [])  # Entradas vacías

    def test_single_word_texts(self):
        texts = ["hello", "world"]
        original_vectors, suspicious_vectors, feature_names = generate_vector_space_models(texts, texts)
        self.assertEqual(len(feature_names), 2)  # Only 'hello' and 'world' as features
        self.assertEqual(original_vectors.shape[1], 2)

    def test_vector_dimensions_consistency(self):
        texts1 = ["hello world", "hello"]
        texts2 = ["hello universe", "hello there"]
        original_vectors, suspicious_vectors, feature_names = generate_vector_space_models(texts1, texts2)
        self.assertEqual(original_vectors.shape[1], suspicious_vectors.shape[1])  # Same number of features

    def test_handling_special_characters(self):
        texts = ["hello!", "@world#"]
        original_vectors, suspicious_vectors, feature_names = generate_vector_space_models(texts, texts)
        self.assertIn('hello', feature_names)
        self.assertIn('world', feature_names)

    def test_non_string_input(self):
        texts = [123, 456]
        with self.assertRaises(TypeError):
            generate_vector_space_models(texts, texts)

if __name__ == '__main__':
    unittest.main()
