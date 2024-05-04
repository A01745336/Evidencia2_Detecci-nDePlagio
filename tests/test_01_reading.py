import unittest
from plagiarism_detection import read_files_in_directory

class TestFileReading(unittest.TestCase):
    def test_empty_directory(self):
        filenames, contents = read_files_in_directory('./empty')
        self.assertEqual(len(filenames), 0)
        self.assertEqual(len(contents), 0)

    def test_nonexistent_directory(self):
        with self.assertRaises(Exception):
            read_files_in_directory('path/to/nonexistent/directory')

    def test_reading_contents(self):
        filenames, contents = read_files_in_directory('./TextosConPlagio')
        self.assertGreater(len(filenames), 0)
        self.assertEqual(len(filenames), len(contents))

    def test_reading_specific_file_type(self):
        filenames, contents = read_files_in_directory('./TextosConPlagio')
        for filename in filenames:
            self.assertTrue(filename.endswith('.txt'))

    def test_ignoring_hidden_files(self):
        filenames, contents = read_files_in_directory('./TextosConPlagio')
        for filename in filenames:
            self.assertFalse(filename.startswith('.'))

    def test_reading_empty_files(self):
        filenames, contents = read_files_in_directory('./TextosVacios')
        for content in contents:
            self.assertEqual(content, '')

    def test_file_read_errors(self):
        with self.assertRaises(IOError):
            read_files_in_directory('./CorruptedFiles')

    def test_file_order_consistency(self):
        filenames1, contents1 = read_files_in_directory('./TextosConPlagio')
        filenames2, contents2 = read_files_in_directory('./TextosConPlagio')
        self.assertEqual(filenames1, filenames2)

if __name__ == '__main__':
    unittest.main()

