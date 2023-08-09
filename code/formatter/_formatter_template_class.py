
from formatter._formatter import Formatter

class FormatterTemplateClass(Formatter):
    def __init__(self, template, **kwargs):

        super().__init__(template)

    def format(self, data):
        return data


# unit tests for FormatterTemplateClass
# TODO: Adapt this to your needs
import unittest
class TestFormatterTemplateClass(unittest.TestCase):
    def test_format(self):
        formatter = FormatterTemplateClass()
        self.assertEqual(formatter.format('test'), 'test')
            
    def test_format_with_multiple_variables(self):
            formatter = FormatterTemplateClass()
            self.assertEqual(formatter.format('test'), 'test')

if __name__ == '__main__':
    unittest.main()
# TODO:
# 0. Make a copy of this file and rename it to your needs
# 1. Adapt the format method to your needs
# 2. Adapt the unit tests to your needs
# 3. Run the unit tests
# 4. Delete this comment
# 5. Add your new formatter to the __init__.py file in the formatter folder
