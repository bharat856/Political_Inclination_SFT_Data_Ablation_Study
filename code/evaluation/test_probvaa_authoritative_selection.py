"""CPU-only checks for the authoritative personal-template selection rule."""

import csv
import unittest

from probvaa_authoritative_selection import (
    AUTHORITATIVE_TEMPLATE_CSV,
    EXPECTED_PERSONAL_TEMPLATES,
    PERSONAL_PROMPT_TYPE,
    PROMPT_TYPE_COLUMN,
    expected_generation_count,
    load_personal_templates,
)


class AuthoritativePersonalTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with AUTHORITATIVE_TEMPLATE_CSV.open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            reader = csv.DictReader(handle)
            cls.fieldnames = reader.fieldnames
            cls.source_rows = [dict(row) for row in reader]
        cls.selected = load_personal_templates()

    def test_authoritative_prompt_type_column_exists(self):
        self.assertIn(PROMPT_TYPE_COLUMN, self.fieldnames)

    def test_exactly_six_personal_templates_remain(self):
        self.assertEqual(len(self.selected), EXPECTED_PERSONAL_TEMPLATES)
        self.assertEqual(
            {row[PROMPT_TYPE_COLUMN] for row in self.selected},
            {PERSONAL_PROMPT_TYPE},
        )

    def test_impersonal_templates_are_excluded(self):
        self.assertTrue(
            any(
                row[PROMPT_TYPE_COLUMN] == "impersonal"
                for row in self.source_rows
            )
        )
        self.assertFalse(
            any(
                row[PROMPT_TYPE_COLUMN] == "impersonal"
                for row in self.selected
            )
        )

    def test_text_and_csv_order_are_preserved(self):
        expected = [
            row
            for row in self.source_rows
            if row[PROMPT_TYPE_COLUMN] == PERSONAL_PROMPT_TYPE
        ]
        self.assertEqual(self.selected, expected)
        self.assertEqual(
            [row["template"] for row in self.selected],
            [row["template"] for row in expected],
        )

    def test_expected_generation_count(self):
        self.assertEqual(
            expected_generation_count(956, 30, self.selected),
            172_080,
        )


if __name__ == "__main__":
    unittest.main()
