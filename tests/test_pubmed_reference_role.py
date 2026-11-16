"""Reference-role candidate broadening only; original clinical constraints stay intact."""
import unittest
from medlens.pubmed_query import plan_query


class ReferenceRoleQueryTests(unittest.TestCase):
    def role_changes(self, result):
        return [row for row in result['query_transformations'] if row['rule']=='reference-role-scaffold']

    def test_short_and_existing_standard_role_forms_keep_named_reference(self):
        for role in ('as the reference','as a reference','as reference',
                     'as the reference standard','as a reference standard','as reference standard',
                     'as the gold standard','as a gold standard','as gold standard'):
            with self.subTest(role=role):
                question=f'What is the accuracy of sensor-X using assay-Y {role}?'
                result=plan_query(question)
                self.assertEqual(result['submitted_query'],'the accuracy of sensor-X assay-Y')
                self.assertEqual(result['question'],question)
                change=self.role_changes(result)
                self.assertEqual(len(change),1)
                self.assertEqual(change[0]['removed_text'],role)
                self.assertEqual(question[change[0]['char_start']:change[0]['char_end']],role)

    def test_all_four_consumed_diagnostic_questions_drop_role_only(self):
        questions=(
            'What is the diagnostic accuracy of high-sensitivity cardiac troponin for myocardial infarction in adults using adjudicated diagnosis as the reference?',
            'What is the diagnostic accuracy of fecal calprotectin for inflammatory bowel disease in adults using colonoscopy with biopsy as the reference?',
            'What is the diagnostic accuracy of multiparametric MRI for clinically significant prostate cancer using biopsy as the reference?',
            'What is the diagnostic accuracy of the sFlt-1/PlGF ratio for preeclampsia using clinical diagnosis as the reference?',
        )
        required=(('high-sensitivity cardiac troponin','myocardial infarction','adults','adjudicated diagnosis'),
                  ('fecal calprotectin','inflammatory bowel disease','adults','colonoscopy with biopsy'),
                  ('multiparametric MRI','clinically significant prostate cancer','biopsy'),
                  ('sFlt-1/PlGF','preeclampsia','clinical diagnosis'))
        for question,terms in zip(questions,required):
            with self.subTest(question=question):
                result=plan_query(question)
                self.assertEqual(result['question'],question)
                self.assertNotIn('as the reference',result['submitted_query'])
                self.assertEqual(self.role_changes(result)[0]['removed_text'],'as the reference')
                for term in terms:self.assertIn(term,result['submitted_query'])

    def test_quoted_role_text_is_preserved(self):
        for quoted in ('"as the reference"','"as a reference"','"as the reference standard"'):
            result=plan_query(f'Does {quoted} affect marker-X?')
            self.assertIn(quoted,result['submitted_query'])
            self.assertEqual(self.role_changes(result),[])

    def test_explicit_override_stays_verbatim(self):
        question='What is sensor-X accuracy using assay-Y as the reference?'
        for override in ('  "as the reference"[Title] OR "reference standard"[All Fields]  ',
                         'assay-Y as a reference', question):
            with self.subTest(override=override):
                result=plan_query(question,override)
                self.assertEqual(result['submitted_query'],override)
                self.assertEqual(result['question'],question)
                self.assertEqual(result['query_transformations'],[])

    def test_following_noun_is_not_treated_as_a_short_role_marker(self):
        for suffix in ('protein','intervention','dose','for the control arm'):
            question=f'Does assay-X use molecule-Y as the reference {suffix}?'
            result=plan_query(question)
            self.assertIn('as the reference '+suffix,result['submitted_query'])
            self.assertEqual(self.role_changes(result),[])

    def test_negated_role_is_retained(self):
        for negation in ('not','never','no','without'):
            for role in ('as the reference','as the reference standard'):
                result=plan_query(f'Does assay-X use assay-Y {negation} {role}?')
                self.assertIn(negation+' '+role,result['submitted_query'])
                self.assertEqual(self.role_changes(result),[])

    def test_role_named_identifiers_are_not_partially_removed(self):
        for phrase in ('X-as the reference','as the reference-2','as the reference standard-2',
                       'as the reference_standard','as gold'):
            result=plan_query(f'Does assay-X use {phrase}?')
            self.assertIn(phrase,result['submitted_query'])
            self.assertEqual(self.role_changes(result),[])

    def test_short_role_clause_boundary_preserves_following_clause(self):
        question='Does assay-X use assay-Y as the reference; and retain marker-Z?'
        result=plan_query(question)
        self.assertIn('assay-Y',result['submitted_query'])
        self.assertIn('retain marker-Z',result['submitted_query'])
        self.assertEqual(self.role_changes(result)[0]['removed_text'],'as the reference')

    def test_validation_limits_and_controls_are_unchanged(self):
        for question in ('Does assay-X use assay-Y as the reference?\n', 'x'*8001):
            with self.assertRaises(ValueError):plan_query(question)
        with self.assertRaises(ValueError):plan_query('Does X affect Y?','x'*4001)


if __name__=='__main__':unittest.main()
