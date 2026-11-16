"""Private typed transport causes survive snapshots; no HTTP/native/model work."""
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from urllib.error import URLError

from medlens.api_contract import public_report
from medlens.api_service import _pipeline_error
from medlens.ingestion.pubmed import PubMedClient, PubMedError
from medlens.ingestion.snapshots import download_snapshot
from pubmed_helpers import Clock, Response, http_error, make_client, search_json, xml_records

SECRET='y0_sensitive-fixture https://user:password@fixture.invalid/?token=private'


class TransportDiagnosticsTests(unittest.TestCase):
    def failure(self, reason):
        client,clock,transport=make_client([URLError(reason)]*3)
        with self.assertRaises(PubMedError)as raised:client.search('fixture query')
        return raised.exception,clock,transport

    def test_permission_reason_errno_and_retry_count_are_retained_without_raw_text(self):
        error,clock,transport=self.failure(PermissionError(1,SECRET))
        self.assertEqual(str(error),'esearch.fcgi: transport failed after 3 attempt(s)')
        self.assertEqual(len(transport.calls),3)
        self.assertEqual(clock.sleeps,[1,2])
        self.assertEqual([d['attempt']for d in error.transport_diagnostics],[1,2,3])
        self.assertTrue(all(d['exception_class']=='URLError'and d['reason_class']=='PermissionError'
            and d['reason_errno']==1 and d['stage']=='open-response'for d in error.transport_diagnostics))
        encoded=json.dumps(error.transport_diagnostics)
        for marker in ('sensitive','https://','password','token=','private'):
            self.assertNotIn(marker,encoded)

    def test_dns_reason_class_is_distinct_and_no_untrusted_reason_string_is_saved(self):
        error,_,_=self.failure(socket.gaierror(-2,SECRET))
        self.assertEqual(error.transport_diagnostics[-1]['reason_class'],'gaierror')
        self.assertEqual(error.transport_diagnostics[-1]['reason_errno'],-2)
        string_error,_,_=self.failure(SECRET)
        self.assertEqual(string_error.transport_diagnostics[-1]['reason_class'],'str')
        self.assertNotIn(SECRET,json.dumps(string_error.transport_diagnostics))

    def test_custom_exception_class_name_cannot_carry_arbitrary_metadata(self):
        SensitiveError=type('y0_sensitive_class_name',(OSError,),{})
        error,_,_=self.failure(SensitiveError(1,SECRET))
        self.assertEqual(error.transport_diagnostics[-1]['reason_class'],'OtherError')
        self.assertNotIn('sensitive',json.dumps(error.transport_diagnostics))

    def test_attempt_elapsed_excludes_backoff_and_marks_body_read_errors(self):
        clock=Clock()
        class BodyFailure(Response):
            def read(self,*args):
                from http.client import IncompleteRead
                clock.value+=.25
                raise IncompleteRead(b'private-body',10)
        def opener(request,timeout):return BodyFailure(b'')
        client=PubMedClient(opener=opener,sleep=clock.sleep,monotonic=clock.now)
        with self.assertRaises(PubMedError)as raised:client.search('fixture')
        rows=raised.exception.transport_diagnostics
        self.assertTrue(all(row['stage']=='read-response'and row['exception_class']=='IncompleteRead'for row in rows))
        self.assertEqual([row['elapsed_seconds']for row in rows],[.25,.25,.25])
        self.assertNotIn('private-body',json.dumps(rows))

    def test_http_status_and_recovered_history_keep_existing_retry_after(self):
        client,clock,transport=make_client([http_error(429,'6'),search_json([],0)])
        with tempfile.TemporaryDirectory()as temp:
            path=Path(temp)/'snapshot';manifest=download_snapshot(client,'fixture',path)
            rows=manifest['responses'][0]['transport_diagnostics']
            self.assertEqual([r['outcome']for r in rows],['error','success'])
            self.assertEqual(rows[0]['http_status'],429)
            self.assertEqual(rows[0]['retry_delay_seconds'],6)
            self.assertEqual(clock.sleeps,[6])
            self.assertEqual(len(transport.calls),2)
            self.assertEqual(json.loads((path/'manifest.json').read_text()),manifest)

    def test_failed_search_diagnostics_are_durable_even_without_any_response(self):
        client,_,_=make_client([URLError(PermissionError(1,SECRET))]*3)
        with tempfile.TemporaryDirectory()as temp:
            path=Path(temp)/'snapshot'
            with self.assertRaises(PubMedError):download_snapshot(client,'fixture',path)
            manifest=json.loads((path/'manifest.json').read_text())
            self.assertEqual(manifest['status'],'failed')
            self.assertEqual(manifest['responses'],[])
            self.assertEqual(len(manifest['error']['transport_diagnostics']),3)
            self.assertEqual(manifest['error']['transport_diagnostics'][0]['reason_errno'],1)
            self.assertNotIn('password',json.dumps(manifest))

    def test_failed_fetch_retains_completed_batch_and_exact_failed_endpoint(self):
        events=[search_json(),xml_records(['31','12']),*[URLError(ConnectionResetError(54,SECRET))]*3]
        client,_,_=make_client(events)
        with tempfile.TemporaryDirectory()as temp:
            path=Path(temp)/'snapshot'
            with self.assertRaises(PubMedError):download_snapshot(client,'fixture',path,limit=3,batch_size=2)
            manifest=json.loads((path/'manifest.json').read_text())
            self.assertTrue((path/'fetch-0001.xml').is_file())
            self.assertEqual(len(manifest['responses']),2)
            self.assertTrue(all(row['endpoint']=='efetch.fcgi'for row in manifest['error']['transport_diagnostics']))
            self.assertEqual(manifest['error']['transport_diagnostics'][0]['reason_class'],'ConnectionResetError')

    def test_public_projection_and_error_code_do_not_expose_private_diagnostics(self):
        error,_,_=self.failure(PermissionError(1,SECRET))
        report={'status':'failed','stage':'pubmed-download','question':'Fixture question?',
            'error':{'type':'PubMedError','message':str(error),'stage':'pubmed-download',
                'transport_diagnostics':list(error.transport_diagnostics)}}
        self.assertEqual(_pipeline_error(report),{'code':'pubmed_error','message':'Fresh PubMed retrieval failed.','stage':'pubmed-download'})
        public=public_report(report)
        self.assertNotIn('transport_diagnostics',json.dumps(public))
        self.assertNotIn('PermissionError',json.dumps(public))

    def test_exception_constructor_retains_runtimeerror_argument_contract(self):
        for args in ((),('fixture',),('fixture','context')):
            self.assertEqual(PubMedError(*args).args,RuntimeError(*args).args)
            self.assertEqual(str(PubMedError(*args)),str(RuntimeError(*args)))

    def test_no_tensor_runtime_is_imported_by_transport_diagnostics(self):
        # Other tests may already have imported tensors. Check this import in a
        # fresh isolated stdlib process instead of depending on suite ordering.
        import subprocess
        source = str(Path(__file__).resolve().parents[1] / 'src')
        code = ('import sys; sys.path.insert(0, ' + repr(source) + '); '
                'import medlens.ingestion.pubmed; '
                'assert not any(n == p or n.startswith(p + ".") '
                'for n in sys.modules for p in '
                '("torch","transformers","numpy","mlx","mlx_lm"))')
        subprocess.run([sys.executable, '-I', '-S', '-c', code], check=True,
                       capture_output=True, text=True, timeout=10)


if __name__=='__main__':unittest.main()
