"""Check the public dry-run CLI without starting inference or GPU work."""
import json
import subprocess
import sys
import unittest

class PlanTests(unittest.TestCase):
    def test_cli_separates_quality_suites(self):
        command=[sys.executable]+(['-O'] if sys.flags.optimize else [])
        result=subprocess.run(command+['-m','sparse_attn.asynctls_offload'],check=True,capture_output=True,text=True)
        plan=json.loads(result.stdout)
        self.assertEqual(plan['ruler_quality_lengths'],[32768])
        self.assertEqual(plan['longbench'],'actual existing task lengths')
        self.assertNotIn('quality_lengths',plan)
        self.assertNotIn('quality_suite',plan)
        self.assertEqual(plan['kernel_lengths'],[32768,65536,131072])
        self.assertEqual(plan['e2e_lengths'],[32768,65536,98304])
        self.assertIs(plan['gpu_started'],False)
        self.assertEqual(plan['status'],'dry-run only')

if __name__=='__main__':unittest.main()
