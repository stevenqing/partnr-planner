import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure we start with CLOUDGPT_USE_AZURE_CLI unset
if "CLOUDGPT_USE_AZURE_CLI" in os.environ:
    del os.environ["CLOUDGPT_USE_AZURE_CLI"]


class TestCloudGPTOffline(unittest.TestCase):
    def test_step_2_unset_raises_runtime_error(self):
        # Import cloudgpt_aoai dynamically or normally
        import cloudgpt_aoai

        # Monkeypatch subprocess to raise if called
        with patch("subprocess.run") as mock_run:
            with self.assertRaises(RuntimeError) as context:
                cloudgpt_aoai._get_azure_cli_token()
            self.assertIn("Set CLOUDGPT_USE_AZURE_CLI=1", str(context.exception))
            mock_run.assert_not_called()
        print(
            "Step 2 passed: _get_azure_cli_token() raised RuntimeError before subprocess when CLOUDGPT_USE_AZURE_CLI is unset."
        )

    def test_step_3_monkeypatch(self):
        import io

        from openai import AzureOpenAI

        import cloudgpt_aoai

        # Reset cached values to a clean slate
        cloudgpt_aoai._cached_token = None
        cloudgpt_aoai._cached_expiry = 0

        fake_token = "fake-secret-token-12345"
        fake_expiry = 2000000000  # far in future

        fake_stdout = f'{{"accessToken": "{fake_token}", "expires_on": {fake_expiry}}}'

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with patch.dict(os.environ, {"CLOUDGPT_USE_AZURE_CLI": "1"}), patch(
            "shutil.which", return_value="/usr/bin/az"
        ), patch("time.time", return_value=1000000000), patch(
            "subprocess.run"
        ) as mock_run, patch(
            "sys.stdout", stdout_capture
        ), patch(
            "sys.stderr", stderr_capture
        ):
            mock_completed_proc = MagicMock()
            mock_completed_proc.stdout = fake_stdout
            mock_run.return_value = mock_completed_proc

            # Call provider twice
            token1 = cloudgpt_aoai._get_azure_cli_token()
            token2 = cloudgpt_aoai._get_azure_cli_token()

            # Assert subprocess run invoked once
            mock_run.assert_called_once()

            # Token returned twice
            self.assertEqual(token1, fake_token)
            self.assertEqual(token2, fake_token)

            # Assert no stdout/stderr contains the fake token
            sys.stdout.write("Captured internal stdout check\n")
            all_stdout = stdout_capture.getvalue()
            all_stderr = stderr_capture.getvalue()

            self.assertNotIn(fake_token, all_stdout)
            self.assertNotIn(fake_token, all_stderr)

            # get_openai_client returns AzureOpenAI without invoking provider/stub
            mock_run.reset_mock()
            client = cloudgpt_aoai.get_openai_client()
            self.assertIsInstance(client, AzureOpenAI)
            mock_run.assert_not_called()

        print("Step 3 passed: Monkeypatching and verification succeeded fully.")


if __name__ == "__main__":
    unittest.main()
