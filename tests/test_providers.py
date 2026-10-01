"""Key loading from Hermes' .env and the stdlib hash embedder."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from embedder import _embed_hash, cosine_similarity  # noqa: E402


class HermesEnvTest(unittest.TestCase):
    def test_reads_only_known_keys(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, ".env")
        with open(path, "w") as f:
            f.write("# comment\nOPENROUTER_API_KEY=sk-or\n"
                    "export DEEPSEEK_API_KEY=\"sk-ds\"\n"
                    "OPENAI_API_KEY=sk-oa # trailing\nDEEPSEEK_BASE_URL='http://x'\n")
        self.assertEqual(config._read_hermes_env(path), {
            "DEEPSEEK_API_KEY": "sk-ds", "OPENAI_API_KEY": "sk-oa",
            "DEEPSEEK_BASE_URL": "http://x"})

    def test_placeholder_is_not_usable(self):
        self.assertFalse(config._usable("your_openai_api_key"))
        self.assertFalse(config._usable(""))
        self.assertTrue(config._usable("sk-real"))


class HashEmbedderTest(unittest.TestCase):
    def sim(self, a, b):
        va, vb = _embed_hash([a, b])
        return cosine_similarity(va, vb)

    def test_related_above_unrelated(self):
        # Thresholds in config.py (0.25 same-topic) rely on this separation.
        related = [("Project database is Postgres 17 on db.internal.",
                    "Project database is Postgres 16 on db.internal."),
                   ("사용자는 이제 Neovim으로 코딩한다.", "사용자는 VS Code로 코딩한다.")]
        unrelated = [("User's timezone is Asia/Seoul.", "The API rate limit is 100 requests per minute."),
                     ("사용자는 간결한 한국어 답변을 선호한다.", "배포는 docker-compose와 nginx로 한다.")]
        for a, b in related:
            self.assertGreater(self.sim(a, b), 0.25, (a, b))
        for a, b in unrelated:
            self.assertLess(self.sim(a, b), 0.25, (a, b))

    def test_deterministic(self):
        self.assertEqual(_embed_hash(["뇽뇽"]), _embed_hash(["뇽뇽"]))


if __name__ == "__main__":
    unittest.main()
