"""
test_cli.py
-----------
Tests pour l'interface en ligne de commande (main.py).
"""

import pytest
import subprocess
import sys
from pathlib import Path


class TestCLIExecution:
    """Tests de bon fonctionnement des commandes CLI de main.py."""

    def test_cli_help(self):
        """Vérifie que 'python main.py --help' s'exécute avec succès et liste les 5 sous-commandes."""
        cmd = [sys.executable, "main.py", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        assert res.returncode == 0
        assert "audit" in res.stdout
        assert "api" in res.stdout
        assert "dashboard" in res.stdout
        assert "anonymize" in res.stdout
        assert "train" in res.stdout

    def test_cli_audit_help(self):
        """Vérifie l'aide de la sous-commande audit."""
        cmd = [sys.executable, "main.py", "audit", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        assert res.returncode == 0
        assert "--threshold" in res.stdout
        assert "--top" in res.stdout
        assert "--output" in res.stdout

    def test_cli_anonymize_help(self):
        """Vérifie l'aide de la sous-commande anonymize."""
        cmd = [sys.executable, "main.py", "anonymize", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        assert res.returncode == 0
        assert "--shift-days" in res.stdout
