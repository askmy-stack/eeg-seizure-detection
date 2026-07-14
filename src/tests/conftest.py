"""
pytest configuration for src/tests.

test_tensors.py and test_wrappers.py were written against a 'core' /
'datasources' module architecture that is not part of this repo. They are
skipped at collection time to keep `pytest src/tests/ -v` clean.

(test_pipeline.py had the same problem and has been removed outright — see
issue #54.)
"""
collect_ignore = [
    "test_tensors.py",
    "test_wrappers.py",
]
