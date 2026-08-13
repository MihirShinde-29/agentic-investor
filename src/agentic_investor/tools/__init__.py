"""Data-source wrappers: market (M1), news (M2), filings (roadmap).

These return plain/Pydantic data and make no LLM calls, so they stay cheap
to unit-test with mocked HTTP.
"""
