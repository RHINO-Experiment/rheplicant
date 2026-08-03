import pytest

from rheplicant import DataIngestionError, DirtError


def test_data_ingestion_error_is_catchable_as_the_family_and_as_value_error():
    assert issubclass(DataIngestionError, DirtError)
    assert issubclass(DataIngestionError, ValueError)
    with pytest.raises(DirtError):
        raise DataIngestionError("bad file")
