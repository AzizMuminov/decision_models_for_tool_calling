from importer.config import BATCH_SIZE, S3_PREFIX
def test_cfg():
    assert BATCH_SIZE == 1000 and S3_PREFIX == "imports/"
