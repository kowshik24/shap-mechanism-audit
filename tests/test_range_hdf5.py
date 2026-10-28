from unittest.mock import patch
import pytest
from pdm_audit.range_hdf5 import RangeReader


class Response:
    def __init__(self,payload,status,content_range):
        self.payload=payload;self.status=status;self.headers={'Content-Range':content_range}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,n):return self.payload[:n]


def test_range_reader_seek_crosses_blocks_and_reuses_cache(tmp_path):
    payload=b'abcdefghij';calls=[]
    def fetch(request,timeout):
        value=request.get_header('Range');a,b=map(int,value[6:].split('-'));calls.append((a,b))
        return Response(payload[a:b+1],206,f'bytes {a}-{b}/10')
    with patch('urllib.request.urlopen',fetch):
        reader=RangeReader('https://example.org/data',10,tmp_path,block_size=4,max_new_bytes=10)
        assert reader.read(6)==b'abcdef';assert reader.seek(-3,1)==3
        assert reader.read(7)==b'defghij';assert reader.read(1)==b''
        reader.seek(0);assert reader.read(10)==payload
    assert calls==[(0,3),(4,7),(8,9)] and reader.downloaded==10


def test_range_reader_rejects_full_object_reply_and_download_budget(tmp_path):
    with patch('urllib.request.urlopen',return_value=Response(b'abcd',200,'bytes 0-3/4')):
        reader=RangeReader('https://example.org/data',4,tmp_path,block_size=4)
        with pytest.raises(ValueError,match='exact bounded'):reader.read(1)
    reader=RangeReader('https://example.org/data',4,tmp_path,block_size=4,max_new_bytes=3)
    with pytest.raises(RuntimeError,match='budget'):reader.read(1)
