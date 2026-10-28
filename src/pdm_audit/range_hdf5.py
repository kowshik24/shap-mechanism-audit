"""Bounded, cached HTTP range reader for the author-hosted DTU HDF5 release."""
import hashlib
import io
import re
import urllib.request
from pathlib import Path


class RangeReader(io.RawIOBase):
    def __init__(self,url,size,cache,block_size=1024*1024,max_new_bytes=128*1024*1024):
        self.url=url;self.size=int(size);self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.block_size=block_size;self.max_new_bytes=max_new_bytes;self.downloaded=0;self.position=0;self.blocks={}

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=0):
        position=offset if whence==0 else (self.position+offset if whence==1 else self.size+offset)
        if position<0:raise ValueError('Negative range position')
        self.position=position;return position

    def _block(self,start):
        end=min(start+self.block_size,self.size)-1;path=self.cache/f'{start:016d}.bin';length=end-start+1
        if path.exists():
            data=path.read_bytes()
            if len(data)!=length:raise ValueError('Cached range length mismatch')
        else:
            if self.downloaded+length>self.max_new_bytes:raise RuntimeError('Range download budget exhausted')
            request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request,timeout=45) as response:
                expected=f'bytes {start}-{end}/{self.size}'
                if response.status!=206 or response.headers.get('Content-Range')!=expected:
                    raise ValueError('Server did not honor exact bounded byte range')
                data=response.read(length+1)
                if len(data)!=length:raise ValueError('HTTP range payload length mismatch')
            path.write_bytes(data);self.downloaded+=length
        self.blocks[str(start)]={'length':length,'sha256':hashlib.sha256(data).hexdigest()}
        return data

    def read(self,size=-1):
        if size is None or size<0:size=self.size-self.position
        remaining=min(size,max(0,self.size-self.position));parts=[]
        while remaining:
            start=(self.position//self.block_size)*self.block_size
            block=self._block(start);offset=self.position-start;take=min(remaining,len(block)-offset)
            parts.append(block[offset:offset+take]);self.position+=take;remaining-=take
        return b''.join(parts)

    def readinto(self,buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)
