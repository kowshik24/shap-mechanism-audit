"""Cached HTTP range reader, so h5py can read single datasets from a large remote HDF5 file."""
import io
import urllib.request
from pathlib import Path


class RangeReader(io.RawIOBase):
    """File-like view of a remote file that fetches fixed-size blocks on demand and caches them on disk."""

    def __init__(self, url, size, cache, block_size=65536, max_new_bytes=256 * 1024 * 1024):
        self.url, self.size, self.cache = url, int(size), Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.block_size, self.max_new_bytes = block_size, max_new_bytes
        self.downloaded, self.position = 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else (self.position + offset if whence == 1 else self.size + offset)
        if position < 0:
            raise ValueError('negative range position')
        self.position = position
        return position

    def _block(self, start):
        end = min(start + self.block_size, self.size) - 1
        length, path = end - start + 1, self.cache / f'{start:016d}.bin'
        if path.exists():
            data = path.read_bytes()
            if len(data) != length:
                raise ValueError('cached block has the wrong length')
            return data
        if self.downloaded + length > self.max_new_bytes:
            raise RuntimeError('range download budget exhausted')
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=45) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('server did not honour the byte range')
            data = response.read(length + 1)
        if len(data) != length:
            raise ValueError('range payload has the wrong length')
        path.write_bytes(data)
        self.downloaded += length
        return data

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        remaining, parts = min(size, max(0, self.size - self.position)), []
        while remaining:
            start = (self.position // self.block_size) * self.block_size
            block = self._block(start)
            offset = self.position - start
            take = min(remaining, len(block) - offset)
            parts.append(block[offset:offset + take])
            self.position += take
            remaining -= take
        return b''.join(parts)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)
