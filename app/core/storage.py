import asyncio
import hashlib
from pathlib import Path
from typing import BinaryIO

from app.core.config import settings
from app.exceptions.storage import FileTooLargeError, InvalidStorageKeyError


class FileStorage:
    """
    文件存储器
    """
    def __init__(self, base_dir: str):
        # 创建Path对象，将相对路径转换成绝对路径
        self.base_dir = Path(base_dir).resolve()
        # parents=True, 父目录不存在，也一起创建
        # exist_ok=True, 父目录已存在，不要报错
        self.base_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def _resolve_path(
            self,
            storage_key: str
    ) -> Path:
        # / : 拼接文件路径
        path = (self.base_dir / storage_key).resolve()
        if self.base_dir not in path.parents:
            raise InvalidStorageKeyError('Invalid storage key')

        return path

    def _save_sync(
            self,
            source: BinaryIO,
            storage_key: str,
            max_size: int
    ) -> tuple[int, str]:
        """
        同步执行实际的文件写入
        :param source: 文件数据来源，上传文件的二进制对象
        :param storage_key: 保存位置
        :param max_size: 最大允许文件大小
        :return:
        """
        destination = self._resolve_path(storage_key)
        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        sha256 = hashlib.sha256()
        total_size = 0
        try:
            with destination.open('wb') as output:
                while True:
                    chunk: bytes = source.read(1024 * 1024)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > max_size:
                        raise FileTooLargeError()
                    sha256.update(chunk)
                    output.write(chunk)
        except Exception:
            # missing_ok=True, 如果这个文件本来就不存在，也不要因为删除失败再抛一个异常
            destination.unlink(missing_ok=True)
            raise

        return total_size, sha256.hexdigest()

    async def save(
            self,
            source: BinaryIO,
            storage_key: str,
            max_size: int
    ) -> tuple[int, str]:
        """
        异步接口
        实际同步文件写入放到线程中执行
        """
        return await asyncio.to_thread(
            self._save_sync,
            source,
            storage_key,
            max_size
        )

    def _read_bytes_sync(
            self,
            storage_key: str
    ) -> bytes:
        path = self._resolve_path(storage_key)
        # 判断这个路径是不是一个真实存在的文件
        if not path.is_file():
            raise FileNotFoundError(storage_key)

        return path.read_bytes()

    async def read_bytes(
            self,
            storage_key: str
    ) -> bytes:
        """
        异步读取文件
        """
        return await asyncio.to_thread(
            self._read_bytes_sync,
            storage_key
        )

    def _delete_sync(
            self,
            storage_key: str
    ) -> None:
        path = self._resolve_path(storage_key)
        path.unlink(missing_ok=True)

    async def delete(
            self,
            storage_key: str
    ) -> None:
        """
        异步删除文件
        """
        await asyncio.to_thread(
            self._delete_sync,
            storage_key
        )


file_storage = FileStorage(settings.file_storage_dir)
