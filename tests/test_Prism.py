import pytest
from magnet.ic.field import Magnet

# Magnet can be initialized with a valid MagnetConfig instance or a dictionary.
@pytest.mark.asyncio
async def test_valid_initialization():
    config = {
        "host": "127.0.0.1",
        "credentials": None,
        "domain": "magnet",
        "stream_name": "my_stream",
        "kv_name": "my_kv",
        "os_name": "my_object_store"
    }

    magnet = Magnet(config)
    assert isinstance(magnet, Magnet)

@pytest.mark.asyncio
async def test_invalid_initialization():
    config = "invalid_config"

    with pytest.raises(ValueError):
        Magnet(config)

@pytest.mark.asyncio
async def test_prism_connect_and_setup():
    config = {
        "host": "127.0.0.1",
        "credentials": None,
        "domain": None,
        "stream_name": "my_stream",
        "kv_name": "my_kv",
        "os_name": "my_object_store"
    }

    magnet = Magnet(config)
    assert isinstance(magnet, Magnet)

    js, kv, os = await magnet.align()
    assert js is not None
    assert kv is not None
    assert os is not None

@pytest.mark.asyncio
async def test_close_connection():
    config = {
        "host": "127.0.0.1",
        "credentials": None,
        "domain": None,
        "stream_name": "my_stream",
        "kv_name": "my_kv",
        "os_name": "my_object_store"
    }

    magnet = Magnet(config)
    await magnet.align()
    
    assert magnet.kv is not None
    assert magnet.os is not None

    await magnet.off()
    assert magnet.nc.is_closed