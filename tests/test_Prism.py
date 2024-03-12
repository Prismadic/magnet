import pytest
from magnet.ic.field import Prism

# Prism can be initialized with a valid PrismConfig instance or a dictionary.
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

    prism = Prism(config)
    assert isinstance(prism, Prism)

@pytest.mark.asyncio
async def test_invalid_initialization():
    config = "invalid_config"

    with pytest.raises(ValueError):
        Prism(config)

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

    prism = Prism(config)
    assert isinstance(prism, Prism)

    js, kv, os = await prism.align()
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

    prism = Prism(config)
    await prism.align()
    
    assert prism.kv is not None
    assert prism.os is not None

    await prism.off()
    assert prism.nc.is_closed