import os, asyncio
from roles import index
from magnet.utils.globals import _f
magnet_role = os.getenv('MAGNET_ROLE')

def main():
    if magnet_role == 'index':
        asyncio.run(index.main())
    else:
        _f('fatal', 'No MAGNET_ROLE set! 😭')

if __name__ == "__main__":
    main()
