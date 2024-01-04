from magnet.utils.globals import _f

def error_cb(error, connection):
    _f('info', connection)
    _f('error', error)

def closed_cb(error, connection):
    _f('info', connection)
    _f('warn', "connection closed")

def discovery_cb(error, connection):
    _f('info', connection)
    _f('info', "server found")

def reconn_cb(error, connection):
    _f('info', connection)
    _f('warn', "reconnected")

def stats(connection):
    return connection.stats