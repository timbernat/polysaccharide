import pkgutil, importlib

LOGGERS_MASTER = [] # contains module-specific loggers for all submodules
for _loader, _module_name, _ispkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f'{__package__}.{_module_name}')
    globals()[_module_name] = module # register module to namespace

    if hasattr(module, 'LOGGER'): # make record of logger if one is present
        LOGGERS_MASTER.append(getattr(module, 'LOGGER'))