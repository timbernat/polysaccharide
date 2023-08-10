# Creating module-specific logger
import logging
LOGGER = logging.getLogger(__name__)


# auto-importing subpackage and submodules into namespace
from pathlib import Path
import pkgutil, importlib
import importlib_resources as impres

_MODULE_PATH = Path(__file__).parent
for _loader, _module_name, _ispkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f'{__package__}.{_module_name}')
    globals()[_module_name] = module # register module to namespace


# load/generating functional group smarts table
from . import _daylight_scrape
import pandas as pd

_fn_group_table_name : str = 'fn_group_smarts'

fn_group_table_path = _MODULE_PATH / f'{_fn_group_table_name}.csv'
if not fn_group_table_path.exists(): # if data table is missing, scrape data back off of Daylight SMARTS sight and save
    LOGGER.warning(F'No functional group SMARTS data from LUT found on system; regenerating from {_daylight_scrape.DAYLIGHT_URL}')
    FN_GROUP_TABLE = _daylight_scrape.scrape_SMARTS()
    FN_GROUP_TABLE.to_csv(fn_group_table_path)
else:
    LOGGER.info('Loading functional group SMARTS data from LUT')
    FN_GROUP_TABLE = pd.read_csv(fn_group_table_path, index_col=0) # otherwise, read data from table on import