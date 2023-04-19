# Create molecule charging toolkit registries
from openff.toolkit import GLOBAL_TOOLKIT_REGISTRY as GTR
from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
GTR.register_toolkit(EspalomaChargeToolkitWrapper)

TOOLKITS = { 
    tk.toolkit_name : tk
        for tk in GTR.registered_toolkits
}

# import submodules
from . import application, averaging, residues, types # must come last due to variables defined above

# logging setup (will trickle through to submodules)
import logging
LOGGER = logging.getLogger(__name__)