# Custom Imports
from .general import strip_units
from .extratypes import ArrayLike

# File I/O
from pathlib import Path
from shutil import copyfile

# Logging and Shell
import subprocess

# Representation imports
from dataclasses import dataclass, field
from openmm.unit import mole, AVOGADRO_CONSTANT_NA


# Representation classes
@dataclass
class Solvent:
    '''For encapsulating information about solvents'''
    name    : str
    formula : str
    smarts  : str

    density : float
    MW : float # molecular weight

    charges : dict[int, float] = field(default=None)
    structure_file : Path      = field(default=None)
    forcefield_file : Path     = field(default=None)

    @property
    def number_density(self) -> float:
        '''
        Determine the number of solvent molecules per unit volume from known physical constants
        For best results, provide arguments as Quantities with associated units
        '''
        return (self.density / self.MW) * AVOGADRO_CONSTANT_NA

    @property
    def monomer_json_data(self):
        '''Generate a monomer-spec-conformant JSON dictionary entry'''
        return {
            "monomers": {
                self.name : self.smarts
            },
            "caps": {
                self.name : []
            },
            "charges" : {
                self.name : self.charges
            }
        }

# Packmol wrapper methods
def populate_solv_inp_template(template_path : Path, outname : str, outdir : Path, polymer_name : str, solvent_name : str, N : int, box_dims : ArrayLike, precision : int=4) -> Path:
    '''
    Function for programmatically generating packmol .inp files for solvating arbitrary polymer in a box of solvent

    Args:
    outname : str         = name for the resulting .inp file
    outdir : Path         = the target directory to save the resulting .inp file into
    polymer_pdb : path    = path to the desired polymer .pdb file
    solvent_pdb : path    = path to the desired solvent .pdb file
    
    template_path : Path  = the base template file to populate values into
    N : int               = number of solvent molecules to populate into box 
    box_dims : ArrayLike  = array representing the (x, y, z)-lengths of the simulation box - values should have units of angstroms
    '''
    x, y, z = strip_units(box_dims) # ensure only floats are passed on to packmol
    repl_dict = {
        '$POLYMER_FILE' : polymer_name,
        '$SOLVENT_FILE' : solvent_name,
        '$N' : N,
        '$XR' : round(x / 2.0, ndigits=precision), # rounding ensures packmol won;t get stuck in GENCAN loop due to floating-point precision errors
        '$YR' : round(y / 2.0, ndigits=precision), # place polymer at center of box
        '$ZR' : round(z / 2.0, ndigits=precision),
        '$XD' : round(x, ndigits=precision),
        '$YD' : round(y, ndigits=precision),
        '$ZD' : round(z, ndigits=precision),
        '$OUTNAME' : outname
    }
    
    with template_path.open('r') as src_file:
        code = src_file.read()

    for token, value in repl_dict.items():
        code = code.replace(token, str(value))

    outpath = outdir/f'{outname}.inp'
    outpath.touch()
    with outpath.open('w') as packmol_file:
        packmol_file.write(code)

    return outpath

def packmol_solvate_wrapper(template_path : Path, polymer_pdb : Path, solvent_pdb : Path, outdir : Path, outname : str, N : int, box_dims : ArrayLike, precision : int=4) -> Path:
    '''Takes molecule and solvent pdb files and generates a solvated pdb at the specified location with the name <outpath>.pdb. Wrapper for packmol backend'''
    ref_paths, temp_paths = [polymer_pdb, solvent_pdb], []
    for ref_path in ref_paths:
        temp_path = outdir/ref_path.name
        temp_paths.append(temp_path)
        copyfile(ref_path, temp_path)

    inp_temp = populate_solv_inp_template(
        outname=outname, 
        outdir=outdir, 
        polymer_name=polymer_pdb.name, # this deliberately includes the .pdb suffix
        solvent_name=solvent_pdb.name, # this deliberately includes the .pdb suffix
        template_path=template_path,
        N=N,
        box_dims=box_dims,
        precision=precision
    )
    # temp_paths.append(inp_temp)

    cmd = f'packmol < ./{inp_temp.name} > packmol_log.txt'
    proc = subprocess.run(cmd, cwd=f'./{str(outdir)}', shell=True)
    # print(proc)

    for temp_path in temp_paths:
        temp_path.unlink() # delete temporary files

    return Path(outdir/f'{outname}.pdb')
