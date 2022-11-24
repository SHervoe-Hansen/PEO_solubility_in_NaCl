# Imports
import sys
import os
import openmm as mm
from openmm import app
from openmm import unit as u
from mdtraj.reporters import XTCReporter

print('Loading initial configuration and toplogy')
pdb = app.PDBFile('PEG_4_NaSCN.pdb')
forcefield = app.ForceField('/home/usr6/r70276b/PEO-Solubility/Force_fields/peg.xml',
                            '/home/usr6/r70276b/PEO-Solubility/Force_fields/spce.xml',
                            '/home/usr6/r70276b/PEO-Solubility/Force_fields/SCN.xml',
                            '/home/usr6/r70276b/PEO-Solubility/Force_fields/ions.xml')
    
# Find all PEG atoms   
PEG_atoms = []
for residue in pdb.topology.residues():
    if residue.name in ['PGH', 'PGM', 'PGT']:
        for atom in residue.atoms():
            PEG_atoms.append(atom)

# Creating system
print('Creating OpenMM System')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, ewaldErrorTolerance=0.0005, switchDistance=1*u.nanometer,
                                 nonbondedCutoff=1.2*u.nanometers, constraints=app.HBonds, rigidWater=True)

# Calculating total mass of system
total_mass = u.sum([system.getParticleMass(i) for i in range(system.getNumParticles())])

# Freeze PEG atoms by setting mass to 0
for atom in PEG_atoms:
    system.setParticleMass(atom.index, 0.000*u.dalton)
        
# Temperature-coupling by leap frog "middle"-discretization Langevin integrator (NVT)
integrator = mm.LangevinMiddleIntegrator(0.15*u.kelvin, 100.0/u.picoseconds, 0.05*u.femtoseconds)

# Pressure-coupling by a Monte Carlo Barostat (NPT)
system.addForce(mm.MonteCarloBarostat(1*u.bar, 0.15*u.kelvin, 25))

platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': '0'}

# Create the Simulation object
sim = app.Simulation(pdb.topology, system, integrator, platform, properties)

# Set the particle positions
sim.context.setPositions(pdb.positions)

# Minimize the energy
print('Minimizing energy using Langevin dynamics at low temperature and high drag...')
sim.step(500000)  # 500000*0.05 fs = 25 ps

# Save minimized coordinates
positions = sim.context.getState(getPositions=True).getPositions()

# Change Langevin integrator and Monte Carlo barostat back to correct parameters
sim.integrator.setFriction(1.0/u.picoseconds)
sim.integrator.setStepSize(2.0*u.femtoseconds)
sim.integrator.setTemperature(298.15*u.kelvin)
for param in sim.context.getParameters():
    if 'MonteCarloTemperature' in param:
        sim.context.setParameter(param, 298.15*u.kelvin)
sim.context.setTime(0)
sim.context.reinitialize()
sim.context.setPositions(positions)

# Draw new MB velocities
sim.context.setVelocitiesToTemperature(298.15*u.kelvin)

# Equlibrate simulation
print('Equilibrating...')
sim.step(500000)  # 500000*2 fs = 1.0 ns

# Set up the reporters
sim.reporters.append(app.StateDataReporter('output.dat', 1000, totalSteps=50000000+500000,
    time=True, potentialEnergy=True, kineticEnergy=True, temperature=True, volume=True, density=True,
    systemMass=total_mass, remainingTime=True, speed=True, separator='	'))

# Set up trajectory reporter
sim.reporters.append(XTCReporter('trajectory.xtc', reportInterval=1000, append=False))

# Run dynamics
print('Running dynamics! (NPT)')
sim.step(50000000)

# Print PME information
print('''
PARTICLE MESH EWALD PARAMETERS
Separation parameter: {}
Number of grid points along the X axis: {}
Number of grid points along the Y axis: {}
Number of grid points along the Z axis: {}
'''.format(*sim.system.getForces()[3].getPMEParametersInContext(sim.context)))
