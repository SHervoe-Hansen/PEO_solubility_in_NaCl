# Imports
import sys
import os
import openmm as mm
from openmm import app
from openmm import unit as u
from mdtraj.reporters import XTCReporter

print('Loading initial configuration and toplogy')
pdb = app.PDBFile('/work/1/ERCINCR/stefan/PEO-Solubility/PDB_files/PEO-8-mer.pdb')
forcefield = app.ForceField('/work/1/ERCINCR/stefan/PEO-Solubility/Force_fields/peg.xml',
                            '/work/1/ERCINCR/stefan/PEO-Solubility/Force_fields/spce.xml',
                            '/work/1/ERCINCR/stefan/PEO-Solubility/Force_fields/SCN.xml',
                            '/work/1/ERCINCR/stefan/PEO-Solubility/Force_fields/ions.xml')
                         
modeller = app.Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(forcefield, pH=7.0)
modeller.addSolvent(forcefield, model='spce', numAdded=10000, neutralize=False)

app.PDBFile.writeFile(modeller.topology, modeller.positions, open('PEG_8_hydrated.pdb', 'w'))

# Creating system
print('Creating OpenMM System')
system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, ewaldErrorTolerance=0.0005, switchDistance=1*u.nanometer,
                                 nonbondedCutoff=1.2*u.nanometers, constraints=app.HBonds, rigidWater=True)

# Calculating total mass of system
total_mass = u.sum([system.getParticleMass(i) for i in range(system.getNumParticles())])
        
# Temperature-coupling by leap frog (BAOAB) Langevin integrator (NVT)
integrator = mm.LangevinMiddleIntegrator(298.15*u.kelvin, 1.0/u.picoseconds, 2.0*u.femtoseconds)

# Pressure-coupling by a Monte Carlo Barostat (NPT)
system.addForce(mm.MonteCarloBarostat(1*u.bar, 298.15*u.kelvin, 25))

platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': '0,1,2,3'}

# Create the Simulation object
sim = app.Simulation(modeller.topology, system, integrator, platform, properties)

# Set the particle positions
sim.context.setPositions(modeller.positions)

# Minimize the energy
print('Minimizing energy')
sim.minimizeEnergy(tolerance=1*u.kilojoule/u.mole, maxIterations=1000000)
    
# Draw initial MB velocities
sim.context.setVelocitiesToTemperature(298.15*u.kelvin)

# Equlibrate simulation
print('Equilibrating...')
sim.step(500000)  # 500000*2 fs = 1.0 ns

# Set up the reporters
sim.reporters.append(app.StateDataReporter('output.dat', 5000, totalSteps=100000000+250000,
    time=True, potentialEnergy=True, kineticEnergy=True, temperature=True, volume=True, density=True,
    systemMass=total_mass, remainingTime=True, speed=True, separator='	'))

# Set up trajectory reporter
sim.reporters.append(XTCReporter('trajectory.xtc', reportInterval=5000, append=False))

# Run dynamics
print('Running dynamics! (NPT)')
sim.step(100000000)
