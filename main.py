#!/usr/bin/env ipython3

import meep as mp
from meep.materials import Al 
import numpy as np
from matplotlib import pyplot as plt

res = 50

#### User control variables ####
# Define slits
d = 3
slit_width = 0.5
wall_width = 0.5

# Define source values
wvl_min = 0.3
wvl_max = 0.7

#### System interprate variables ####
# Far field value
ff_dist = 200
ff_angle = 20 # degrees
ff_screen_size = 2*ff_dist*np.tan( np.deg2rad( ff_angle ) ) 
ff_n_pnts = 500
ff_res = ff_n_pnts/ff_screen_size

# Build simulation cell
dpml = 1
padding = 1
sx = 2 + 2*dpml + wall_width
sy = 2*( dpml + padding ) + d + 0.5*( wall_width + slit_width )

cell = mp.Vector3( sx, sy )
cell_1D = mp.Vector3( sx )
pml_layer = [ mp.PML( thickness=dpml ) ]
pml_layer_1D = [ mp.PML( thickness=dpml, direction=mp.Y ) ]
symmetries = [ mp.Mirror( mp.Y ) ]

# Build final source values
frq_min = 1/wvl_max
frq_max = 1/wvl_min
frq_cen = 0.5*( frq_min + frq_max )
dfrq = frq_max - frq_min
nfrq = 11

src_pos = -0.5*slit_width - 0.5
source = [ mp.Source( mp.GaussianSource( frq_cen, dfrq, is_integrated=True),
                      component=mp.Ez,
                      center=mp.Vector3( src_pos ),
                      size=mp.Vector3( y=sy )) ]
source_1D = [ mp.Source( mp.GaussianSource( frq_cen, dfrq, is_integrated=True ),
                      component=mp.Ez,
                      center=mp.Vector3( src_pos ) ) ]

# Define blocks for gratings
top_slit_cen = 0.5*( sy - padding )
bot_slit_cen = -top_slit_cen

middle_block_len = d - 2*slit_width

geometry = [ mp.Block( material=Al,
                       center=mp.Vector3( y=top_slit_cen ),
                       size=mp.Vector3( wall_width, padding, mp.inf )),
             mp.Block( material=Al,
                       center=mp.Vector3( y=bot_slit_cen ),
                       size=mp.Vector3( wall_width, padding, mp.inf )),
             mp.Block( material=Al,
                       center=mp.Vector3(  ),
                       size=mp.Vector3( wall_width, middle_block_len ))
           ]


# 1D simulation without geometry
sim = mp.Simulation( cell_size = cell_1D,
                     resolution=res,
                     sources=source_1D,
                     boundary_layers=pml_layer_1D,
                     default_material=Al)

# Add near to far
n2f_point = mp.Vector3( -src_pos )      # The Near2Far is on the opisite side to the source
n2f_obj = sim.add_near2far( frq_cen, dfrq, nfrq, mp.Near2FarRegion( center=n2f_point))

# Run until n2f_point has dropped to a billionth of it's peak value
sim.run( until_after_sources=mp.stop_when_fields_decayed( 50, mp.Ez, n2f_point, 1e-8 ) )

# Get far field data
ff_empty = sim.get_farfields( n2f_obj, ff_res, center=mp.Vector3( ff_dist ), size=mp.Vector3( y=ff_screen_size ) )

sim.reset_meep()

# Grated simulation
sim = mp.Simulation( cell_size = cell,
                     resolution=res,
                     geometry=geometry,
                     sources=source,
                     boundary_layers=pml_layer,
                     symmetries=symmetries )

# Add near to far
n2f_point = mp.Vector3( -src_pos )      # The Near2Far is on the opisite side to the source
n2f_obj = sim.add_near2far( frq_cen, dfrq, nfrq, mp.Near2FarRegion( center=n2f_point,
                                                                    size=mp.Vector3( y=sy )))


sim.run( until_after_sources=mp.stop_when_fields_decayed( 50, mp.Ez, n2f_point, 1e-8 ) )

ff_full = sim.get_farfields( n2f_obj, ff_res, center=mp.Vector3( ff_dist ), size=mp.Vector3( y=ff_screen_size ) )

# Post processing data
ff_frq = mp.get_near2far_freqs( n2f_obj )
ff_points = np.linspace( -0.5*ff_screen_size, 0.5*ff_screen_size, ff_n_pnts )
anlge = [ np.degrees( np.arctan( i ) ) for i in ff_points/ff_dist ]

# Get Fields
field = np.abs( ff_full['Ez']**2/ff_empty['Ez']**2 )

# Get point where 0.5 \mu m is
indx = np.where( np.array( ff_frq ) == 1/0.5 )[0][0]

plt.plot( anlge, field[:,indx] )
plt.show()
