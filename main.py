#!/usr/bin/env ipython3

import meep as mp
from meep.materials import Al 
import numpy as np
from matplotlib import pyplot as plt

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
ff_screen_size = ff_dist*np.tan( np.deg2rad( ff_angle ) ) 

# Build simulation cell
dpml = 1
padding = 1
sx = 2 + 2*dpml + wall_width
sy = 2*( dpml + padding ) + d + 0.5*( wall_width + slit_width )
cell = mp.Vector3( sx, sy )
pml_layer = [ mp.PML( thickness=dpml ) ]

# Build final source values
frq_min = 1/wvl_max
frq_max = 1/wvl_min
frq_cen = 0.5*( frq_min + frq_max )
dfrq = frq_max - frq_min

src_pos = -0.5*slit_width - 0.5
source = [ mp.Source( mp.GaussianSource( frq_cen, dfrq ),
                      component=mp.Ez,
                      center=mp.Vector3( src_pos ),
                      size=mp.Vector3( y=sy-dpml*2 )) ]

# Define blocks for gratings
top_slit_cen = 0.5*( sy - padding )
bot_slit_cen = -top_slit_cen

middle_block_len = d - 2*slit_width

geometry = [ mp.Block( material=Al,
                       center=mp.Vector3( top_slit_cen ),
                       size=mp.Vector3( padding, wall_width, mp.inf )),
             mp.Block( material=Al,
                       center=mp.Vector3( bot_slit_cen ),
                       size=mp.Vector3( padding, wall_width, mp.inf )),
             mp.Block( material=Al,
                       center=mp.Vector3(  ),
                       size=mp.Vector3( middle_block_len, wall_width ))
           ]
