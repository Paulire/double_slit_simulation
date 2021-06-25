#!/usr/bin/env ipython3

import meep as mp
from meep.materials import Al 
import numpy as np
from matplotlib import pyplot as plt

res = 50

#### User control variables ####
# Define slits
d = 3.0
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
padding = 5
sx = 2 + 2*dpml + wall_width
sy = 2*( dpml + padding ) + d + slit_width

cell = mp.Vector3( sx, sy )
pml_layer = [ mp.PML( thickness=dpml ) ]
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

# Define blocks for gratings
top_slit_cen = 0.5*( sy - dpml - padding )
bot_slit_cen = -top_slit_cen

middle_block_len = d - slit_width 

geometry = [ mp.Block( material=Al,
                       center=mp.Vector3( y=top_slit_cen ),
                       size=mp.Vector3( wall_width, padding + dpml, mp.inf )),
             mp.Block( material=Al,
                       center=mp.Vector3( y=bot_slit_cen ),
                       size=mp.Vector3( wall_width, padding + dpml, mp.inf )),
             mp.Block( material=Al,
                       center=mp.Vector3(  ),
                       size=mp.Vector3( wall_width, middle_block_len ))
           ]


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

# Fluxes
box_x_tpos = mp.FluxRegion( center=mp.Vector3( ( 0.5*wall_width - src_pos )/2, 0.5*sy-dpml ), 
                            size=mp.Vector3( -src_pos - 0.5*wall_width ) )
box_x_bpos = mp.FluxRegion( center=mp.Vector3( (0.5*wall_width + -src_pos)/2 , -0.5*sy+dpml ), 
                            size=mp.Vector3( -src_pos - 0.5*wall_width ) )
box_x_outpos = mp.FluxRegion( center=mp.Vector3( -src_pos ),   
                              size=mp.Vector3( y=sy-2*dpml ) )
box_x_top = sim.add_flux( frq_cen, dfrq, nfrq, box_x_tpos )
box_x_bot = sim.add_flux( frq_cen, dfrq, nfrq, box_x_bpos )
box_y_end = sim.add_flux( frq_cen, dfrq, nfrq, box_x_outpos )

# Run sim again
sim.run( until_after_sources=mp.stop_when_fields_decayed( 50, mp.Ez, n2f_point, 1e-5 ) )

# Collect far field data
ff_full = sim.get_farfields( n2f_obj, ff_res, center=mp.Vector3( ff_dist - src_pos ), size=mp.Vector3( y=ff_screen_size ) )

# Collect flux data
box_x_top_flux = mp.get_fluxes( box_x_top )
box_x_bot_flux = mp.get_fluxes( box_x_bot )
box_y_end_flux = mp.get_fluxes( box_y_end )

# Post processing data
ff_frq = mp.get_near2far_freqs( n2f_obj )
ff_points = np.linspace( -0.5*ff_screen_size, 0.5*ff_screen_size, ff_n_pnts )
anlge = [ np.degrees( np.arctan( i ) ) for i in ff_points/ff_dist ]

# Get point where 0.5 \mu m is
indx = np.where( np.array( ff_frq ) == 1/0.5 )[0][0]

# Get Fields
field = np.abs( ff_full['Ez'] )**2
norm = 1/np.max( field[:,indx] )

# Theoretical data
x = np.linspace( -ff_angle, ff_angle, 1000 )
f_x = np.pi*d*np.sin( np.deg2rad( x  ))/0.5
g_x = slit_width*np.sin( np.deg2rad( x  ))/0.5
y = (np.pi*np.sinc(g_x)**2)*np.cos( f_x )**2
nom_y = 1/np.max(y)

plt.plot( x, nom_y*y, '-r', label='Theoretical Model'   )
plt.plot(  anlge , norm*field[:,indx], 'ob', fillstyle='none', label='Meep Resault')
plt.xlabel( "Diffraction Angle (Degrees)", size="x-large" )
plt.ylabel( "Normalised Field Amplitude", size="x-large" )
plt.legend()
plt.tight_layout()
plt.savefig("img/al_slit_pattern.pdf", dpi=300)
plt.show()

# Process flux data
flux_frq = mp.get_flux_freqs( box_y_end )
indx = np.where( np.array( flux_frq ) == 1/0.5 )[0][0]

total_flux = np.abs( box_y_end_flux[indx] ) + np.abs( box_x_bot_flux[indx] ) + np.abs( box_x_top_flux[indx] )
pml_flux = np.abs( box_x_bot_flux[indx] ) + np.abs( box_x_top_flux[indx] )

print( "\n\n\tSide flux percent: " + str( 100*pml_flux/total_flux ) )
