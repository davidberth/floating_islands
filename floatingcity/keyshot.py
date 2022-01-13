# -*- coding: utf-8 -*-
# AUTHOR David
# VERSION 0.1
# Builds a fantasy floating city
print ('importing main scene')
lux_options = {'up_vector':2, 'import_type':2}
lux.importFile("c:/harvard/cityart/out/out.glb", opt=lux_options)
lux.renderImage('c:/harvard/cityart/out/keyout.png', width=2000, height=1000, format=lux.RENDER_OUTPUT_PNG)
print (' done')



